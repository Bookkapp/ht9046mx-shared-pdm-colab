from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .anomaly import StandardScaler3D, reconstruction_error
from .config import PipelineConfig
from .features import engineer_features
from .model import build_lstm_autoencoder, require_tensorflow
from .preprocessing import load_and_prepare
from .windowing import chronological_split, make_windows


def train(paths: list[str], machine_id: str, module_id: int, output_dir: str, config: PipelineConfig, model_version: str = "lstm_autoencoder_v1", max_windows: int | None = None) -> dict:
    if module_id in config.excluded_modules:
        raise ValueError(f"Module {module_id} is excluded by configuration; confirm commissioning before training it")
    valid, rejected = load_and_prepare(paths, machine_id, config, module_id)
    if valid.empty:
        raise ValueError("No stable normal rows after state filtering; choose another date/module or review machine state")
    featured = engineer_features(valid, config)
    windows, metadata = make_windows(featured, config)
    if max_windows is not None and len(windows) > max_windows:
        indices = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
        windows, metadata = windows[indices], metadata.iloc[indices].reset_index(drop=True)
    (train_x, train_meta), (valid_x, valid_meta), (test_x, test_meta) = chronological_split(windows, metadata)
    scaler = StandardScaler3D().fit(train_x)
    train_scaled, valid_scaled, test_scaled = (scaler.transform(item) for item in (train_x, valid_x, test_x))
    tf = require_tensorflow()
    model = build_lstm_autoencoder(config, train_scaled.shape[-1])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    history = model.fit(train_scaled, train_scaled, validation_data=(valid_scaled, valid_scaled), epochs=config.epochs, batch_size=config.batch_size, shuffle=False, verbose=2, callbacks=callbacks)
    valid_pred = model.predict(valid_scaled, verbose=0)
    validation_errors, _ = reconstruction_error(valid_scaled, valid_pred)
    threshold = float(np.percentile(validation_errors, config.threshold_percentile))
    test_pred = model.predict(test_scaled, verbose=0)
    test_errors, _ = reconstruction_error(test_scaled, test_pred)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    scaler.save(str(out / "scaler.npz"))
    config.save(out / "config.json")
    metadata_out = {
        "model_version": model_version,
        "machine_id": machine_id,
        "module_id": module_id,
        "source_files": paths,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(config.feature_columns),
        "window_rows": config.window_rows,
        "step_rows": config.step_rows,
        "threshold": {"method": "percentile", "percentile": config.threshold_percentile, "value": threshold},
        "threshold_version": f"{model_version}_p{config.threshold_percentile:g}",
        "quality": {"accepted_rows": int(len(valid)), "rejected_rows": int(len(rejected)), "windows": {"train": int(len(train_x)), "validation": int(len(valid_x)), "test": int(len(test_x))}},
        "metrics": {"validation_mae_p50": float(np.median(validation_errors)), "validation_mae_p95": float(np.percentile(validation_errors, 95)), "test_mae_p50": float(np.median(test_errors)), "test_exceedance_rate": float((test_errors > threshold).mean())},
        "training": {"epochs_completed": len(history.history["loss"]), "final_loss": float(history.history["loss"][-1]), "final_validation_loss": float(history.history["val_loss"][-1])},
    }
    (out / "metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    test_meta.assign(reconstruction_error=test_errors).to_csv(out / "test_window_results.csv", index=False)
    return metadata_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train state-aware HT-9046MX LSTM Autoencoder for one module")
    parser.add_argument("--input", nargs="+", required=True, help="Daily CSV/TXT handler logs in chronological order")
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--module-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", help="Optional JSON configuration")
    parser.add_argument("--model-version", default="lstm_autoencoder_v1")
    parser.add_argument("--max-windows", type=int, help="Optional evenly spaced cap for a smoke test; do not use for production training")
    args = parser.parse_args()
    result = train(args.input, args.machine_id, args.module_id, args.output_dir, PipelineConfig.load(args.config), args.model_version, args.max_windows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
