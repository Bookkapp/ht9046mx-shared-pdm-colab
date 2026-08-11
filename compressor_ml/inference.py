from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly import StandardScaler3D, anomaly_score, health_score, pseudo_label, reconstruction_error
from .config import PipelineConfig
from .features import engineer_features
from .model import require_tensorflow
from .preprocessing import load_and_prepare
from .windowing import make_windows


def load_artifacts(artifact_dir: str | Path):
    directory = Path(artifact_dir)
    config = PipelineConfig.load(directory / "config.json")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["feature_columns"] != list(config.feature_columns):
        raise ValueError("Artifact feature list does not match config.json")
    tf = require_tensorflow()
    return config, metadata, StandardScaler3D.load(str(directory / "scaler.npz")), tf.keras.models.load_model(directory / "model.keras")


def score_prepared(frame: pd.DataFrame, artifact_dir: str | Path) -> pd.DataFrame:
    config, metadata, scaler, model = load_artifacts(artifact_dir)
    featured = engineer_features(frame, config)
    windows, window_meta = make_windows(featured, config)
    if len(windows) == 0:
        return pd.DataFrame()
    scaled = scaler.transform(windows)
    reconstructed = model.predict(scaled, verbose=0)
    errors, feature_errors = reconstruction_error(scaled, reconstructed)
    scores = anomaly_score(errors, float(metadata["threshold"]["value"]))
    result = window_meta.copy()
    result["reconstruction_error"] = errors
    result["anomaly_score"] = scores
    result["model_version"] = metadata["model_version"]
    result["threshold_version"] = metadata["threshold_version"]
    # Do not smooth through a state/gap boundary: each segment represents a
    # different stable operating episode.
    result["health_score"] = result.groupby(["machine_id", "module_id", "segment_id"], sort=False)["anomaly_score"].transform(
        lambda values: health_score(values.to_numpy(), config.health_smoothing_windows)
    )
    result["condition_status"] = [
        pseudo_label(float(value), config.normal_min, config.watch_min, config.warning_min)
        for value in result["health_score"]
    ]
    result["feature_errors"] = [json.dumps(dict(zip(config.feature_columns, row.tolist()))) for row in feature_errors]
    return result


def predict_condition(paths: list[str], machine_id: str, module_id: int, artifact_dir: str | Path) -> dict:
    config, _, _, _ = load_artifacts(artifact_dir)
    valid, _ = load_and_prepare(paths, machine_id, config, module_id)
    results = score_prepared(valid, artifact_dir)
    if results.empty:
        raise ValueError("No valid complete window after state filtering")
    return results.iloc[-1].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run state-aware compressor condition inference")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--module-id", required=True, type=int)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", help="Optional CSV output of all scored windows")
    args = parser.parse_args()
    config, _, _, _ = load_artifacts(args.artifact_dir)
    valid, _ = load_and_prepare(args.input, args.machine_id, config, args.module_id)
    result = score_prepared(valid, args.artifact_dir)
    if args.output:
        result.to_csv(args.output, index=False)
    print(json.dumps(result.iloc[-1].to_dict() if not result.empty else {"status": "no_valid_window"}, indent=2, default=str))


if __name__ == "__main__":
    main()
