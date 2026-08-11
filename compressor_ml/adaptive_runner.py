from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .adaptive import (
    AdaptiveConfig,
    AdaptiveRuntime,
    CalibrationBuffer,
    CalibrationProfile,
    build_reference_profile,
    score_profile,
    utc_now,
    window_summaries,
)
from .anomaly import StandardScaler3D, reconstruction_error
from .features import engineer_features
from .model import require_tensorflow
from .prepare_dataset import discover_daily_files, load_prepared_dataset, safe_group_name
from .preprocessing import read_handler_log, validate_and_filter
from .windowing import make_windows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class SharedArtifactBundle:
    """Load the Colab shared model plus immutable train-only group scalers."""

    def __init__(self, artifact_dir: str | Path, model: Any | None = None) -> None:
        self.root = Path(artifact_dir)
        from .config import PipelineConfig

        self.config = PipelineConfig.load(self.root / "config.json")
        self.manifest = _read_json(self.root / "manifest.json")
        self.thresholds = _read_json(self.root / "thresholds.json")
        self.model_version = str(self.manifest.get("model_version", "shared_lstm_unknown"))
        if model is None:
            tf = require_tensorflow()
            model = tf.keras.models.load_model(self.root / "shared_model.keras")
        self.model = model
        self._scalers: dict[str, StandardScaler3D] = {}

    def has_group(self, group_name: str) -> bool:
        return group_name in self.thresholds and (self.root / "scalers" / f"{group_name}.npz").exists()

    def scaler(self, group_name: str) -> StandardScaler3D:
        if group_name not in self._scalers:
            self._scalers[group_name] = StandardScaler3D.load(
                str(self.root / "scalers" / f"{group_name}.npz")
            )
        return self._scalers[group_name]

    def threshold(self, group_name: str) -> float:
        return float(self.thresholds[group_name]["value"])

    def reconstruct(
        self,
        group_name: str,
        raw_windows: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.scaler(group_name).transform(raw_windows)
        predicted = self.model.predict(scaled, batch_size=batch_size, verbose=0)
        return reconstruction_error(scaled, predicted)


def bootstrap_from_prepared(
    prepared_dir: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    adaptive_config: AdaptiveConfig | None = None,
    model: Any | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Create immutable golden profiles and frozen holdouts after Colab training."""

    config = adaptive_config or AdaptiveConfig()
    config.validate()
    pipeline_config, dataset_manifest, datasets = load_prepared_dataset(prepared_dir)
    bundle = SharedArtifactBundle(artifact_dir, model=model)
    if list(pipeline_config.feature_columns) != list(bundle.config.feature_columns):
        raise ValueError("Prepared dataset and model artifact feature lists differ")

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Adaptive seed output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config.save(destination / "adaptive_config.json")
    profile_rows: list[dict[str, Any]] = []

    for (machine_id, module_id), dataset in datasets.items():
        group_name = safe_group_name(machine_id, module_id)
        if not bundle.has_group(group_name):
            raise KeyError(f"Shared artifact is missing scaler/threshold for {group_name}")
        validation_errors, _ = bundle.reconstruct(
            group_name, dataset["validation"], batch_size=batch_size
        )
        test_errors, _ = bundle.reconstruct(
            group_name, dataset["test"], batch_size=batch_size
        )
        profile = build_reference_profile(
            group_name=group_name,
            machine_id=machine_id,
            module_id=module_id,
            feature_columns=pipeline_config.feature_columns,
            model_version=bundle.model_version,
            train_windows=dataset["train"],
            reconstruction_threshold=bundle.threshold(group_name),
            config=config,
        )
        group_dir = destination / "profiles" / group_name
        profile.save(group_dir / "golden.json")
        champion_payload = asdict(profile)
        champion_payload.update(
            {
                "profile_version": f"{group_name}_champion_v1",
                "status": "CHAMPION",
                "parent_version": profile.profile_version,
                "created_at_utc": utc_now(),
            }
        )
        champion = CalibrationProfile(**champion_payload)

        frozen_windows = np.concatenate([dataset["validation"], dataset["test"]])
        frozen_errors = np.concatenate([validation_errors, test_errors])
        frozen_summaries = window_summaries(frozen_windows)
        frozen_result = score_profile(champion, frozen_summaries, frozen_errors, config)
        synthetic_summaries = frozen_summaries.copy()
        sensor_count = min(7, synthetic_summaries.shape[1])
        golden_scale = np.asarray(champion.golden_scale, dtype=np.float64)
        for index in range(len(synthetic_summaries)):
            feature_index = index % sensor_count
            synthetic_summaries[index, feature_index] += (
                config.synthetic_shift_mad * golden_scale[feature_index]
            )
        synthetic_result = score_profile(
            champion, synthetic_summaries, frozen_errors, config
        )
        champion.approval = {
            "outcome": "BOOTSTRAP_APPROVED",
            "deployment_baseline": {
                "reference_alert_rate": float(
                    np.mean(frozen_result["operational_risk"] >= 1.0)
                ),
                "synthetic_detection_rate": float(
                    np.mean(synthetic_result["operational_risk"] >= 1.0)
                ),
            },
            "caveat": "unsupervised calibration validation; fault accuracy requires labels",
        }
        champion.save(group_dir / "champion.json")
        frozen_path = destination / "frozen" / f"{group_name}.npz"
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            frozen_path,
            summaries=frozen_summaries.astype(np.float32),
            errors=frozen_errors.astype(np.float32),
        )
        profile_rows.append(
            {
                "group_name": group_name,
                "machine_id": machine_id,
                "module_id": int(module_id),
                "train_windows": int(len(dataset["train"])),
                "frozen_windows": int(len(frozen_windows)),
                "golden_reconstruction_threshold": profile.golden_reconstruction_threshold,
            }
        )

    seed_manifest = {
        "format": "ht9046mx_adaptive_seed_v1",
        "created_at_utc": utc_now(),
        "model_version": bundle.model_version,
        "prepared_dataset_version": dataset_manifest["dataset_version"],
        "feature_columns": list(pipeline_config.feature_columns),
        "groups": profile_rows,
        "safety_contract": {
            "shared_model_weights": "immutable",
            "train_only_input_scalers": "immutable",
            "golden_profiles": "immutable",
            "automatic_updates": "operational_calibration_only",
            "fault_accuracy_validation": "requires_maintenance_linked_labels",
        },
    }
    _write_json(destination / "seed_manifest.json", seed_manifest)
    return {
        "output_dir": str(destination.resolve()),
        "groups": len(profile_rows),
        "model_version": bundle.model_version,
        "prepared_dataset_version": dataset_manifest["dataset_version"],
    }


class ProcessedFileRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.payload = _read_json(self.path)
        else:
            self.payload = {"files": {}, "initialized_machines": {}, "updated_at_utc": None}

    @staticmethod
    def identity(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }

    @staticmethod
    def key(path: Path) -> str:
        return str(path.resolve()).lower()

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def is_processed(self, path: Path) -> bool:
        record = self.payload["files"].get(self.key(path))
        if not record:
            return False
        identity = self.identity(path)
        return (
            int(record.get("size_bytes", -1)) == identity["size_bytes"]
            and int(record.get("modified_ns", -1)) == identity["modified_ns"]
        )

    def mark(
        self,
        path: Path,
        machine_id: str,
        status: str,
        details: dict[str, Any] | None = None,
        *,
        include_hash: bool = True,
        sha256_value: str | None = None,
    ) -> None:
        record = self.identity(path)
        record.update(
            {
                "machine_id": machine_id,
                "status": status,
                "sha256": sha256_value
                if sha256_value is not None
                else (self.sha256(path) if include_hash else None),
                "processed_at_utc": utc_now(),
                "details": details or {},
            }
        )
        self.payload["files"][self.key(path)] = record

    def initialize_machine(self, machine_id: str, files: list[Path], initial_count: int) -> list[Path]:
        if self.payload["initialized_machines"].get(machine_id):
            return files
        initial_count = max(1, int(initial_count))
        to_process = files[-initial_count:]
        for path in files[:-initial_count]:
            self.mark(path, machine_id, "baseline_skipped", include_hash=False)
        self.payload["initialized_machines"][machine_id] = {
            "initialized_at_utc": utc_now(),
            "historical_files_skipped": max(0, len(files) - len(to_process)),
        }
        return to_process

    def save(self) -> None:
        self.payload["updated_at_utc"] = utc_now()
        _write_json(self.path, self.payload)


def _append_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _score_group_windows(
    bundle: SharedArtifactBundle,
    runtime: AdaptiveRuntime,
    group_name: str,
    windows: np.ndarray,
    metadata: pd.DataFrame,
    source_file: Path,
    source_sha256: str,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    champion = runtime.load_champion(group_name)
    errors, feature_errors = bundle.reconstruct(group_name, windows, batch_size=batch_size)
    summaries = window_summaries(windows)
    scored = score_profile(champion, summaries, errors, runtime.config)
    top_indices = np.argmax(feature_errors, axis=1)
    result = metadata.copy().reset_index(drop=True)
    result["source_file"] = str(source_file.resolve())
    result["source_sha256"] = source_sha256
    result["reconstruction_error"] = errors
    for name in (
        "golden_risk",
        "operational_risk",
        "combined_risk",
        "health_score",
        "condition_status",
        "baseline_drift",
        "eligible_for_calibration",
        "golden_feature_deviation",
        "golden_relation_distance",
        "adaptive_feature_deviation",
        "adaptive_relation_distance",
    ):
        result[name] = scored[name]
    result["top_error_feature"] = [champion.feature_columns[index] for index in top_indices]
    result["top_feature_error"] = feature_errors[np.arange(len(feature_errors)), top_indices]
    result["model_version"] = champion.model_version
    result["profile_version"] = champion.profile_version
    result["golden_reconstruction_threshold"] = champion.golden_reconstruction_threshold
    result["adaptive_reconstruction_threshold"] = champion.adaptive_reconstruction_threshold
    eligible = np.asarray(scored["eligible_for_calibration"], dtype=bool)
    return result, summaries[eligible], errors[eligible], eligible


def _discover_cycle_files(
    machine_id: str,
    source: Path,
    registry: ProcessedFileRegistry,
    initial_files_per_machine: int,
    max_files_per_cycle: int,
    process_history: bool,
) -> list[Path]:
    files = discover_daily_files(source)
    if not files:
        return []
    if not process_history:
        files = registry.initialize_machine(machine_id, files, initial_files_per_machine)
    unprocessed = [path for path in files if not registry.is_processed(path)]
    return unprocessed[:max_files_per_cycle]


def run_cycle(system_config_path: str | Path, *, model: Any | None = None) -> dict[str, Any]:
    system_path = Path(system_config_path)
    system = _read_json(system_path)
    artifact_dir = Path(system["artifact_dir"])
    runtime_dir = Path(system.get("runtime_dir", "adaptive_runtime"))
    seed_dir = artifact_dir / "adaptive_seed"
    runtime = AdaptiveRuntime(runtime_dir)
    if not runtime.root.exists() or not (runtime.root / "profiles").exists():
        runtime.initialize_from_seed(seed_dir)
    bundle = SharedArtifactBundle(artifact_dir, model=model)
    registry = ProcessedFileRegistry(runtime.root / "state" / "processed_files.json")
    modules = [int(value) for value in system.get("modules", [1, 2, 3, 4, 5, 6, 8])]
    batch_size = int(system.get("batch_size", 256))
    max_files = int(system.get("max_files_per_cycle", 10))
    initial_files = int(system.get("initial_files_per_machine", 1))
    process_history = bool(system.get("process_history", False))

    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    processed_files = 0
    scored_windows = 0
    eligible_windows = 0
    touched_groups: set[str] = set()
    quality_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    champion_self_tests: dict[str, dict[str, Any]] = {}

    for group_dir in sorted(path for path in runtime.groups_dir.iterdir() if path.is_dir()):
        decision = runtime.self_test_champion(group_dir.name)
        champion_self_tests[group_dir.name] = decision.to_dict()

    for machine_id, source_value in system["machine_sources"].items():
        source = Path(source_value)
        files = _discover_cycle_files(
            machine_id,
            source,
            registry,
            initial_files,
            max_files,
            process_history,
        )
        for path in files:
            pending_results: list[tuple[str, pd.DataFrame]] = []
            pending_buffers: list[tuple[str, np.ndarray, np.ndarray, list[str]]] = []
            pending_quality: list[dict[str, Any]] = []
            try:
                source_sha256 = registry.sha256(path)
                raw = read_handler_log(path, machine_id)
                for module_id in modules:
                    group_name = safe_group_name(machine_id, module_id)
                    if not bundle.has_group(group_name) or not runtime.champion_path(group_name).exists():
                        continue
                    module_raw = raw.loc[raw["module_id"].eq(module_id)].copy()
                    valid, rejected = validate_and_filter(module_raw, bundle.config)
                    featured = engineer_features(valid, bundle.config)
                    windows, metadata = make_windows(featured, bundle.config)
                    quality = {
                        "cycle_id": cycle_id,
                        "machine_id": machine_id,
                        "module_id": module_id,
                        "group_name": group_name,
                        "source_file": str(path.resolve()),
                        "raw_rows": int(len(module_raw)),
                        "stable_rows": int(len(valid)),
                        "rejected_rows": int(len(rejected)),
                        "complete_windows": int(len(windows)),
                        "imputed_rows": int(valid.get("is_imputed_short_gap", pd.Series(dtype=bool)).sum()),
                    }
                    if len(windows) == 0:
                        quality.update({"scored_windows": 0, "eligible_windows": 0})
                        pending_quality.append(quality)
                        continue
                    result, eligible_summaries, eligible_errors, eligible_mask = _score_group_windows(
                        bundle,
                        runtime,
                        group_name,
                        windows,
                        metadata,
                        path,
                        source_sha256,
                        batch_size,
                    )
                    eligible_timestamps = result.loc[eligible_mask, "timestamp"].astype(str).tolist()
                    quality.update(
                        {
                            "scored_windows": int(len(result)),
                            "eligible_windows": int(len(eligible_summaries)),
                            "baseline_drift_windows": int(result["baseline_drift"].sum()),
                            "warning_or_critical_windows": int(
                                result["condition_status"].isin(["Warning", "Critical"]).sum()
                            ),
                        }
                    )
                    pending_results.append((group_name, result))
                    pending_buffers.append(
                        (group_name, eligible_summaries, eligible_errors, eligible_timestamps)
                    )
                    pending_quality.append(quality)

                for group_name, result in pending_results:
                    _append_frame(runtime.root / "predictions" / f"{group_name}.csv", result)
                    scored_windows += len(result)
                    touched_groups.add(group_name)
                for group_name, summaries, group_errors, timestamps in pending_buffers:
                    runtime.append_eligible(group_name, summaries, group_errors, timestamps)
                    eligible_windows += len(summaries)
                quality_rows.extend(pending_quality)
                registry.mark(
                    path,
                    machine_id,
                    "processed",
                    {
                        "groups": len(pending_results),
                        "scored_windows": int(sum(len(frame) for _, frame in pending_results)),
                    },
                    sha256_value=source_sha256,
                )
                processed_files += 1
            except Exception as exc:  # keep the cycle alive for other machines
                errors.append(
                    {
                        "machine_id": machine_id,
                        "source_file": str(path.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    registry.save()
    if quality_rows:
        _append_frame(runtime.root / "monitoring" / "data_quality.csv", pd.DataFrame(quality_rows))
    if errors:
        _append_frame(runtime.root / "monitoring" / "cycle_errors.csv", pd.DataFrame(errors))

    decisions: dict[str, dict[str, Any]] = {}
    for group_name in sorted(touched_groups):
        decision = runtime.propose_or_advance(group_name)
        decisions[group_name] = decision.to_dict()

    summary = {
        "cycle_id": cycle_id,
        "completed_at_utc": utc_now(),
        "processed_files": processed_files,
        "scored_windows": int(scored_windows),
        "eligible_windows": int(eligible_windows),
        "groups_touched": len(touched_groups),
        "champion_self_tests": champion_self_tests,
        "approval_decisions": decisions,
        "errors": errors,
    }
    _write_json(runtime.root / "runs" / f"cycle_{cycle_id}.json", summary)
    _write_json(runtime.root / "latest_cycle.json", summary)
    return summary


def runtime_status(runtime_dir: str | Path) -> dict[str, Any]:
    runtime = AdaptiveRuntime(runtime_dir)
    groups: list[dict[str, Any]] = []
    if not runtime.groups_dir.exists():
        return {"runtime_dir": str(runtime.root.resolve()), "groups": [], "status": "not_initialized"}
    for group_dir in sorted(path for path in runtime.groups_dir.iterdir() if path.is_dir()):
        champion = CalibrationProfile.load(group_dir / "champion.json")
        candidate_path = group_dir / "candidate.json"
        buffer = CalibrationBuffer.load(runtime.buffer_path(group_dir.name), len(champion.feature_columns))
        groups.append(
            {
                "group_name": group_dir.name,
                "champion_version": champion.profile_version,
                "champion_status": champion.status,
                "candidate_version": CalibrationProfile.load(candidate_path).profile_version
                if candidate_path.exists()
                else None,
                "buffer_windows": len(buffer.summaries),
                "buffer_total_seen": buffer.total_seen,
            }
        )
    return {"runtime_dir": str(runtime.root.resolve()), "groups": groups, "status": "ready"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HT-9046MX adaptive scoring, calibration validation, and approval"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Create golden profiles after shared-model training")
    bootstrap_parser.add_argument("--prepared-dir", required=True)
    bootstrap_parser.add_argument("--artifact-dir", required=True)
    bootstrap_parser.add_argument("--output-dir", required=True)
    bootstrap_parser.add_argument("--adaptive-config")
    bootstrap_parser.add_argument("--batch-size", type=int, default=256)

    init_parser = subparsers.add_parser("init", help="Initialize a writable runtime from a Colab seed")
    init_parser.add_argument("--seed-dir", required=True)
    init_parser.add_argument("--runtime-dir", default="adaptive_runtime")

    cycle_parser = subparsers.add_parser("cycle", help="Score new logs and advance safe calibration candidates")
    cycle_parser.add_argument("--system-config", default="configs/adaptive_system.json")

    status_parser = subparsers.add_parser("status", help="Show champion/candidate status")
    status_parser.add_argument("--runtime-dir", default="adaptive_runtime")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    if args.command == "bootstrap":
        result = bootstrap_from_prepared(
            args.prepared_dir,
            args.artifact_dir,
            args.output_dir,
            adaptive_config=AdaptiveConfig.load(args.adaptive_config),
            batch_size=args.batch_size,
        )
    elif args.command == "init":
        runtime = AdaptiveRuntime(args.runtime_dir)
        runtime.initialize_from_seed(args.seed_dir)
        result = runtime_status(args.runtime_dir)
    elif args.command == "cycle":
        result = run_cycle(args.system_config)
    else:
        result = runtime_status(args.runtime_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
