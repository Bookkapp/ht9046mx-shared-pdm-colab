from __future__ import annotations

from collections import defaultdict
import csv
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .catalog import METRIC_BY_KEY
from .handler_store import list_handlers, normalize_machine
from .settings import settings


if str(settings.model_project_root) not in sys.path:
    sys.path.insert(0, str(settings.model_project_root))

from compressor_ml.controlled_monitoring.config import ControlledMonitoringConfig
from compressor_ml.controlled_monitoring.lifecycle import (
    LifecycleState,
    ProfileRepository,
)
from compressor_ml.controlled_monitoring.windowing import build_event_windows
from compressor_ml.prepare_dataset import discover_daily_files, safe_group_name
from compressor_ml.preprocessing import read_handler_log


def _json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _tail_jsonl(path: Path, max_lines: int) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    # A model decision is normally below 4 KiB. Cap the read so a long-running
    # server does not scan the entire append-only file on every dashboard poll.
    read_bytes = min(path.stat().st_size, max(1_048_576, max_lines * 8_192))
    with path.open("rb") as handle:
        handle.seek(-read_bytes, 2)
        payload = handle.read().decode("utf-8", errors="replace")
    lines = payload.splitlines()
    if read_bytes < path.stat().st_size and lines:
        lines = lines[1:]
    output: list[dict[str, Any]] = []
    for line in lines[-max_lines:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            output.append(item)
    return output


@lru_cache(maxsize=64)
def _prediction_cache(
    path_text: str, size: int, modified_ns: int, max_lines: int
) -> tuple[dict[str, Any], ...]:
    del size, modified_ns
    return tuple(_tail_jsonl(Path(path_text), max_lines))


@lru_cache(maxsize=12)
def _window_cache(
    path_text: str,
    size: int,
    modified_ns: int,
    machine_id: str,
    module_id: int,
    policy_path: str,
) -> pd.DataFrame:
    del size, modified_ns
    raw = read_handler_log(Path(path_text), machine_id, module_ids=[module_id])
    if raw.empty:
        return pd.DataFrame()
    policy = ControlledMonitoringConfig.load(policy_path)
    return build_event_windows(raw, policy)


class ModelMonitorStore:
    def __init__(self) -> None:
        self.repository = ProfileRepository(settings.controlled_runtime_dir / "profiles")

    def _system(self) -> dict[str, Any]:
        return _json(settings.controlled_system_config, {"machine_sources": {}, "modules": []})

    def _policy(self) -> dict[str, Any]:
        return _json(settings.controlled_policy_file, {})

    def sync_status(self) -> dict[str, Any]:
        sync = self._system().get("sync", {})
        if not isinstance(sync, dict) or not sync.get("state_dir"):
            return {
                "available": False,
                "message": "No sync.state_dir is configured for this model monitor.",
            }
        state_dir = Path(str(sync["state_dir"]))
        if not state_dir.is_absolute():
            state_dir = settings.controlled_system_config.parent / state_dir
        latest = _json(state_dir / "latest_sync.json", {})
        return {
            "available": bool(latest),
            "state_dir": str(state_dir),
            "latest": latest or None,
            "message": (
                "SMB sync has not completed yet."
                if not latest
                else None
            ),
        }

    def _source_roots(self) -> dict[str, Path]:
        roots: dict[str, Path] = {}
        for handler in list_handlers():
            destination = Path(handler["destination"])
            if destination.exists():
                roots[handler["name"]] = destination
        for machine, source in self._system().get("machine_sources", {}).items():
            candidate = Path(str(source))
            if not candidate.is_absolute():
                candidate = settings.model_project_root / candidate
            if machine not in roots and candidate.exists():
                roots[machine] = candidate
        return roots

    def _files(self, machine_id: str) -> list[Path]:
        root = self._source_roots().get(machine_id)
        if root is None:
            return []
        try:
            return discover_daily_files(root)
        except (FileNotFoundError, ValueError):
            return []

    @staticmethod
    def _select_file(files: list[Path], selected_date: date | None) -> Path | None:
        if not files:
            return None
        if selected_date is None:
            return files[-1]
        tokens = {
            selected_date.strftime("%Y_%m_%d"),
            selected_date.strftime("%Y-%m-%d"),
            selected_date.strftime("%Y%m%d"),
        }
        matches = [path for path in files if any(token in path.name for token in tokens)]
        return matches[-1] if matches else None

    def _predictions(self, machine_id: str) -> list[dict[str, Any]]:
        path = settings.controlled_runtime_dir / "predictions" / f"{machine_id}.jsonl"
        if not path.exists():
            return []
        stat = path.stat()
        return list(
            _prediction_cache(
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                settings.prediction_read_limit,
            )
        )

    def _lifecycle(self, machine_id: str) -> dict[str, Any]:
        return self.repository.read_lifecycle(machine_id)

    def artifact(self) -> dict[str, Any]:
        root = settings.shared_model_artifact
        manifest = _json(root / "manifest.json", {})
        config = _json(root / "config.json", {})
        thresholds = _json(root / "thresholds.json", {})
        metrics: list[dict[str, Any]] = []
        metrics_path = root / "group_metrics.csv"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
                metrics = list(csv.DictReader(handle))
        exceedance = [
            float(item["test_exceedance_rate"])
            for item in metrics
            if item.get("test_exceedance_rate") not in {None, ""}
        ]
        return {
            "available": bool(manifest and (root / "shared_model.keras").exists()),
            "model_version": manifest.get("model_version"),
            "model_type": manifest.get("model_type"),
            "run_mode": manifest.get("run_mode"),
            "created_at_utc": manifest.get("created_at_utc"),
            "epochs_completed": manifest.get("epochs_completed"),
            "final_loss": manifest.get("final_loss"),
            "final_validation_loss": manifest.get("final_validation_loss"),
            "threshold_method": manifest.get("threshold_method"),
            "group_count": len(manifest.get("groups", [])),
            "groups": manifest.get("groups", []),
            "input_shape": [config.get("window_size_sec"), len(config.get("feature_columns", []))],
            "sampling_interval_seconds": config.get("sampling_interval_sec"),
            "step_size_seconds": config.get("step_size_sec"),
            "feature_columns": config.get("feature_columns", []),
            "threshold_count": len(thresholds),
            "mean_test_exceedance_rate": float(np.mean(exceedance)) if exceedance else None,
            "weights_mutable": False,
            "role": "immutable shadow evidence and bootstrap clean-window selection",
        }

    def artifact_group(self, machine_id: str, module_id: int) -> dict[str, Any]:
        group = safe_group_name(machine_id, module_id)
        root = settings.shared_model_artifact
        thresholds = _json(root / "thresholds.json", {})
        metric: dict[str, Any] | None = None
        metrics_path = root / "group_metrics.csv"
        if metrics_path.exists():
            frame = pd.read_csv(metrics_path)
            selected = frame.loc[
                frame["machine_id"].astype(str).eq(machine_id)
                & frame["module_id"].astype(int).eq(module_id)
            ]
            if not selected.empty:
                metric = {
                    key: _finite(value) if key not in {"machine_id", "module_id"} else value
                    for key, value in selected.iloc[0].to_dict().items()
                }
        return {
            "group_name": group,
            "configured": group in thresholds,
            "threshold": thresholds.get(group),
            "held_out_metrics": metric,
        }

    def _profile(self, machine_id: str, module_id: int) -> dict[str, Any]:
        lifecycle = self._lifecycle(machine_id)
        source = "active" if lifecycle.get("active_version") else "candidate"
        version = lifecycle.get("active_version") or lifecycle.get("candidate_version")
        if not version:
            return {
                "available": False,
                "source": None,
                "version": None,
                "status": "NOT_CREATED",
                "contexts": [],
            }
        directory = (
            self.repository.active_dir(machine_id, str(version))
            if source == "active"
            else self.repository.candidate_dir(machine_id, str(version))
        )
        path = directory / f"M{module_id:02d}.joblib"
        if not path.exists():
            return {
                "available": False,
                "source": source,
                "version": version,
                "status": "MODULE_NOT_PROFILED",
                "contexts": [],
            }
        bundle = joblib.load(path)
        contexts: list[dict[str, Any]] = []
        for key, context in sorted(bundle.contexts.items()):
            contexts.append(
                {
                    "key": key,
                    "operating_mode": context.operating_mode,
                    "regime": context.regime,
                    "training_windows": context.training_windows,
                    "feature_center": context.feature_center,
                    "feature_scale": context.feature_scale,
                    "ridge": {
                        "target": "lp2",
                        "features": ["hp2", "valve", "temphi", "templo"],
                        "intercept": context.ridge_intercept,
                        "coefficients": context.ridge_coefficients,
                        "residual_center": context.residual_center,
                        "residual_scale": context.residual_scale,
                    },
                    "isolation_forest": {
                        "entry_threshold": context.isolation_entry_threshold,
                        "exit_threshold": context.isolation_exit_threshold,
                    },
                }
            )
        regimes = {
            mode: {
                "component_count": model.component_count,
                "training_windows": model.training_windows,
                "selected_bic": model.selected_bic,
                "likelihood_floor": model.likelihood_floor,
                "policy_min_log_likelihood": model.policy_min_log_likelihood,
                "min_posterior": model.min_posterior,
                "fallback_single_regime": model.fallback_single_regime,
            }
            for mode, model in bundle.regime_models.items()
        }
        calibration = bundle.lstm_calibration
        local_lstm = None
        if calibration is not None:
            local_lstm = {
                "source": "LOCAL_BOOTSTRAP",
                "threshold": calibration.threshold,
                "training_windows": calibration.training_windows,
                "validation_windows": calibration.validation_windows,
                "model_version": calibration.model_version,
            }
        return {
            "available": True,
            "source": source,
            "version": version,
            "status": bundle.status,
            "created_at_utc": bundle.created_at_utc,
            "policy_version": bundle.policy_version,
            "source_window_start": bundle.source_window_start,
            "source_window_end": bundle.source_window_end,
            "source_windows": bundle.source_windows,
            "selection_metrics": bundle.selection_metrics,
            "contexts": contexts,
            "regime_models": regimes,
            "local_lstm_calibration": local_lstm,
        }

    @staticmethod
    def _flatten_prediction(item: dict[str, Any]) -> dict[str, Any]:
        com2 = item.get("com2") or {}
        lstm = item.get("lstm") or {}
        com2_details = com2.get("details") or {}
        lstm_details = lstm.get("details") or {}
        lstm_score = _finite(lstm.get("score"))
        lstm_threshold = _finite(lstm.get("threshold"))
        return {
            "event_time": item.get("event_time"),
            "processing_time": item.get("processing_time"),
            "window_status": item.get("window_status"),
            "review_level": item.get("review_level"),
            "operating_mode": item.get("operating_mode"),
            "regime": item.get("regime"),
            "regime_posterior": _finite(item.get("regime_posterior")),
            "regime_log_likelihood": _finite(item.get("regime_log_likelihood")),
            "profile_version": item.get("profile_version"),
            "reason_codes": item.get("reason_codes", []),
            "coverage": _finite(item.get("coverage")),
            "point_count": item.get("point_count"),
            "com2_active": bool(com2.get("active", False)),
            "com2_score": _finite(com2.get("score")),
            "com2_threshold": _finite(com2.get("threshold")),
            "com2_persistent_seconds": item.get("com2_persistent_seconds", 0),
            "predicted_lp2": _finite(com2_details.get("predicted_lp2")),
            "lp2_residual": _finite(com2_details.get("lp2_residual")),
            "z_hp2": _finite(com2_details.get("z_hp2")),
            "z_lp2_residual": _finite(com2_details.get("z_lp2_residual")),
            "z_pressure_gap": _finite(com2_details.get("z_pressure_gap")),
            "z_temperature_span": _finite(com2_details.get("z_temperature_span")),
            "isolation_score": _finite(com2_details.get("isolation_score")),
            "isolation_entry_threshold": _finite(com2_details.get("isolation_entry_threshold")),
            "isolation_exit_threshold": _finite(com2_details.get("isolation_exit_threshold")),
            "lp2_trend": _finite(com2_details.get("lp2_trend")),
            "lstm_active": bool(lstm.get("active", False)),
            "lstm_score": lstm_score,
            "lstm_threshold": lstm_threshold,
            "lstm_ratio": (
                lstm_score / lstm_threshold
                if lstm_score is not None and lstm_threshold not in {None, 0}
                else None
            ),
            "lstm_persistent_seconds": item.get("lstm_persistent_seconds", 0),
            "lstm_sequence_count": lstm_details.get("sequence_count"),
            "lstm_exceedance_fraction": _finite(lstm_details.get("exceedance_fraction")),
            "lstm_top_error_feature": lstm_details.get("top_error_feature"),
            "lstm_calibration_source": lstm_details.get("calibration_source"),
            "lstm_model_version": lstm_details.get("model_version"),
        }

    def _signal_windows(
        self, machine_id: str, module_id: int, selected_date: date | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        files = self._files(machine_id)
        source = self._select_file(files, selected_date)
        if source is None:
            return [], {
                "available": False,
                "file_name": None,
                "message": "No synchronized log file matched the selected date.",
            }
        stat = source.stat()
        signature = hashlib.sha256(
            f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{machine_id}|{module_id}".encode()
        ).hexdigest()[:24]
        cache_path = (
            settings.controlled_runtime_dir
            / "dashboard_cache"
            / "windows"
            / f"{machine_id}__M{module_id:02d}__{signature}.joblib"
        )
        try:
            if cache_path.exists():
                frame = joblib.load(cache_path)
            else:
                frame = _window_cache(
                    str(source.resolve()),
                    stat.st_size,
                    stat.st_mtime_ns,
                    machine_id,
                    module_id,
                    str(settings.controlled_policy_file.resolve()),
                ).copy()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
                joblib.dump(frame, temporary)
                temporary.replace(cache_path)
        except Exception as error:
            return [], {
                "available": False,
                "file_name": source.name,
                "message": f"{type(error).__name__}: {error}",
            }
        if selected_date is not None and not frame.empty:
            frame = frame.loc[pd.to_datetime(frame["event_time"]).dt.date.eq(selected_date)]
        frame = frame.tail(settings.chart_point_limit)
        points: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            points.append(
                {
                    "event_time": _iso(row.get("event_time")),
                    "window_end": _iso(row.get("window_end")),
                    "window_status": row.get("window_status"),
                    "quality_reasons": row.get("quality_reasons", []),
                    "hp1": _finite(row.get("hp1")),
                    "lp1": _finite(row.get("lp1")),
                    "hp2": _finite(row.get("hp2")),
                    "lp2": _finite(row.get("lp2")),
                    "valve": _finite(row.get("valve")),
                    "temphi": _finite(row.get("temphi")),
                    "templo": _finite(row.get("templo")),
                    "pressure_gap": _finite(row.get("pressure_gap")),
                    "pressure_ratio": _finite(row.get("pressure_ratio")),
                    "temperature_span": _finite(row.get("temperature_span")),
                    "sv": row.get("sv"),
                    "module_status": row.get("module_status"),
                    "global_status": row.get("global_status"),
                    "busy": _finite(row.get("busy")),
                    "coverage": _finite(row.get("coverage")),
                    "point_count": int(row.get("point_count", 0)),
                    "maximum_gap_seconds": _finite(row.get("maximum_gap_seconds")),
                }
            )
        return points, {
            "available": True,
            "file_name": source.name,
            "file_modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "point_count": len(points),
            "source_grain": "five-minute median derived from raw event-time rows",
        }

    def machine_monitor(
        self,
        machine_id: str,
        module_id: int,
        selected_date: date | None = None,
    ) -> dict[str, Any]:
        machine = normalize_machine(machine_id)
        if not 1 <= module_id <= 8:
            raise ValueError("module_id must be between 1 and 8")
        signals, source = self._signal_windows(machine, module_id, selected_date)
        predictions = [
            self._flatten_prediction(item)
            for item in self._predictions(machine)
            if int(item.get("module_id", 0)) == module_id
            and (
                selected_date is None
                or pd.Timestamp(item.get("event_time")).date() == selected_date
            )
        ][-settings.chart_point_limit :]
        lifecycle = self._lifecycle(machine)
        profile = self._profile(machine, module_id)
        latest = predictions[-1] if predictions else None
        selected_context = None
        if latest and profile.get("contexts"):
            key = f"{latest.get('operating_mode')}::{latest.get('regime')}"
            selected_context = next(
                (item for item in profile["contexts"] if item["key"] == key), None
            )
        policy = self._policy()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "machine_id": machine,
            "module_id": module_id,
            "selected_date": selected_date,
            "source": source,
            "signals": signals,
            "model_points": predictions,
            "latest_decision": latest,
            "lifecycle": lifecycle,
            "profile": profile,
            "selected_context": selected_context,
            "shared_lstm_group": self.artifact_group(machine, module_id),
            "quality_gate": {
                "minimum_coverage": _finite(policy.get("minimum_coverage")),
                "minimum_window_points": policy.get("minimum_window_points"),
                "max_gap_seconds": _finite(policy.get("max_gap_seconds")),
                "settling_seconds": _finite(policy.get("settling_seconds")),
            },
            "interpretation_guardrail": (
                "Review levels prioritize investigation. They are not failure probabilities, "
                "fault diagnoses, or automatic stop commands."
            ),
        }

    def profiles(self, machine_id: str) -> dict[str, Any]:
        machine = normalize_machine(machine_id)
        modules = self._system().get("modules", [1, 2, 3, 4, 5, 6, 8])
        return {
            "machine_id": machine,
            "lifecycle": self._lifecycle(machine),
            "modules": [self._profile(machine, int(module)) for module in modules],
        }

    def fleet(self) -> dict[str, Any]:
        artifact = self.artifact()
        roots = self._source_roots()
        machines: list[dict[str, Any]] = []
        review_counts: defaultdict[str, int] = defaultdict(int)
        for handler in list_handlers():
            machine = handler["name"]
            lifecycle = self._lifecycle(machine)
            predictions = [self._flatten_prediction(item) for item in self._predictions(machine)]
            latest_by_module: dict[int, dict[str, Any]] = {}
            for raw, flat in zip(self._predictions(machine), predictions, strict=True):
                latest_by_module[int(raw.get("module_id", 0))] = flat
                review_counts[str(flat.get("review_level"))] += 1
            latest = max(
                latest_by_module.values(),
                key=lambda item: str(item.get("event_time") or ""),
                default=None,
            )
            files = self._files(machine)
            active_version = lifecycle.get("active_version")
            candidate_version = lifecycle.get("candidate_version")
            profile_directory = None
            if active_version:
                profile_directory = self.repository.active_dir(machine, str(active_version))
            elif candidate_version:
                profile_directory = self.repository.candidate_dir(machine, str(candidate_version))
            profiled_modules = (
                len(list(profile_directory.glob("M*.joblib")))
                if profile_directory and profile_directory.exists()
                else 0
            )
            machines.append(
                {
                    "machine_id": machine,
                    "enabled": handler["enabled"],
                    "ip": handler["ip"],
                    "data_source_available": machine in roots,
                    "latest_source_file": files[-1].name if files else None,
                    "latest_source_modified_at": (
                        datetime.fromtimestamp(files[-1].stat().st_mtime, tz=timezone.utc).isoformat()
                        if files
                        else None
                    ),
                    "lifecycle_state": lifecycle.get("state"),
                    "candidate_version": candidate_version,
                    "active_version": active_version,
                    "profiled_modules": profiled_modules,
                    "latest_decision": latest,
                    "modules": list(latest_by_module.values()),
                    "shared_lstm_groups": sum(
                        1
                        for module in [1, 2, 3, 4, 5, 6, 8]
                        if safe_group_name(machine, module) in artifact.get("groups", [])
                    ),
                }
            )
        states: defaultdict[str, int] = defaultdict(int)
        for machine in machines:
            states[str(machine["lifecycle_state"])] += 1
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "policy_version": self._policy().get("policy_version"),
            "artifact": artifact,
            "summary": {
                "configured_handlers": len(machines),
                "data_sources_available": sum(bool(item["data_source_available"]) for item in machines),
                "active_frozen": states[LifecycleState.ACTIVE.value],
                "approval_required": states[LifecycleState.APPROVAL_REQUIRED.value],
                "shadow_validation": states[LifecycleState.SHADOW_VALIDATION.value],
                "collecting_data": states[LifecycleState.COLLECTING_DATA.value],
                "p1_review_records": review_counts["P1_REVIEW"],
                "p2_review_records": review_counts["P2_REVIEW"],
            },
            "machines": machines,
            "freshness_note": "Source file timestamps are filesystem freshness; event-time freshness appears after scoring.",
        }

    def comparison(
        self,
        selected_date: date | None,
        requested: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not 2 <= len(requested) <= 6:
            raise ValueError("comparison needs between 2 and 6 series")
        series: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for request in requested:
            machine = normalize_machine(str(request["machine_id"]))
            module = int(request["module_id"])
            metric_key = str(request["metric"])
            metric = METRIC_BY_KEY.get(metric_key)
            if metric is None:
                raise ValueError(f"Unknown comparison metric: {metric_key}")
            identity = (machine, module, metric_key)
            if identity in seen:
                raise ValueError(f"Duplicate comparison series: {identity}")
            seen.add(identity)
            monitor = self.machine_monitor(machine, module, selected_date)
            source_points = (
                monitor["signals"] if metric.source == "signal" else monitor["model_points"]
            )
            points = [
                {"event_time": item["event_time"], "value": item.get(metric_key)}
                for item in source_points
                if item.get(metric_key) is not None
            ]
            series.append(
                {
                    "id": f"{machine}:M{module}:{metric_key}",
                    "label": f"{machine} · M{module} · {metric.short_label}",
                    "machine_id": machine,
                    "module_id": module,
                    "metric": metric_key,
                    "family": metric.family,
                    "unit": metric.unit,
                    "color": metric.color,
                    "point_count": len(points),
                    "points": points,
                }
            )
        left = {item["event_time"]: float(item["value"]) for item in series[0]["points"]}
        right = {item["event_time"]: float(item["value"]) for item in series[1]["points"]}
        aligned_times = sorted(set(left) & set(right))
        pairs = [(left[key], right[key]) for key in aligned_times]
        correlation = None
        if len(pairs) >= 3:
            correlation_value = np.corrcoef(
                [item[0] for item in pairs], [item[1] for item in pairs]
            )[0, 1]
            correlation = _finite(correlation_value)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selected_date": selected_date,
            "series": series,
            "relationship": {
                "pearson_r": correlation,
                "pair_count": len(pairs),
                "points": [
                    {"event_time": key, "x": left[key], "y": right[key]}
                    for key in aligned_times
                ],
                "warning": "Correlation measures co-movement, not causation.",
            },
        }

    def pipeline(self) -> dict[str, Any]:
        policy = self._policy()
        return {
            "policy": policy,
            "stages": [
                {"key": "window", "name": "5-minute event-time window", "status": "ACTIVE", "formula": "median{xᵢ | i ∈ Wₜ}", "detail": "Median aggregation; processing time is audit-only."},
                {"key": "quality", "name": "Quality / state gate", "status": "ACTIVE", "formula": "coverage = n / floor(300 / median Δt)", "detail": "Abstains on gaps, sentinels, duplicate/out-of-order timestamps, OFF, transition, or settling."},
                {"key": "mode", "name": "SV + Valve operating mode", "status": "ACTIVE", "formula": "mode = (SV state, Valve bucket)", "detail": "Deterministic control context before clustering."},
                {"key": "gmm", "name": "GMM regime", "status": "ACTIVE", "formula": "p(x)=Σ πₖ N(x|μₖ,Σₖ); choose K by minimum BIC", "detail": "Returns UNKNOWN_REGIME when posterior or likelihood is insufficient."},
                {"key": "robust", "name": "Robust profile evidence", "status": "ACTIVE", "formula": "z=(x−median)/(1.4826·MAD)", "detail": "Context-specific HP2, pressure-gap, and temperature-span evidence."},
                {"key": "ridge", "name": "Conditional LP2 residual", "status": "ACTIVE", "formula": "r=LP2−(β₀+β₁HP2+β₂Valve+β₃TempHi+β₄TempLo)", "detail": "Ridge α=1; negative robust residual is the main COM2 directional evidence."},
                {"key": "iforest", "name": "Isolation Forest", "status": "ACTIVE", "formula": "v=[zHP2,zLP2 residual,zGap,zTempSpan]", "detail": "200 trees; entry Q99 and exit Q95 are calibrated per frozen context."},
                {"key": "lstm", "name": "Shared LSTM Full shadow", "status": "ACTIVE_SHADOW", "formula": "e=mean|X−fθ(X)|; bucket score=Q95(e)", "detail": "30-epoch immutable shared weights; scaler/threshold remain group-specific."},
                {"key": "fusion", "name": "Persistence and fusion", "status": "ACTIVE", "formula": "P1: COM2∧LSTM ≥15m; P2: either ≥30m", "detail": "Produces review priority, not failure probability or automatic shutdown."},
            ],
            "lifecycle": [
                "COLLECTING_DATA",
                "LEARNING",
                "CANDIDATE_PROFILE_READY",
                "SHADOW_VALIDATION",
                "APPROVAL_REQUIRED",
                "ACTIVE",
            ],
            "activation_policy": "human approval creates versioned ACTIVE_FROZEN",
            "sources": {
                "signals": "synchronized handler log files",
                "decisions": "controlled_runtime/predictions/<machine>.jsonl",
                "profiles": "controlled_runtime/profiles versioned joblib bundles",
                "shared_model": "artifacts/shared_lstm_colab_full manifest, thresholds, metrics, and immutable Keras weights",
            },
        }

    def approve(
        self, machine_id: str, approved_by: str, reason: str | None = None
    ) -> dict[str, Any]:
        return self.repository.approve(
            normalize_machine(machine_id), approved_by=approved_by, reason=reason
        )

    def reject(self, machine_id: str, rejected_by: str, reason: str) -> dict[str, Any]:
        return self.repository.reject(
            normalize_machine(machine_id), rejected_by=rejected_by, reason=reason
        )

    def continue_learning(
        self, machine_id: str, requested_by: str, reason: str | None = None
    ) -> dict[str, Any]:
        return self.repository.transition(
            normalize_machine(machine_id),
            LifecycleState.COLLECTING_DATA,
            reason=f"continue_learning_requested_by:{requested_by}",
            updates={
                "shadow": {},
                "continue_learning_reason": reason,
                "continue_learning_requested_by": requested_by,
            },
        )


store = ModelMonitorStore()
