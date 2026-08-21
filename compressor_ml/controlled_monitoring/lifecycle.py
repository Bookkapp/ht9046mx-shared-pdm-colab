from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..prepare_dataset import safe_group_name
from .config import ControlledMonitoringConfig
from .context import add_operating_modes
from .profiles import FrozenProfileBundle, fit_frozen_profile_bundle, score_context_profile
from .shadow import SharedLSTMShadow
from .types import WindowStatus
from .windowing import build_event_windows


class LifecycleState(str, Enum):
    COLLECTING_DATA = "COLLECTING_DATA"
    LEARNING = "LEARNING"
    CANDIDATE_PROFILE_READY = "CANDIDATE_PROFILE_READY"
    SHADOW_VALIDATION = "SHADOW_VALIDATION"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class ProfileRepository:
    """Versioned candidate and immutable active-profile repository."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def lifecycle_path(self, machine_id: str) -> Path:
        return self.root / "machines" / machine_id / "lifecycle.json"

    def read_lifecycle(self, machine_id: str) -> dict[str, Any]:
        path = self.lifecycle_path(machine_id)
        if not path.exists():
            return {
                "machine_id": machine_id,
                "state": LifecycleState.COLLECTING_DATA.value,
                "candidate_version": None,
                "active_version": None,
                "history": [],
                "shadow": {},
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def write_lifecycle(self, machine_id: str, payload: dict[str, Any]) -> None:
        _atomic_json(self.lifecycle_path(machine_id), payload)

    def transition(
        self,
        machine_id: str,
        state: LifecycleState,
        *,
        reason: str,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.read_lifecycle(machine_id)
        payload["state"] = state.value
        if updates:
            payload.update(updates)
        payload.setdefault("history", []).append(
            {
                "state": state.value,
                "reason": reason,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.write_lifecycle(machine_id, payload)
        return payload

    def candidate_dir(self, machine_id: str, version: str) -> Path:
        return self.root / "candidates" / machine_id / version

    def active_dir(self, machine_id: str, version: str) -> Path:
        return self.root / "active" / machine_id / version

    def save_candidate(
        self,
        machine_id: str,
        version: str,
        bundles: dict[int, FrozenProfileBundle],
        metrics: dict[str, Any],
    ) -> None:
        destination = self.candidate_dir(machine_id, version)
        if destination.exists():
            raise FileExistsError(f"Candidate already exists: {destination}")
        destination.mkdir(parents=True)
        for module_id, bundle in bundles.items():
            joblib.dump(bundle, destination / f"M{int(module_id):02d}.joblib")
        _atomic_json(
            destination / "manifest.json",
            {
                "machine_id": machine_id,
                "profile_version": version,
                "status": "CANDIDATE",
                "modules": sorted(int(value) for value in bundles),
                "metrics": metrics,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def load_candidate(self, machine_id: str, version: str) -> dict[int, FrozenProfileBundle]:
        directory = self.candidate_dir(machine_id, version)
        return {
            int(path.stem[1:]): joblib.load(path)
            for path in sorted(directory.glob("M*.joblib"))
        }

    def load_active(self, machine_id: str) -> dict[int, FrozenProfileBundle]:
        lifecycle = self.read_lifecycle(machine_id)
        version = lifecycle.get("active_version")
        if not version:
            return {}
        directory = self.active_dir(machine_id, version)
        return {
            int(path.stem[1:]): joblib.load(path)
            for path in sorted(directory.glob("M*.joblib"))
        }

    def approve(
        self,
        machine_id: str,
        *,
        approved_by: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        lifecycle = self.read_lifecycle(machine_id)
        if lifecycle.get("state") != LifecycleState.APPROVAL_REQUIRED.value:
            raise ValueError("Profile must be in APPROVAL_REQUIRED before activation")
        version = str(lifecycle["candidate_version"])
        source = self.candidate_dir(machine_id, version)
        destination = self.active_dir(machine_id, version)
        if destination.exists():
            raise FileExistsError(f"Active version already exists: {destination}")
        destination.mkdir(parents=True)
        for path in source.glob("M*.joblib"):
            bundle: FrozenProfileBundle = joblib.load(path)
            bundle.status = "ACTIVE_FROZEN"
            joblib.dump(bundle, destination / path.name)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "ACTIVE_FROZEN",
                "approved_by": approved_by,
                "approved_at_utc": datetime.now(timezone.utc).isoformat(),
                "approval_reason": reason,
            }
        )
        _atomic_json(destination / "manifest.json", manifest)
        return self.transition(
            machine_id,
            LifecycleState.ACTIVE,
            reason="human_approval",
            updates={
                "active_version": version,
                "approved_by": approved_by,
                "approval_reason": reason,
            },
        )

    def reject(self, machine_id: str, *, rejected_by: str, reason: str) -> dict[str, Any]:
        return self.transition(
            machine_id,
            LifecycleState.REJECTED,
            reason=reason,
            updates={"rejected_by": rejected_by},
        )


def _raw_bucket(
    raw: pd.DataFrame,
    event_time: pd.Timestamp,
    machine_id: str,
    module_id: int,
    config: ControlledMonitoringConfig,
) -> pd.DataFrame:
    stamps = pd.to_datetime(raw["timestamp"], errors="coerce")
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(config.timezone, nonexistent="shift_forward", ambiguous="NaT")
    else:
        stamps = stamps.dt.tz_convert(config.timezone)
    bucket = stamps.dt.floor(f"{config.window_seconds}s")
    return raw.loc[
        raw["machine_id"].eq(machine_id)
        & raw["module_id"].eq(module_id)
        & bucket.eq(event_time)
    ].copy()


def _assign_and_score(
    bundle: FrozenProfileBundle,
    frame: pd.DataFrame,
    config: ControlledMonitoringConfig,
) -> np.ndarray:
    flags: list[bool] = []
    with_modes = add_operating_modes(frame, config)
    for _, row in with_modes.iterrows():
        mode = str(row["operating_mode"])
        regime_model = bundle.regime_models.get(mode)
        if regime_model is None:
            flags.append(True)
            continue
        resolution = regime_model.resolve(row)
        profile = bundle.context(mode, resolution.regime)
        if resolution.regime == "UNKNOWN_REGIME" or profile is None:
            flags.append(True)
            continue
        flags.append(score_context_profile(profile, row, config).active)
    return np.asarray(flags, dtype=bool)


class BootstrapLifecycle:
    """Auto-build candidates while retaining a mandatory first human approval."""

    def __init__(
        self,
        repository: ProfileRepository,
        config: ControlledMonitoringConfig,
        lstm_shadow: SharedLSTMShadow,
    ) -> None:
        self.repository = repository
        self.config = config
        self.lstm_shadow = lstm_shadow

    def bootstrap(self, machine_id: str, raw_history: pd.DataFrame) -> dict[str, Any]:
        windows = build_event_windows(raw_history, self.config)
        machine_windows = windows.loc[windows["machine_id"].eq(machine_id)].copy()
        eligible = machine_windows.loc[
            machine_windows["window_status"].eq(WindowStatus.ELIGIBLE.value)
        ].copy()
        day_count = pd.to_datetime(eligible["event_time"]).dt.date.nunique() if not eligible.empty else 0
        if (
            day_count < self.config.bootstrap_min_days
            or len(eligible) < self.config.bootstrap_min_eligible_windows
        ):
            return self.repository.transition(
                machine_id,
                LifecycleState.COLLECTING_DATA,
                reason="insufficient_bootstrap_history",
                updates={
                    "eligible_windows": int(len(eligible)),
                    "eligible_days": int(day_count),
                },
            )
        self.repository.transition(machine_id, LifecycleState.LEARNING, reason="bootstrap_threshold_met")
        version = f"{machine_id}_auto_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v1"
        bundles: dict[int, FrozenProfileBundle] = {}
        module_metrics: dict[str, Any] = {}
        for module_id, module_windows in eligible.groupby("module_id", sort=True):
            module_id = int(module_id)
            group_name = safe_group_name(machine_id, module_id)
            local_calibration = None
            if not self.lstm_shadow.bundle.has_group(group_name):
                local_calibration = self.lstm_shadow.fit_local_calibration(
                    raw_history, machine_id, module_id
                )
            ratios: list[float] = []
            for _, window in module_windows.iterrows():
                bucket = _raw_bucket(
                    raw_history,
                    pd.Timestamp(window["event_time"]),
                    machine_id,
                    module_id,
                    self.config,
                )
                evidence = self.lstm_shadow.score_bucket(
                    bucket,
                    machine_id,
                    module_id,
                    local_calibration=local_calibration,
                )
                ratio = (
                    float(evidence.score / evidence.threshold)
                    if evidence.score is not None and evidence.threshold
                    else float("inf")
                )
                ratios.append(ratio)
            module_windows = module_windows.assign(lstm_ratio=ratios)
            lstm_clean = module_windows.loc[
                module_windows["lstm_ratio"].le(self.config.bootstrap_lstm_max_ratio)
            ].copy()
            if len(lstm_clean) < self.config.profile_min_context_windows:
                continue
            preliminary = fit_frozen_profile_bundle(
                lstm_clean,
                machine_id,
                module_id,
                self.config,
                profile_version=f"{version}_M{module_id:02d}_preliminary",
            )
            preliminary_flags = _assign_and_score(preliminary, lstm_clean, self.config)
            clean = lstm_clean.loc[~preliminary_flags].copy()
            if len(clean) < self.config.profile_min_context_windows:
                continue
            final = fit_frozen_profile_bundle(
                clean,
                machine_id,
                module_id,
                self.config,
                profile_version=f"{version}_M{module_id:02d}",
                selection_metrics={
                    "eligible_windows": int(len(module_windows)),
                    "lstm_clean_windows": int(len(lstm_clean)),
                    "two_pass_clean_windows": int(len(clean)),
                    "preliminary_com2_flag_rate": float(np.mean(preliminary_flags)),
                },
            )
            final.lstm_calibration = local_calibration
            bundles[module_id] = final
            module_metrics[str(module_id)] = final.selection_metrics
        if not bundles:
            return self.repository.transition(
                machine_id,
                LifecycleState.COLLECTING_DATA,
                reason="no_module_profile_passed_two_pass_selection",
            )
        metrics = {
            "eligible_windows": int(len(eligible)),
            "eligible_days": int(day_count),
            "modules_profiled": sorted(bundles),
            "module_metrics": module_metrics,
        }
        self.repository.save_candidate(machine_id, version, bundles, metrics)
        return self.repository.transition(
            machine_id,
            LifecycleState.CANDIDATE_PROFILE_READY,
            reason="candidate_generated_automatically",
            updates={"candidate_version": version, "candidate_metrics": metrics, "shadow": {}},
        )

    def begin_shadow(self, machine_id: str) -> dict[str, Any]:
        lifecycle = self.repository.read_lifecycle(machine_id)
        if lifecycle.get("state") != LifecycleState.CANDIDATE_PROFILE_READY.value:
            raise ValueError("Candidate must be ready before shadow validation")
        return self.repository.transition(
            machine_id,
            LifecycleState.SHADOW_VALIDATION,
            reason="shadow_validation_started",
        )

    def record_shadow(self, machine_id: str, decisions: pd.DataFrame) -> dict[str, Any]:
        lifecycle = self.repository.read_lifecycle(machine_id)
        if lifecycle.get("state") != LifecycleState.SHADOW_VALIDATION.value:
            raise ValueError("Machine is not in SHADOW_VALIDATION")
        shadow = lifecycle.setdefault("shadow", {})
        attempted = decisions.loc[
            decisions["window_status"].isin(
                [WindowStatus.ELIGIBLE.value, WindowStatus.UNKNOWN_REGIME.value]
            )
        ]
        known = attempted.loc[
            attempted["window_status"].eq(WindowStatus.ELIGIBLE.value)
        ]
        shadow["windows"] = int(shadow.get("windows", 0) + len(attempted))
        shadow["known_windows"] = int(
            shadow.get("known_windows", 0) + len(known)
        )
        dates = set(shadow.get("dates", []))
        dates.update(
            pd.to_datetime(attempted["event_time"]).dt.strftime("%Y-%m-%d").tolist()
        )
        shadow["dates"] = sorted(dates)
        shadow["unknown_regime"] = int(
            shadow.get("unknown_regime", 0) + int(attempted["unknown_regime"].sum())
        )
        for field in ("com2_flag", "lstm_flag"):
            shadow[field] = int(shadow.get(field, 0) + int(known[field].sum()))
        attempted_denominator = max(1, shadow["windows"])
        known_denominator = max(1, shadow["known_windows"])
        rates = {
            "unknown_regime_rate": shadow["unknown_regime"] / attempted_denominator,
            "com2_flag_rate": shadow["com2_flag"] / known_denominator,
            "lstm_flag_rate": shadow["lstm_flag"] / known_denominator,
        }
        lifecycle["shadow"] = {**shadow, **rates}
        enough = (
            len(dates) >= self.config.shadow_min_days
            and shadow["windows"] >= self.config.shadow_min_windows
        )
        passed = (
            rates["unknown_regime_rate"] <= self.config.shadow_max_unknown_regime_rate
            and rates["com2_flag_rate"] <= self.config.shadow_max_com2_flag_rate
            and rates["lstm_flag_rate"] <= self.config.shadow_max_lstm_flag_rate
        )
        self.repository.write_lifecycle(machine_id, lifecycle)
        if enough and passed:
            return self.repository.transition(
                machine_id,
                LifecycleState.APPROVAL_REQUIRED,
                reason="shadow_acceptance_gates_passed",
                updates={"shadow": lifecycle["shadow"]},
            )
        return lifecycle
