from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import ControlledMonitoringConfig
from .context import operating_mode
from .fusion import PersistenceResult, PersistenceTracker, fuse_review_level
from .lifecycle import ProfileRepository
from .profiles import FrozenProfileBundle, score_context_profile
from .shadow import SharedLSTMShadow
from .types import Evidence, WindowDecision, WindowStatus
from .windowing import build_event_windows


def _raw_bucket(
    raw: pd.DataFrame,
    event_time: pd.Timestamp,
    machine_id: str,
    module_id: int,
    config: ControlledMonitoringConfig,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(raw["timestamp"], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            config.timezone, nonexistent="shift_forward", ambiguous="NaT"
        )
    else:
        timestamps = timestamps.dt.tz_convert(config.timezone)
    bucket = timestamps.dt.floor(f"{config.window_seconds}s")
    return raw.loc[
        raw["machine_id"].eq(machine_id)
        & raw["module_id"].eq(module_id)
        & bucket.eq(event_time)
    ].copy()


class ControlledMonitoringEngine:
    """Score COM2 first and fuse immutable Shared-LSTM shadow persistence."""

    def __init__(
        self,
        repository: ProfileRepository,
        config: ControlledMonitoringConfig,
        lstm_shadow: SharedLSTMShadow,
        *,
        persistence: PersistenceTracker | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self.lstm_shadow = lstm_shadow
        self.persistence = persistence or PersistenceTracker(config)

    def _bundles(self, machine_id: str, profile_source: str) -> dict[int, FrozenProfileBundle]:
        if profile_source == "active":
            return self.repository.load_active(machine_id)
        if profile_source == "candidate":
            lifecycle = self.repository.read_lifecycle(machine_id)
            version = lifecycle.get("candidate_version")
            if not version:
                return {}
            return self.repository.load_candidate(machine_id, str(version))
        raise ValueError("profile_source must be 'active' or 'candidate'")

    def score_frame(
        self,
        raw_frame: pd.DataFrame,
        *,
        profile_source: str = "active",
    ) -> pd.DataFrame:
        windows = build_event_windows(raw_frame, self.config)
        if windows.empty:
            return pd.DataFrame()
        processing_time = datetime.now(timezone.utc).isoformat()
        machine_bundles = {
            machine_id: self._bundles(str(machine_id), profile_source)
            for machine_id in windows["machine_id"].unique()
        }
        decisions: list[dict[str, Any]] = []
        for _, row in windows.iterrows():
            machine_id = str(row["machine_id"])
            module_id = int(row["module_id"])
            event_time = pd.Timestamp(row["event_time"])
            base_status = str(row["window_status"])
            com2 = Evidence(model="COM2", active=False)
            lstm = Evidence(model="LSTM", active=False)
            mode: str | None = None
            regime: str | None = None
            regime_posterior: float | None = None
            regime_log_likelihood: float | None = None
            regime_reason: str | None = None
            profile_version: str | None = None
            persistence = PersistenceResult(0, 0, None)
            reasons = list(row.get("quality_reasons", []))
            unknown_regime = False
            bundle = machine_bundles.get(machine_id, {}).get(module_id)

            if base_status == WindowStatus.ELIGIBLE.value and bundle is None:
                base_status = WindowStatus.PROFILE_NOT_ACTIVE.value
                reasons.append("PROFILE_NOT_ACTIVE")
            elif base_status == WindowStatus.ELIGIBLE.value and bundle is not None:
                profile_version = bundle.profile_version
                mode = operating_mode(row["sv"], float(row["valve"]), self.config)
                regime_model = bundle.regime_models.get(mode)
                if regime_model is None:
                    base_status = WindowStatus.UNKNOWN_REGIME.value
                    regime = "UNKNOWN_REGIME"
                    unknown_regime = True
                    reasons.append("MODE_NOT_IN_PROFILE")
                else:
                    resolution = regime_model.resolve(row)
                    regime = resolution.regime
                    regime_posterior = resolution.posterior
                    regime_log_likelihood = resolution.log_likelihood
                    regime_reason = resolution.reason
                    if regime == "UNKNOWN_REGIME":
                        base_status = WindowStatus.UNKNOWN_REGIME.value
                        unknown_regime = True
                        reasons.append(resolution.reason or "UNKNOWN_REGIME")
                    context_profile = bundle.context(mode, regime)
                    if base_status == WindowStatus.ELIGIBLE.value and context_profile is None:
                        base_status = WindowStatus.UNKNOWN_REGIME.value
                        unknown_regime = True
                        reasons.append("CONTEXT_PROFILE_NOT_AVAILABLE")
                    if base_status == WindowStatus.ELIGIBLE.value and context_profile is not None:
                        key = f"{machine_id}__M{module_id:02d}::{mode}::{regime}"
                        com2 = score_context_profile(
                            context_profile,
                            row,
                            self.config,
                            active_reason_codes=self.persistence.active_reasons(key),
                        )
                        bucket = _raw_bucket(
                            raw_frame, event_time, machine_id, module_id, self.config
                        )
                        lstm = self.lstm_shadow.score_bucket(
                            bucket,
                            machine_id,
                            module_id,
                            local_calibration=bundle.lstm_calibration,
                        )
                        persistence = self.persistence.update(
                            key,
                            event_time,
                            com2,
                            lstm,
                            z_lp2=float(com2.details["z_lp2_residual"]),
                        )

            review_level = fuse_review_level(
                base_status, com2, lstm, persistence, self.config
            )
            reasons.extend(com2.reason_codes)
            reasons.extend(lstm.reason_codes)
            decision = WindowDecision(
                event_time=event_time.isoformat(),
                processing_time=processing_time,
                machine_id=machine_id,
                module_id=module_id,
                window_status=base_status,
                operating_mode=mode,
                regime=regime,
                profile_version=profile_version,
                com2=com2,
                lstm=lstm,
                com2_persistent_seconds=persistence.com2_seconds,
                lstm_persistent_seconds=persistence.lstm_seconds,
                review_level=review_level,
                reason_codes=sorted(set(reasons)),
            ).to_dict()
            decision.update(
                {
                    "unknown_regime": unknown_regime,
                    "regime_posterior": regime_posterior,
                    "regime_log_likelihood": regime_log_likelihood,
                    "regime_reason": regime_reason,
                    "com2_flag": bool(com2.active),
                    "lstm_flag": bool(lstm.active),
                    "coverage": float(row["coverage"]),
                    "point_count": int(row["point_count"]),
                }
            )
            decisions.append(decision)
        return pd.DataFrame(decisions)
