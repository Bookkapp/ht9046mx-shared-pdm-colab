from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ControlledMonitoringConfig
from .types import Evidence, ReviewLevel, WindowStatus


@dataclass
class PersistenceResult:
    com2_seconds: int
    lstm_seconds: int
    trend: float | None


class PersistenceTracker:
    """Event-time persistence, hysteresis context and one-hour residual history."""

    def __init__(self, config: ControlledMonitoringConfig, state_path: str | Path | None = None):
        self.config = config
        self.state_path = Path(state_path) if state_path else None
        self.state: dict[str, Any] = {}
        if self.state_path and self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def _group(self, key: str) -> dict[str, Any]:
        return self.state.setdefault(
            key,
            {
                "last_event_time": None,
                "models": {
                    "COM2": {"active_start": None, "reason_codes": []},
                    "LSTM": {"active_start": None, "reason_codes": []},
                },
                "lp2_history": [],
            },
        )

    def active_reasons(self, key: str, model: str = "COM2") -> set[str]:
        group = self._group(key)
        return set(group["models"][model].get("reason_codes", []))

    def trend(self, key: str, event_time: pd.Timestamp, z_lp2: float | None) -> float | None:
        group = self._group(key)
        if z_lp2 is None:
            return None
        history = group["lp2_history"]
        history.append([event_time.isoformat(), float(z_lp2)])
        cutoff = event_time - pd.Timedelta(seconds=self.config.trend_lookback_seconds)
        retained = [item for item in history if pd.Timestamp(item[0]) >= cutoff]
        group["lp2_history"] = retained
        if len(retained) < 2:
            return 0.0
        return float(retained[-1][1] - retained[0][1])

    def _duration(
        self,
        model_state: dict[str, Any],
        evidence: Evidence,
        event_time: pd.Timestamp,
    ) -> int:
        if not evidence.active:
            model_state["active_start"] = None
            model_state["reason_codes"] = []
            return 0
        if model_state.get("active_start") is None:
            model_state["active_start"] = event_time.isoformat()
        model_state["reason_codes"] = list(evidence.reason_codes)
        start = pd.Timestamp(model_state["active_start"])
        return int((event_time - start).total_seconds()) + self.config.window_seconds

    def update(
        self,
        key: str,
        event_time: pd.Timestamp,
        com2: Evidence,
        lstm: Evidence,
        *,
        z_lp2: float | None = None,
    ) -> PersistenceResult:
        group = self._group(key)
        last = pd.Timestamp(group["last_event_time"]) if group.get("last_event_time") else None
        if last is not None:
            gap = (event_time - last).total_seconds()
            if gap < 0 or gap > self.config.persistence_reset_gap_seconds:
                group["models"] = {
                    "COM2": {"active_start": None, "reason_codes": []},
                    "LSTM": {"active_start": None, "reason_codes": []},
                }
                group["lp2_history"] = []
        trend = self.trend(key, event_time, z_lp2)
        trend_active_before = "LP2_DOWNWARD_TREND" in self.active_reasons(key)
        if trend is not None:
            trend_active = (
                abs(trend) >= self.config.trend_exit_abs
                if trend_active_before
                else trend <= self.config.trend_entry
            )
            if trend_active and "LP2_DOWNWARD_TREND" not in com2.reason_codes:
                com2.reason_codes.append("LP2_DOWNWARD_TREND")
                com2.active = True
            com2.details["lp2_trend"] = trend
        com2_seconds = self._duration(group["models"]["COM2"], com2, event_time)
        lstm_seconds = self._duration(group["models"]["LSTM"], lstm, event_time)
        group["last_event_time"] = event_time.isoformat()
        self.save()
        return PersistenceResult(com2_seconds, lstm_seconds, trend)

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)


def fuse_review_level(
    window_status: str,
    com2: Evidence,
    lstm: Evidence,
    persistence: PersistenceResult,
    config: ControlledMonitoringConfig,
) -> str:
    if window_status != WindowStatus.ELIGIBLE.value:
        return window_status
    if (
        persistence.com2_seconds >= config.p1_dual_seconds
        and persistence.lstm_seconds >= config.p1_dual_seconds
    ):
        return ReviewLevel.P1_REVIEW.value
    if (
        persistence.com2_seconds >= config.p2_single_seconds
        or persistence.lstm_seconds >= config.p2_single_seconds
    ):
        return ReviewLevel.P2_REVIEW.value
    if com2.active or lstm.active:
        return ReviewLevel.SHADOW.value
    return ReviewLevel.NORMAL.value
