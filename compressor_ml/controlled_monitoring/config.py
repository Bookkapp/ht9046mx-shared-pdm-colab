from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ControlledMonitoringConfig:
    """Versioned policy for the COM2-primary, LSTM-shadow pipeline."""

    policy_version: str = "controlled_hybrid_v1"
    timezone: str = "Asia/Bangkok"
    window_seconds: int = 300
    max_gap_seconds: int = 15
    persistence_reset_gap_seconds: int = 600
    settling_seconds: int = 120
    minimum_window_points: int = 30
    minimum_coverage: float = 0.90
    pressure_min: float = 0.0
    pressure_max: float = 1000.0
    temperature_min: float = -150.0
    temperature_max: float = 250.0
    sentinel_values: tuple[float, ...] = (-200.0,)
    transition_states: tuple[str, ...] = (
        "changevalve",
        "adjustvalve",
        "mvalvehome",
        "startup",
        "shutdown",
    )
    running_states: tuple[str, ...] = ("on", "run", "running", "2")
    sv_on_tokens: tuple[str, ...] = ("on", "true", "1", "2")
    valve_bins: tuple[float, ...] = (20.0, 50.0, 80.0)
    regime_features: tuple[str, ...] = ("hp2", "lp2", "valve", "temphi", "templo")
    regime_min_windows: int = 40
    regime_max_components: int = 3
    regime_min_posterior: float = 0.60
    regime_likelihood_quantile: float = 0.01
    regime_min_log_likelihood: float = -12.0
    profile_min_context_windows: int = 25
    ridge_alpha: float = 1.0
    robust_entry_z: float = 3.5
    robust_exit_z: float = 2.5
    isolation_estimators: int = 200
    isolation_entry_quantile: float = 0.99
    isolation_exit_quantile: float = 0.95
    random_seed: int = 42
    trend_lookback_seconds: int = 3600
    trend_entry: float = -3.0
    trend_exit_abs: float = 2.0
    p1_dual_seconds: int = 900
    p2_single_seconds: int = 1800
    lstm_bucket_quantile: float = 0.95
    lstm_min_sequences: int = 3
    bootstrap_min_days: int = 7
    bootstrap_recommended_days: int = 14
    bootstrap_min_eligible_windows: int = 200
    bootstrap_lstm_max_ratio: float = 1.0
    bootstrap_com2_max_flag_rate: float = 0.05
    shadow_min_days: int = 3
    shadow_min_windows: int = 100
    shadow_max_unknown_regime_rate: float = 0.05
    shadow_max_com2_flag_rate: float = 0.05
    shadow_max_lstm_flag_rate: float = 0.10
    activation_policy: str = "human_approval"
    active_profile_is_frozen: bool = True
    shared_lstm_weights_mutable: bool = False
    profile_features: tuple[str, ...] = (
        "hp1",
        "lp1",
        "hp2",
        "lp2",
        "valve",
        "temphi",
        "templo",
        "pressure_gap",
        "pressure_ratio",
        "temperature_span",
    )

    def validate(self) -> None:
        if self.window_seconds <= 0 or self.minimum_window_points < 2:
            raise ValueError("Window policy must be positive")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1]")
        if not 0 < self.regime_min_posterior <= 1:
            raise ValueError("regime_min_posterior must be in (0, 1]")
        if not 0 < self.regime_likelihood_quantile < 1:
            raise ValueError("regime_likelihood_quantile must be in (0, 1)")
        if not 0 < self.isolation_exit_quantile < self.isolation_entry_quantile < 1:
            raise ValueError("Isolation Forest exit quantile must be below entry quantile")
        if not 0 < self.bootstrap_lstm_max_ratio:
            raise ValueError("bootstrap_lstm_max_ratio must be positive")
        if self.activation_policy != "human_approval":
            raise ValueError("Production policy requires human_approval")
        if not self.active_profile_is_frozen or self.shared_lstm_weights_mutable:
            raise ValueError("Active profiles and shared LSTM weights must remain frozen")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ControlledMonitoringConfig":
        tuple_fields = {
            "sentinel_values",
            "transition_states",
            "running_states",
            "sv_on_tokens",
            "valve_bins",
            "regime_features",
            "profile_features",
        }
        known = cls.__dataclass_fields__
        cleaned = {
            key: tuple(value) if key in tuple_fields and isinstance(value, list) else value
            for key, value in payload.items()
            if key in known
        }
        config = cls(**cleaned)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ControlledMonitoringConfig":
        if path is None:
            config = cls()
        else:
            config = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
