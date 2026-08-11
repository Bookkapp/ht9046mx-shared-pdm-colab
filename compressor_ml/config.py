from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    sampling_interval_sec: int = 1
    max_interpolation_gap_sec: int = 5
    window_size_sec: int = 60
    step_size_sec: int = 5
    excluded_modules: tuple[int, ...] = (7,)
    transition_states: tuple[str, ...] = ("ChangeValve", "AdjustValve", "MValveHome")
    normal_status: str = "On"
    scaler: str = "standard"
    threshold_percentile: float = 99.0
    health_smoothing_windows: int = 12
    normal_min: float = 80.0
    watch_min: float = 60.0
    warning_min: float = 40.0
    encoder_units: tuple[int, int] = (64, 32)
    decoder_units: tuple[int, int] = (32, 64)
    loss: str = "mae"
    epochs: int = 30
    batch_size: int = 128
    feature_columns: tuple[str, ...] = (
        "hp1", "lp1", "hp2", "lp2", "valve", "temphi", "templo",
        "hp_gap_1", "hp_gap_2", "hp_lp_ratio_1", "hp_lp_ratio_2",
        "delta_hp1", "delta_lp1", "delta_hp2", "delta_lp2", "delta_valve",
        "hp1_mean_60", "hp1_std_60", "hp2_mean_60", "hp2_std_60",
        "lp2_mean_60", "lp2_std_60", "templo_mean_60", "templo_std_60",
    )

    @property
    def window_rows(self) -> int:
        return self.window_size_sec // self.sampling_interval_sec

    @property
    def step_rows(self) -> int:
        return self.step_size_sec // self.sampling_interval_sec

    def validate(self) -> None:
        if self.window_rows < 2 or self.step_rows < 1:
            raise ValueError("window_size_sec and step_size_sec must be valid multiples of sampling_interval_sec")
        if not 0 < self.threshold_percentile < 100:
            raise ValueError("threshold_percentile must be between 0 and 100")
        if not 0 <= self.warning_min <= self.watch_min <= self.normal_min <= 100:
            raise ValueError("Pseudo-label thresholds must satisfy 0 <= warning <= watch <= normal <= 100")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        tuple_fields = {"excluded_modules", "transition_states", "encoder_units", "decoder_units", "feature_columns"}
        cleaned = {k: tuple(v) if k in tuple_fields and isinstance(v, list) else v for k, v in data.items()}
        cfg = cls(**cleaned)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "PipelineConfig":
        if path is None:
            cfg = cls()
        else:
            cfg = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        cfg.validate()
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
