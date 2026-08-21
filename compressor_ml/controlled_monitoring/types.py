from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class WindowStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    DATA_QUALITY_REVIEW = "DATA_QUALITY_REVIEW"
    INCOMPLETE_WINDOW = "INCOMPLETE_WINDOW"
    OFF_OR_TRANSITION = "OFF_OR_TRANSITION"
    UNKNOWN_REGIME = "UNKNOWN_REGIME"
    PROFILE_NOT_ACTIVE = "PROFILE_NOT_ACTIVE"


class ReviewLevel(str, Enum):
    NORMAL = "NORMAL"
    SHADOW = "SHADOW"
    P2_REVIEW = "P2_REVIEW"
    P1_REVIEW = "P1_REVIEW"


@dataclass
class Evidence:
    model: str
    active: bool
    score: float | None = None
    threshold: float | None = None
    reason_codes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowDecision:
    event_time: str
    processing_time: str
    machine_id: str
    module_id: int
    window_status: str
    operating_mode: str | None
    regime: str | None
    profile_version: str | None
    com2: Evidence
    lstm: Evidence
    com2_persistent_seconds: int
    lstm_persistent_seconds: int
    review_level: str
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
