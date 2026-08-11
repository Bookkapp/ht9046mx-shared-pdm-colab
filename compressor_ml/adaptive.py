from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AdaptiveConfig:
    """Conservative gates for per-machine/module calibration updates.

    The shared model and its train-only input scaler are immutable. Only the
    operational decision profile is allowed to adapt automatically.
    """

    min_candidate_windows: int = 200
    min_candidate_days: int = 3
    max_buffer_windows: int = 5000
    eligibility_max_operational_risk: float = 0.50
    eligibility_max_golden_risk: float = 0.80
    golden_drift_weight: float = 0.50
    adaptation_rate: float = 0.10
    max_center_step_mad: float = 0.10
    max_scale_change_fraction: float = 0.10
    max_threshold_change_fraction: float = 0.10
    covariance_shrinkage: float = 0.10
    max_reference_alert_rate: float = 0.05
    min_synthetic_detection_rate: float = 0.80
    max_synthetic_detection_drop: float = 0.05
    synthetic_shift_mad: float = 3.0
    shadow_min_observations: int = 3
    normal_risk_max: float = 0.50
    watch_risk_max: float = 1.00
    warning_risk_max: float = 1.50

    def validate(self) -> None:
        if self.min_candidate_windows < 10 or self.min_candidate_days < 1:
            raise ValueError("Candidate gates require at least 10 windows and one day")
        if self.max_buffer_windows < self.min_candidate_windows:
            raise ValueError("max_buffer_windows must be >= min_candidate_windows")
        for name in (
            "eligibility_max_operational_risk",
            "eligibility_max_golden_risk",
            "golden_drift_weight",
            "adaptation_rate",
            "max_center_step_mad",
            "max_scale_change_fraction",
            "max_threshold_change_fraction",
            "covariance_shrinkage",
            "max_reference_alert_rate",
            "min_synthetic_detection_rate",
            "max_synthetic_detection_drop",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.synthetic_shift_mad <= 0 or self.shadow_min_observations < 1:
            raise ValueError("Synthetic shift and shadow gates must be positive")
        if not 0 < self.normal_risk_max < self.watch_risk_max < self.warning_risk_max:
            raise ValueError("Risk bands must be strictly increasing")

    @classmethod
    def load(cls, path: str | Path | None) -> "AdaptiveConfig":
        if path is None:
            config = cls()
        else:
            config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config

    def save(self, path: str | Path) -> None:
        self.validate()
        _atomic_json(Path(path), asdict(self))


@dataclass
class CalibrationProfile:
    group_name: str
    machine_id: str
    module_id: int
    feature_columns: list[str]
    model_version: str
    profile_version: str
    status: str
    created_at_utc: str
    parent_version: str | None
    golden_center: list[float]
    golden_scale: list[float]
    golden_inverse_covariance: list[list[float]]
    adaptive_center: list[float]
    adaptive_scale: list[float]
    adaptive_inverse_covariance: list[list[float]]
    golden_reconstruction_threshold: float
    adaptive_reconstruction_threshold: float
    golden_feature_threshold: float
    adaptive_feature_threshold: float
    golden_relation_threshold: float
    adaptive_relation_threshold: float
    reference_window_count: int
    buffer_total_seen: int = 0
    shadow_observations: int = 0
    last_validated_buffer_total: int = 0
    approval: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        feature_count = len(self.feature_columns)
        vectors = (
            self.golden_center,
            self.golden_scale,
            self.adaptive_center,
            self.adaptive_scale,
        )
        if feature_count == 0 or any(len(vector) != feature_count for vector in vectors):
            raise ValueError("Profile vector width does not match feature_columns")
        matrices = (self.golden_inverse_covariance, self.adaptive_inverse_covariance)
        if any(np.asarray(matrix).shape != (feature_count, feature_count) for matrix in matrices):
            raise ValueError("Profile covariance matrix has the wrong shape")
        thresholds = (
            self.golden_reconstruction_threshold,
            self.adaptive_reconstruction_threshold,
            self.golden_feature_threshold,
            self.adaptive_feature_threshold,
            self.golden_relation_threshold,
            self.adaptive_relation_threshold,
        )
        if any(not np.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("All profile thresholds must be finite and positive")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        profile = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        profile.validate()
        return profile

    def save(self, path: str | Path) -> None:
        self.validate()
        _atomic_json(Path(path), asdict(self))


@dataclass
class ApprovalDecision:
    outcome: str
    passed: bool
    reasons: list[str]
    metrics: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def window_summaries(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3 or len(windows) == 0:
        raise ValueError("Expected non-empty windows shaped (samples, time, features)")
    if not np.isfinite(windows).all():
        raise ValueError("Calibration windows contain NaN or infinite values")
    return np.median(windows.astype(np.float64), axis=1)


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Expected at least two two-dimensional observations")
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0, ddof=0)
    scale = np.where(mad > EPSILON, mad, np.where(fallback > EPSILON, fallback, 1.0))
    return center, scale


def inverse_shrunk_covariance(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    standardized = (values - center) / np.maximum(scale, EPSILON)
    if len(standardized) < 2:
        return np.eye(standardized.shape[1], dtype=np.float64)
    covariance = np.cov(standardized, rowvar=False, ddof=0)
    covariance = np.atleast_2d(covariance).astype(np.float64)
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    shrunk += np.eye(shrunk.shape[0], dtype=np.float64) * 1e-6
    return np.linalg.pinv(shrunk, hermitian=True)


def statistical_components(
    summaries: np.ndarray,
    center: Iterable[float],
    scale: Iterable[float],
    inverse_covariance: Iterable[Iterable[float]],
) -> tuple[np.ndarray, np.ndarray]:
    center_array = np.asarray(center, dtype=np.float64)
    scale_array = np.maximum(np.asarray(scale, dtype=np.float64), EPSILON)
    inverse_array = np.asarray(inverse_covariance, dtype=np.float64)
    standardized = (summaries - center_array) / scale_array
    feature_deviation = np.max(np.abs(standardized), axis=1)
    squared_distance = np.einsum("ij,jk,ik->i", standardized, inverse_array, standardized)
    relation_distance = np.sqrt(np.maximum(squared_distance, 0.0) / standardized.shape[1])
    return feature_deviation, relation_distance


def build_reference_profile(
    *,
    group_name: str,
    machine_id: str,
    module_id: int,
    feature_columns: Iterable[str],
    model_version: str,
    train_windows: np.ndarray,
    reconstruction_threshold: float,
    config: AdaptiveConfig,
) -> CalibrationProfile:
    summaries = window_summaries(train_windows)
    center, scale = robust_center_scale(summaries)
    inverse_covariance = inverse_shrunk_covariance(
        summaries, center, scale, config.covariance_shrinkage
    )
    feature_deviation, relation_distance = statistical_components(
        summaries, center, scale, inverse_covariance
    )
    version = f"{group_name}_golden_v1"
    profile = CalibrationProfile(
        group_name=group_name,
        machine_id=machine_id,
        module_id=int(module_id),
        feature_columns=list(feature_columns),
        model_version=model_version,
        profile_version=version,
        status="GOLDEN",
        created_at_utc=utc_now(),
        parent_version=None,
        golden_center=center.tolist(),
        golden_scale=scale.tolist(),
        golden_inverse_covariance=inverse_covariance.tolist(),
        adaptive_center=center.tolist(),
        adaptive_scale=scale.tolist(),
        adaptive_inverse_covariance=inverse_covariance.tolist(),
        golden_reconstruction_threshold=float(reconstruction_threshold),
        adaptive_reconstruction_threshold=float(reconstruction_threshold),
        golden_feature_threshold=float(max(np.percentile(feature_deviation, 99), EPSILON)),
        adaptive_feature_threshold=float(max(np.percentile(feature_deviation, 99), EPSILON)),
        golden_relation_threshold=float(max(np.percentile(relation_distance, 99), EPSILON)),
        adaptive_relation_threshold=float(max(np.percentile(relation_distance, 99), EPSILON)),
        reference_window_count=int(len(train_windows)),
    )
    profile.validate()
    return profile


def score_profile(
    profile: CalibrationProfile,
    summaries: np.ndarray,
    reconstruction_errors: np.ndarray,
    config: AdaptiveConfig,
) -> dict[str, np.ndarray]:
    errors = np.asarray(reconstruction_errors, dtype=np.float64)
    if summaries.ndim != 2 or len(summaries) != len(errors):
        raise ValueError("Summary/error lengths do not match")
    golden_feature, golden_relation = statistical_components(
        summaries,
        profile.golden_center,
        profile.golden_scale,
        profile.golden_inverse_covariance,
    )
    adaptive_feature, adaptive_relation = statistical_components(
        summaries,
        profile.adaptive_center,
        profile.adaptive_scale,
        profile.adaptive_inverse_covariance,
    )
    golden_risk = np.maximum.reduce(
        [
            errors / profile.golden_reconstruction_threshold,
            golden_feature / profile.golden_feature_threshold,
            golden_relation / profile.golden_relation_threshold,
        ]
    )
    operational_risk = np.maximum.reduce(
        [
            errors / profile.adaptive_reconstruction_threshold,
            adaptive_feature / profile.adaptive_feature_threshold,
            adaptive_relation / profile.adaptive_relation_threshold,
        ]
    )
    combined_risk = np.maximum(operational_risk, config.golden_drift_weight * golden_risk)
    health = np.clip(100.0 - 40.0 * combined_risk, 0.0, 100.0)
    status = np.full(len(combined_risk), "Critical", dtype=object)
    status[combined_risk < config.warning_risk_max] = "Warning"
    status[combined_risk < config.watch_risk_max] = "Watch"
    status[combined_risk < config.normal_risk_max] = "Normal"
    eligible = (
        (operational_risk < config.eligibility_max_operational_risk)
        & (golden_risk < config.eligibility_max_golden_risk)
    )
    return {
        "golden_risk": golden_risk,
        "operational_risk": operational_risk,
        "combined_risk": combined_risk,
        "health_score": health,
        "condition_status": status,
        "baseline_drift": golden_risk >= 1.0,
        "eligible_for_calibration": eligible,
        "golden_feature_deviation": golden_feature,
        "golden_relation_distance": golden_relation,
        "adaptive_feature_deviation": adaptive_feature,
        "adaptive_relation_distance": adaptive_relation,
    }


def _bounded_scalar(old: float, target: float, rate: float, fraction: float) -> float:
    blended = old + rate * (target - old)
    lower = old * (1.0 - fraction)
    upper = old * (1.0 + fraction)
    return float(np.clip(blended, min(lower, upper), max(lower, upper)))


def build_candidate_profile(
    champion: CalibrationProfile,
    candidate_summaries: np.ndarray,
    candidate_errors: np.ndarray,
    buffer_total_seen: int,
    config: AdaptiveConfig,
) -> CalibrationProfile:
    if len(candidate_summaries) != len(candidate_errors):
        raise ValueError("Candidate summaries and errors must have equal length")
    target_center, target_scale = robust_center_scale(candidate_summaries)
    old_center = np.asarray(champion.adaptive_center, dtype=np.float64)
    old_scale = np.asarray(champion.adaptive_scale, dtype=np.float64)
    golden_scale = np.maximum(np.asarray(champion.golden_scale, dtype=np.float64), EPSILON)

    blended_center = old_center + config.adaptation_rate * (target_center - old_center)
    center_step = np.clip(
        blended_center - old_center,
        -config.max_center_step_mad * golden_scale,
        config.max_center_step_mad * golden_scale,
    )
    new_center = old_center + center_step
    blended_scale = old_scale + config.adaptation_rate * (target_scale - old_scale)
    new_scale = np.clip(
        blended_scale,
        old_scale * (1.0 - config.max_scale_change_fraction),
        old_scale * (1.0 + config.max_scale_change_fraction),
    )
    new_inverse = inverse_shrunk_covariance(
        candidate_summaries, new_center, new_scale, config.covariance_shrinkage
    )
    feature_deviation, relation_distance = statistical_components(
        candidate_summaries, new_center, new_scale, new_inverse
    )
    candidate_error_threshold = max(float(np.percentile(candidate_errors, 99)), EPSILON)
    version = f"{champion.group_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    candidate = CalibrationProfile(
        group_name=champion.group_name,
        machine_id=champion.machine_id,
        module_id=champion.module_id,
        feature_columns=list(champion.feature_columns),
        model_version=champion.model_version,
        profile_version=version,
        status="CANDIDATE",
        created_at_utc=utc_now(),
        parent_version=champion.profile_version,
        golden_center=list(champion.golden_center),
        golden_scale=list(champion.golden_scale),
        golden_inverse_covariance=[list(row) for row in champion.golden_inverse_covariance],
        adaptive_center=new_center.tolist(),
        adaptive_scale=new_scale.tolist(),
        adaptive_inverse_covariance=new_inverse.tolist(),
        golden_reconstruction_threshold=champion.golden_reconstruction_threshold,
        adaptive_reconstruction_threshold=_bounded_scalar(
            champion.adaptive_reconstruction_threshold,
            candidate_error_threshold,
            config.adaptation_rate,
            config.max_threshold_change_fraction,
        ),
        golden_feature_threshold=champion.golden_feature_threshold,
        adaptive_feature_threshold=_bounded_scalar(
            champion.adaptive_feature_threshold,
            float(max(np.percentile(feature_deviation, 99), EPSILON)),
            config.adaptation_rate,
            config.max_threshold_change_fraction,
        ),
        golden_relation_threshold=champion.golden_relation_threshold,
        adaptive_relation_threshold=_bounded_scalar(
            champion.adaptive_relation_threshold,
            float(max(np.percentile(relation_distance, 99), EPSILON)),
            config.adaptation_rate,
            config.max_threshold_change_fraction,
        ),
        reference_window_count=champion.reference_window_count,
        buffer_total_seen=int(buffer_total_seen),
        shadow_observations=0,
        last_validated_buffer_total=int(buffer_total_seen),
    )
    candidate.validate()
    return candidate


def validate_candidate(
    champion: CalibrationProfile,
    candidate: CalibrationProfile,
    reference_summaries: np.ndarray,
    reference_errors: np.ndarray,
    recent_summaries: np.ndarray,
    recent_errors: np.ndarray,
    recent_timestamps: np.ndarray,
    config: AdaptiveConfig,
) -> ApprovalDecision:
    reasons: list[str] = []
    unique_days = len({str(value)[:10] for value in recent_timestamps.tolist()})
    if len(recent_summaries) < config.min_candidate_windows:
        reasons.append("insufficient_candidate_windows")
    if unique_days < config.min_candidate_days:
        reasons.append("insufficient_candidate_days")

    champion_reference = score_profile(champion, reference_summaries, reference_errors, config)
    candidate_reference = score_profile(candidate, reference_summaries, reference_errors, config)
    champion_alert_rate = float(np.mean(champion_reference["operational_risk"] >= 1.0))
    candidate_alert_rate = float(np.mean(candidate_reference["operational_risk"] >= 1.0))
    allowed_reference_rate = max(
        config.max_reference_alert_rate,
        champion_alert_rate + 0.02,
    )
    if candidate_alert_rate > allowed_reference_rate:
        reasons.append("reference_false_alert_rate_regressed")

    recent_result = score_profile(candidate, recent_summaries, recent_errors, config)
    recent_alert_rate = float(np.mean(recent_result["operational_risk"] >= 1.0))
    if recent_alert_rate > config.max_reference_alert_rate:
        reasons.append("candidate_does_not_fit_eligible_recent_data")

    synthetic = reference_summaries.copy()
    sensor_count = min(7, synthetic.shape[1])
    golden_scale = np.asarray(champion.golden_scale, dtype=np.float64)
    for index in range(len(synthetic)):
        feature_index = index % sensor_count
        synthetic[index, feature_index] += config.synthetic_shift_mad * golden_scale[feature_index]
    champion_synthetic = score_profile(champion, synthetic, reference_errors, config)
    candidate_synthetic = score_profile(candidate, synthetic, reference_errors, config)
    champion_detection = float(np.mean(champion_synthetic["operational_risk"] >= 1.0))
    candidate_detection = float(np.mean(candidate_synthetic["operational_risk"] >= 1.0))
    if candidate_detection < config.min_synthetic_detection_rate:
        reasons.append("synthetic_detection_below_guardrail")
    if candidate_detection + config.max_synthetic_detection_drop < champion_detection:
        reasons.append("synthetic_detection_regressed")

    center_shift = np.max(
        np.abs(
            np.asarray(candidate.adaptive_center) - np.asarray(champion.adaptive_center)
        )
        / np.maximum(np.asarray(champion.golden_scale), EPSILON)
    )
    scale_change = np.max(
        np.abs(
            np.asarray(candidate.adaptive_scale) / np.maximum(np.asarray(champion.adaptive_scale), EPSILON)
            - 1.0
        )
    )
    threshold_change = abs(
        candidate.adaptive_reconstruction_threshold
        / max(champion.adaptive_reconstruction_threshold, EPSILON)
        - 1.0
    )
    if center_shift > config.max_center_step_mad + 1e-9:
        reasons.append("center_step_exceeded")
    if scale_change > config.max_scale_change_fraction + 1e-9:
        reasons.append("scale_step_exceeded")
    if threshold_change > config.max_threshold_change_fraction + 1e-9:
        reasons.append("threshold_step_exceeded")

    metrics: dict[str, float | int | str] = {
        "candidate_windows": int(len(recent_summaries)),
        "candidate_days": int(unique_days),
        "champion_reference_alert_rate": champion_alert_rate,
        "candidate_reference_alert_rate": candidate_alert_rate,
        "candidate_recent_alert_rate": recent_alert_rate,
        "champion_synthetic_detection_rate": champion_detection,
        "candidate_synthetic_detection_rate": candidate_detection,
        "max_center_step_mad": float(center_shift),
        "max_scale_change_fraction": float(scale_change),
        "reconstruction_threshold_change_fraction": float(threshold_change),
    }
    if reasons:
        insufficient_only = set(reasons).issubset(
            {"insufficient_candidate_windows", "insufficient_candidate_days"}
        )
        outcome = "REVIEW_REQUIRED" if insufficient_only else "AUTO_REJECTED"
        return ApprovalDecision(outcome, False, reasons, metrics)
    return ApprovalDecision("SHADOW", True, ["all_validation_gates_passed"], metrics)


class CalibrationBuffer:
    def __init__(
        self,
        summaries: np.ndarray,
        errors: np.ndarray,
        timestamps: np.ndarray,
        total_seen: int = 0,
    ) -> None:
        self.summaries = np.asarray(summaries, dtype=np.float32)
        self.errors = np.asarray(errors, dtype=np.float32)
        self.timestamps = np.asarray(timestamps, dtype=str)
        self.total_seen = int(total_seen)
        if not (len(self.summaries) == len(self.errors) == len(self.timestamps)):
            raise ValueError("Calibration buffer arrays have different lengths")

    @classmethod
    def empty(cls, feature_count: int) -> "CalibrationBuffer":
        return cls(
            np.empty((0, feature_count), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=str),
            0,
        )

    @classmethod
    def load(cls, path: str | Path, feature_count: int) -> "CalibrationBuffer":
        source = Path(path)
        if not source.exists():
            return cls.empty(feature_count)
        with np.load(source, allow_pickle=False) as arrays:
            return cls(
                arrays["summaries"],
                arrays["errors"],
                arrays["timestamps"],
                int(arrays["total_seen"].item()),
            )

    def append(
        self,
        summaries: np.ndarray,
        errors: np.ndarray,
        timestamps: Iterable[str],
        max_windows: int,
    ) -> None:
        timestamps_array = np.asarray(list(timestamps), dtype=str)
        summaries_array = np.asarray(summaries, dtype=np.float32)
        errors_array = np.asarray(errors, dtype=np.float32)
        if not (len(summaries_array) == len(errors_array) == len(timestamps_array)):
            raise ValueError("Appended calibration arrays have different lengths")
        if len(summaries_array) == 0:
            return
        self.total_seen += len(summaries_array)
        self.summaries = np.concatenate([self.summaries, summaries_array])[-max_windows:]
        self.errors = np.concatenate([self.errors, errors_array])[-max_windows:]
        self.timestamps = np.concatenate([self.timestamps, timestamps_array])[-max_windows:]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            summaries=self.summaries,
            errors=self.errors,
            timestamps=self.timestamps,
            total_seen=np.asarray(self.total_seen, dtype=np.int64),
        )
        temporary.replace(destination)


class AdaptiveRuntime:
    def __init__(self, root: str | Path, config: AdaptiveConfig | None = None) -> None:
        self.root = Path(root)
        config_path = self.root / "adaptive_config.json"
        self.config = config or AdaptiveConfig.load(config_path if config_path.exists() else None)

    @property
    def groups_dir(self) -> Path:
        return self.root / "profiles"

    def initialize_from_seed(self, seed_dir: str | Path) -> None:
        source = Path(seed_dir)
        if not source.exists():
            raise FileNotFoundError(f"Adaptive seed directory not found: {source}")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Adaptive runtime is not empty: {self.root}")
        shutil.copytree(source, self.root, dirs_exist_ok=True)
        self.config = AdaptiveConfig.load(self.root / "adaptive_config.json")

    def group_dir(self, group_name: str) -> Path:
        return self.groups_dir / group_name

    def golden_path(self, group_name: str) -> Path:
        return self.group_dir(group_name) / "golden.json"

    def champion_path(self, group_name: str) -> Path:
        return self.group_dir(group_name) / "champion.json"

    def candidate_path(self, group_name: str) -> Path:
        return self.group_dir(group_name) / "candidate.json"

    def frozen_path(self, group_name: str) -> Path:
        return self.root / "frozen" / f"{group_name}.npz"

    def buffer_path(self, group_name: str) -> Path:
        return self.root / "buffers" / f"{group_name}.npz"

    def load_champion(self, group_name: str) -> CalibrationProfile:
        return CalibrationProfile.load(self.champion_path(group_name))

    def append_eligible(
        self,
        group_name: str,
        summaries: np.ndarray,
        errors: np.ndarray,
        timestamps: Iterable[str],
    ) -> CalibrationBuffer:
        champion = self.load_champion(group_name)
        buffer = CalibrationBuffer.load(self.buffer_path(group_name), len(champion.feature_columns))
        buffer.append(summaries, errors, timestamps, self.config.max_buffer_windows)
        buffer.save(self.buffer_path(group_name))
        return buffer

    def propose_or_advance(self, group_name: str) -> ApprovalDecision:
        champion = self.load_champion(group_name)
        buffer = CalibrationBuffer.load(self.buffer_path(group_name), len(champion.feature_columns))
        candidate_path = self.candidate_path(group_name)
        if candidate_path.exists():
            candidate = CalibrationProfile.load(candidate_path)
            if buffer.total_seen <= candidate.last_validated_buffer_total:
                return ApprovalDecision(
                    "SHADOW_WAIT",
                    False,
                    ["waiting_for_new_eligible_data"],
                    {
                        "shadow_observations": candidate.shadow_observations,
                        "buffer_total_seen": buffer.total_seen,
                    },
                )
        else:
            new_windows = buffer.total_seen - champion.buffer_total_seen
            if new_windows < self.config.min_candidate_windows:
                return ApprovalDecision(
                    "NO_CANDIDATE",
                    False,
                    ["insufficient_new_eligible_windows"],
                    {"new_eligible_windows": max(0, new_windows)},
                )
            candidate = build_candidate_profile(
                champion,
                buffer.summaries,
                buffer.errors,
                buffer.total_seen,
                self.config,
            )

        with np.load(self.frozen_path(group_name), allow_pickle=False) as frozen:
            decision = validate_candidate(
                champion,
                candidate,
                frozen["summaries"],
                frozen["errors"],
                buffer.summaries,
                buffer.errors,
                buffer.timestamps,
                self.config,
            )
        candidate.approval = decision.to_dict()
        candidate.last_validated_buffer_total = buffer.total_seen
        if not decision.passed:
            if decision.outcome == "AUTO_REJECTED":
                self._archive_profile(candidate, "rejected")
                candidate_path.unlink(missing_ok=True)
            else:
                candidate.status = "REVIEW_REQUIRED"
                candidate.save(candidate_path)
            self._append_audit(group_name, candidate.profile_version, decision)
            return decision

        candidate.status = "SHADOW"
        candidate.shadow_observations += 1
        if candidate.shadow_observations >= self.config.shadow_min_observations:
            decision = ApprovalDecision(
                "AUTO_APPROVED",
                True,
                ["validation_and_shadow_gates_passed"],
                {**decision.metrics, "shadow_observations": candidate.shadow_observations},
            )
            candidate.status = "AUTO_APPROVED"
            candidate.buffer_total_seen = buffer.total_seen
            candidate.approval = decision.to_dict()
            candidate.approval["deployment_baseline"] = {
                "reference_alert_rate": float(
                    decision.metrics["candidate_reference_alert_rate"]
                ),
                "synthetic_detection_rate": float(
                    decision.metrics["candidate_synthetic_detection_rate"]
                ),
            }
            self._archive_profile(champion, "history")
            candidate.save(self.champion_path(group_name))
            candidate_path.unlink(missing_ok=True)
        else:
            candidate.save(candidate_path)
        self._append_audit(group_name, candidate.profile_version, decision)
        return decision

    def self_test_champion(self, group_name: str) -> ApprovalDecision:
        """Verify the deployed profile against its immutable frozen holdout.

        Automatic rollback is limited to calibration-regression failures. A
        machine anomaly in live data never triggers rollback because that could
        hide a real fault.
        """

        champion = self.load_champion(group_name)
        with np.load(self.frozen_path(group_name), allow_pickle=False) as frozen:
            summaries = frozen["summaries"]
            errors = frozen["errors"]
        reference = score_profile(champion, summaries, errors, self.config)
        reference_alert_rate = float(np.mean(reference["operational_risk"] >= 1.0))

        synthetic = summaries.copy()
        sensor_count = min(7, synthetic.shape[1])
        golden_scale = np.asarray(champion.golden_scale, dtype=np.float64)
        for index in range(len(synthetic)):
            feature_index = index % sensor_count
            synthetic[index, feature_index] += (
                self.config.synthetic_shift_mad * golden_scale[feature_index]
            )
        synthetic_result = score_profile(champion, synthetic, errors, self.config)
        synthetic_detection = float(
            np.mean(synthetic_result["operational_risk"] >= 1.0)
        )
        metrics: dict[str, float | int | str] = {
            "reference_alert_rate": reference_alert_rate,
            "synthetic_detection_rate": synthetic_detection,
        }
        deployment_baseline = champion.approval.get("deployment_baseline", {})
        baseline_alert_rate = float(
            deployment_baseline.get("reference_alert_rate", reference_alert_rate)
        )
        baseline_detection = float(
            deployment_baseline.get("synthetic_detection_rate", synthetic_detection)
        )
        allowed_alert_rate = max(
            self.config.max_reference_alert_rate, baseline_alert_rate + 0.02
        )
        required_detection = min(
            self.config.min_synthetic_detection_rate,
            max(0.0, baseline_detection - self.config.max_synthetic_detection_drop),
        )
        metrics.update(
            {
                "allowed_reference_alert_rate": allowed_alert_rate,
                "required_synthetic_detection_rate": required_detection,
            }
        )
        failures: list[str] = []
        if reference_alert_rate > allowed_alert_rate:
            failures.append("deployed_reference_alert_rate_failed")
        if synthetic_detection < required_detection:
            failures.append("deployed_synthetic_detection_failed")
        if not failures:
            return ApprovalDecision(
                "CHAMPION_HEALTHY", True, ["frozen_self_test_passed"], metrics
            )

        history_dir = self.group_dir(group_name) / "history"
        history_files = sorted(
            history_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True
        ) if history_dir.exists() else []
        if not history_files:
            decision = ApprovalDecision(
                "REVIEW_REQUIRED",
                False,
                failures + ["no_rollback_profile_available"],
                metrics,
            )
            self._append_audit(group_name, champion.profile_version, decision)
            return decision

        previous = CalibrationProfile.load(history_files[0])
        self._archive_profile(champion, "failed_champions")
        previous.status = "AUTO_ROLLED_BACK"
        previous.created_at_utc = utc_now()
        previous.approval = {
            "outcome": "AUTO_ROLLED_BACK",
            "reasons": failures,
            "failed_profile_version": champion.profile_version,
        }
        previous.save(self.champion_path(group_name))
        decision = ApprovalDecision(
            "AUTO_ROLLED_BACK",
            True,
            failures + ["restored_previous_champion"],
            {**metrics, "restored_profile_version": previous.profile_version},
        )
        self._append_audit(group_name, champion.profile_version, decision)
        return decision

    def _archive_profile(self, profile: CalibrationProfile, category: str) -> None:
        profile.save(
            self.group_dir(profile.group_name)
            / category
            / f"{profile.profile_version}.json"
        )

    def _append_audit(
        self,
        group_name: str,
        profile_version: str,
        decision: ApprovalDecision,
    ) -> None:
        path = self.root / "audit" / "approval_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": utc_now(),
            "group_name": group_name,
            "profile_version": profile_version,
            **decision.to_dict(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
