from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

from .config import ControlledMonitoringConfig
from .context import RegimeModel, add_operating_modes, fit_regime_models
from .types import Evidence


EPSILON = 1e-6
RIDGE_FEATURES = ("hp2", "valve", "temphi", "templo")
ROBUST_TRIGGER_FEATURES = ("hp2", "pressure_gap", "temperature_span")


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("Robust baseline needs finite observations")
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    return center, max(scale, EPSILON)


@dataclass
class ContextProfile:
    operating_mode: str
    regime: str
    training_windows: int
    feature_center: dict[str, float]
    feature_scale: dict[str, float]
    ridge_intercept: float
    ridge_coefficients: list[float]
    residual_center: float
    residual_scale: float
    isolation_model: IsolationForest
    isolation_entry_threshold: float
    isolation_exit_threshold: float

    @property
    def key(self) -> str:
        return f"{self.operating_mode}::{self.regime}"


@dataclass
class FrozenProfileBundle:
    machine_id: str
    module_id: int
    profile_version: str
    created_at_utc: str
    policy_version: str
    status: str
    source_window_start: str
    source_window_end: str
    source_windows: int
    regime_models: dict[str, RegimeModel]
    contexts: dict[str, ContextProfile]
    lstm_calibration: Any | None = None
    selection_metrics: dict[str, Any] = field(default_factory=dict)

    def context(self, operating_mode: str, regime: str) -> ContextProfile | None:
        return self.contexts.get(f"{operating_mode}::{regime}")


def _ridge_fit(frame: pd.DataFrame, alpha: float) -> tuple[float, np.ndarray, np.ndarray]:
    x = frame[list(RIDGE_FEATURES)].to_numpy(dtype=np.float64)
    y = frame["lp2"].to_numpy(dtype=np.float64)
    if len(frame) >= 25 and np.linalg.matrix_rank(x) >= 2:
        model = Ridge(alpha=alpha).fit(x, y)
        prediction = model.predict(x)
        return float(model.intercept_), np.asarray(model.coef_, dtype=np.float64), y - prediction
    intercept = float(np.median(y))
    coefficients = np.zeros(len(RIDGE_FEATURES), dtype=np.float64)
    return intercept, coefficients, y - intercept


def _residual(
    frame: pd.DataFrame,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    x = frame[list(RIDGE_FEATURES)].to_numpy(dtype=np.float64)
    return frame["lp2"].to_numpy(dtype=np.float64) - (intercept + x @ coefficients)


def fit_context_profile(
    frame: pd.DataFrame,
    operating_mode: str,
    regime: str,
    config: ControlledMonitoringConfig,
) -> ContextProfile:
    if len(frame) < config.profile_min_context_windows:
        raise ValueError(
            f"Context {operating_mode}/{regime} has {len(frame)} windows; "
            f"need {config.profile_min_context_windows}"
        )
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in config.profile_features:
        centers[feature], scales[feature] = robust_center_scale(
            frame[feature].to_numpy(dtype=np.float64)
        )
    intercept, coefficients, residuals = _ridge_fit(frame, config.ridge_alpha)
    residual_center, residual_scale = robust_center_scale(residuals)
    z_hp2 = (frame["hp2"].to_numpy(dtype=np.float64) - centers["hp2"]) / scales["hp2"]
    z_lp2 = (residuals - residual_center) / residual_scale
    z_gap = (
        frame["pressure_gap"].to_numpy(dtype=np.float64) - centers["pressure_gap"]
    ) / scales["pressure_gap"]
    z_temperature = (
        frame["temperature_span"].to_numpy(dtype=np.float64)
        - centers["temperature_span"]
    ) / scales["temperature_span"]
    vectors = np.column_stack([z_hp2, z_lp2, z_gap, z_temperature])
    isolation = IsolationForest(
        n_estimators=config.isolation_estimators,
        contamination="auto",
        random_state=config.random_seed,
    ).fit(vectors)
    scores = -isolation.score_samples(vectors)
    return ContextProfile(
        operating_mode=operating_mode,
        regime=regime,
        training_windows=int(len(frame)),
        feature_center=centers,
        feature_scale=scales,
        ridge_intercept=intercept,
        ridge_coefficients=coefficients.tolist(),
        residual_center=residual_center,
        residual_scale=residual_scale,
        isolation_model=isolation,
        isolation_entry_threshold=float(np.quantile(scores, config.isolation_entry_quantile)),
        isolation_exit_threshold=float(np.quantile(scores, config.isolation_exit_quantile)),
    )


def _z_score(value: float, center: float, scale: float) -> float:
    return float((value - center) / max(scale, EPSILON))


def score_context_profile(
    profile: ContextProfile,
    row: pd.Series | dict[str, Any],
    config: ControlledMonitoringConfig,
    *,
    active_reason_codes: set[str] | None = None,
) -> Evidence:
    active_reasons = active_reason_codes or set()
    coefficients = np.asarray(profile.ridge_coefficients, dtype=np.float64)
    ridge_x = np.asarray([float(row[column]) for column in RIDGE_FEATURES])
    predicted_lp2 = float(profile.ridge_intercept + ridge_x @ coefficients)
    residual_lp2 = float(row["lp2"]) - predicted_lp2
    z_hp2 = _z_score(
        float(row["hp2"]), profile.feature_center["hp2"], profile.feature_scale["hp2"]
    )
    z_lp2 = _z_score(residual_lp2, profile.residual_center, profile.residual_scale)
    z_gap = _z_score(
        float(row["pressure_gap"]),
        profile.feature_center["pressure_gap"],
        profile.feature_scale["pressure_gap"],
    )
    z_temperature = _z_score(
        float(row["temperature_span"]),
        profile.feature_center["temperature_span"],
        profile.feature_scale["temperature_span"],
    )
    vector = np.asarray([[z_hp2, z_lp2, z_gap, z_temperature]], dtype=np.float64)
    isolation_score = float(-profile.isolation_model.score_samples(vector)[0])
    reasons: list[str] = []

    def absolute_trigger(code: str, score: float) -> bool:
        threshold = config.robust_exit_z if code in active_reasons else config.robust_entry_z
        return abs(score) >= threshold

    if absolute_trigger("HP2_ROBUST_Z", z_hp2):
        reasons.append("HP2_ROBUST_Z")
    lp2_threshold = config.robust_exit_z if "LP2_NEGATIVE_RESIDUAL" in active_reasons else config.robust_entry_z
    if z_lp2 <= -lp2_threshold:
        reasons.append("LP2_NEGATIVE_RESIDUAL")
    if absolute_trigger("PRESSURE_GAP_ANOMALY", z_gap):
        reasons.append("PRESSURE_GAP_ANOMALY")
    if absolute_trigger("TEMPERATURE_SPAN_ANOMALY", z_temperature):
        reasons.append("TEMPERATURE_SPAN_ANOMALY")
    isolation_threshold = (
        profile.isolation_exit_threshold
        if "PRESSURE_PATTERN_ANOMALY" in active_reasons
        else profile.isolation_entry_threshold
    )
    if isolation_score >= isolation_threshold:
        reasons.append("PRESSURE_PATTERN_ANOMALY")
    return Evidence(
        model="COM2",
        active=bool(reasons),
        score=max(abs(z_hp2), abs(z_lp2), abs(z_gap), abs(z_temperature)),
        threshold=config.robust_entry_z,
        reason_codes=reasons,
        details={
            "predicted_lp2": predicted_lp2,
            "lp2_residual": residual_lp2,
            "z_hp2": z_hp2,
            "z_lp2_residual": z_lp2,
            "z_pressure_gap": z_gap,
            "z_temperature_span": z_temperature,
            "isolation_score": isolation_score,
            "isolation_entry_threshold": profile.isolation_entry_threshold,
            "isolation_exit_threshold": profile.isolation_exit_threshold,
        },
    )


def fit_frozen_profile_bundle(
    frame: pd.DataFrame,
    machine_id: str,
    module_id: int,
    config: ControlledMonitoringConfig,
    *,
    profile_version: str,
    status: str = "CANDIDATE",
    selection_metrics: dict[str, Any] | None = None,
) -> FrozenProfileBundle:
    subset = frame.loc[
        frame["machine_id"].eq(machine_id) & frame["module_id"].eq(module_id)
    ].copy()
    if subset.empty:
        raise ValueError(f"No windows for {machine_id} module {module_id}")
    subset = add_operating_modes(subset, config)
    regime_models = fit_regime_models(subset, config)
    regime_labels: list[str] = []
    for _, row in subset.iterrows():
        model = regime_models[str(row["operating_mode"])]
        resolution = model.resolve(row, apply_policy_likelihood_floor=False)
        regime_labels.append(resolution.regime)
    subset["regime"] = regime_labels
    contexts: dict[str, ContextProfile] = {}
    for (mode, regime), group in subset.groupby(["operating_mode", "regime"], sort=True):
        if regime == "UNKNOWN_REGIME" or len(group) < config.profile_min_context_windows:
            continue
        profile = fit_context_profile(group, str(mode), str(regime), config)
        contexts[profile.key] = profile
    if not contexts:
        raise ValueError(f"No stable contexts could be fitted for {machine_id} module {module_id}")
    event_times = pd.to_datetime(subset["event_time"])
    return FrozenProfileBundle(
        machine_id=machine_id,
        module_id=int(module_id),
        profile_version=profile_version,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        policy_version=config.policy_version,
        status=status,
        source_window_start=event_times.min().isoformat(),
        source_window_end=event_times.max().isoformat(),
        source_windows=int(len(subset)),
        regime_models=regime_models,
        contexts=contexts,
        selection_metrics=selection_metrics or {},
    )
