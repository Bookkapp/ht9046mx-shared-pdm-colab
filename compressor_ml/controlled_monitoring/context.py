from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .config import ControlledMonitoringConfig


def _token(value: object) -> str:
    return str(value).strip().lower()


def operating_mode(sv: object, valve: float, config: ControlledMonitoringConfig) -> str:
    sv_state = "SV_ON" if _token(sv) in config.sv_on_tokens else "SV_OFF"
    boundaries = config.valve_bins
    if valve < boundaries[0]:
        bucket = "VALVE_B0"
    elif valve < boundaries[1]:
        bucket = "VALVE_B1"
    elif valve < boundaries[2]:
        bucket = "VALVE_B2"
    else:
        bucket = "VALVE_B3"
    return f"{sv_state}_{bucket}"


def add_operating_modes(frame: pd.DataFrame, config: ControlledMonitoringConfig) -> pd.DataFrame:
    result = frame.copy()
    result["operating_mode"] = [
        operating_mode(sv, float(valve), config)
        for sv, valve in zip(result["sv"], result["valve"], strict=True)
    ]
    return result


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))).squeeze(axis)


@dataclass
class RegimeResolution:
    regime: str
    posterior: float
    log_likelihood: float
    reason: str | None = None


@dataclass
class RegimeModel:
    operating_mode: str
    feature_columns: list[str]
    component_count: int
    weights: list[float]
    means: list[list[float]]
    covariances: list[list[list[float]]]
    likelihood_floor: float
    min_posterior: float
    policy_min_log_likelihood: float
    fallback_single_regime: bool
    training_windows: int
    selected_bic: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RegimeModel":
        return cls(**payload)

    def _component_log_probabilities(self, values: np.ndarray) -> np.ndarray:
        features = values.shape[1]
        result = np.empty((len(values), self.component_count), dtype=np.float64)
        for component in range(self.component_count):
            mean = np.asarray(self.means[component], dtype=np.float64)
            covariance = np.asarray(self.covariances[component], dtype=np.float64)
            covariance = covariance + np.eye(features) * 1e-6
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0:
                covariance = covariance + np.eye(features) * 1e-3
                _, log_determinant = np.linalg.slogdet(covariance)
            inverse = np.linalg.pinv(covariance, hermitian=True)
            centered = values - mean
            mahalanobis = np.einsum("ij,jk,ik->i", centered, inverse, centered)
            normalizer = features * np.log(2.0 * np.pi) + log_determinant
            result[:, component] = (
                np.log(max(float(self.weights[component]), 1e-12))
                - 0.5 * (normalizer + mahalanobis)
            )
        return result

    def resolve_many(
        self,
        values: np.ndarray,
        *,
        apply_policy_likelihood_floor: bool = True,
    ) -> list[RegimeResolution]:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_columns):
            raise ValueError("Regime values have the wrong shape")
        if self.fallback_single_regime:
            return [RegimeResolution("R0", 1.0, 0.0) for _ in range(len(matrix))]
        component_logs = self._component_log_probabilities(matrix)
        mixture_logs = _logsumexp(component_logs)
        posterior = np.exp(component_logs - mixture_logs[:, None])
        winners = np.argmax(posterior, axis=1)
        effective_floor = self.likelihood_floor
        if apply_policy_likelihood_floor:
            effective_floor = max(effective_floor, self.policy_min_log_likelihood)
        resolutions: list[RegimeResolution] = []
        for index, component in enumerate(winners):
            probability = float(posterior[index, component])
            likelihood = float(mixture_logs[index])
            if probability < self.min_posterior:
                resolutions.append(
                    RegimeResolution("UNKNOWN_REGIME", probability, likelihood, "LOW_POSTERIOR")
                )
            elif likelihood < effective_floor:
                resolutions.append(
                    RegimeResolution("UNKNOWN_REGIME", probability, likelihood, "LOW_LIKELIHOOD")
                )
            else:
                resolutions.append(RegimeResolution(f"R{int(component)}", probability, likelihood))
        return resolutions

    def resolve(
        self,
        row: pd.Series | dict[str, Any],
        *,
        apply_policy_likelihood_floor: bool = True,
    ) -> RegimeResolution:
        values = np.asarray([[float(row[column]) for column in self.feature_columns]])
        return self.resolve_many(
            values, apply_policy_likelihood_floor=apply_policy_likelihood_floor
        )[0]


def fit_regime_model(
    frame: pd.DataFrame,
    operating_mode_name: str,
    config: ControlledMonitoringConfig,
) -> RegimeModel:
    clean = frame.dropna(subset=list(config.regime_features))
    matrix = clean[list(config.regime_features)].to_numpy(dtype=np.float64)
    if len(matrix) == 0:
        raise ValueError(f"No regime data available for {operating_mode_name}")
    if len(matrix) < config.regime_min_windows:
        covariance = np.cov(matrix, rowvar=False, ddof=0) if len(matrix) > 1 else np.eye(matrix.shape[1])
        return RegimeModel(
            operating_mode=operating_mode_name,
            feature_columns=list(config.regime_features),
            component_count=1,
            weights=[1.0],
            means=[np.median(matrix, axis=0).tolist()],
            covariances=[np.atleast_2d(covariance).tolist()],
            likelihood_floor=float("-inf"),
            min_posterior=config.regime_min_posterior,
            policy_min_log_likelihood=config.regime_min_log_likelihood,
            fallback_single_regime=True,
            training_windows=int(len(matrix)),
            selected_bic=None,
        )
    maximum = min(config.regime_max_components, max(1, len(matrix) // 20))
    candidates: list[tuple[float, GaussianMixture]] = []
    for components in range(1, maximum + 1):
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            n_init=5,
            reg_covar=1e-5,
            random_state=config.random_seed,
        ).fit(matrix)
        candidates.append((float(model.bic(matrix)), model))
    selected_bic, selected = min(candidates, key=lambda item: item[0])
    ordering = np.lexsort((selected.means_[:, 0], selected.means_[:, 2]))
    weights = selected.weights_[ordering]
    means = selected.means_[ordering]
    covariances = selected.covariances_[ordering]
    provisional = RegimeModel(
        operating_mode=operating_mode_name,
        feature_columns=list(config.regime_features),
        component_count=int(selected.n_components),
        weights=weights.tolist(),
        means=means.tolist(),
        covariances=covariances.tolist(),
        likelihood_floor=float("-inf"),
        min_posterior=config.regime_min_posterior,
        policy_min_log_likelihood=config.regime_min_log_likelihood,
        fallback_single_regime=False,
        training_windows=int(len(matrix)),
        selected_bic=selected_bic,
    )
    component_logs = provisional._component_log_probabilities(matrix)
    likelihoods = _logsumexp(component_logs)
    provisional.likelihood_floor = float(
        np.quantile(likelihoods, config.regime_likelihood_quantile)
    )
    return provisional


def fit_regime_models(
    frame: pd.DataFrame,
    config: ControlledMonitoringConfig,
) -> dict[str, RegimeModel]:
    with_modes = add_operating_modes(frame, config)
    return {
        str(mode): fit_regime_model(group, str(mode), config)
        for mode, group in with_modes.groupby("operating_mode", sort=True)
    }
