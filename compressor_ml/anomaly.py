from __future__ import annotations

import numpy as np


class StandardScaler3D:
    """A serializable scaler fitted on training windows only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler3D":
        if values.ndim != 3 or len(values) == 0:
            raise ValueError("Expected non-empty array of shape (samples, time, features)")
        flat = values.reshape(-1, values.shape[-1])
        self.mean_ = flat.mean(axis=0, dtype=np.float64)
        self.scale_ = flat.std(axis=0, dtype=np.float64)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform")
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def save(self, path: str) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot save an unfitted scaler")
        np.savez(path, mean=self.mean_, scale=self.scale_)

    @classmethod
    def load(cls, path: str) -> "StandardScaler3D":
        data = np.load(path)
        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.scale_ = data["scale"]
        return scaler


def reconstruction_error(actual: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return window MAE and per-feature MAE for explainable anomaly output."""
    absolute = np.abs(actual - reconstructed)
    return absolute.mean(axis=(1, 2)), absolute.mean(axis=1)


def anomaly_score(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Monotonic severity score; values above threshold retain different severity."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return 1.0 - np.exp(-np.maximum(errors, 0.0) / threshold)


def health_score(scores: np.ndarray, smoothing_windows: int) -> np.ndarray:
    raw = 100.0 * (1.0 - np.clip(scores, 0.0, 1.0))
    if smoothing_windows <= 1:
        return raw
    result = np.empty_like(raw)
    for index in range(len(raw)):
        result[index] = raw[max(0, index - smoothing_windows + 1): index + 1].mean()
    return result


def pseudo_label(health: float, normal_min: float = 80, watch_min: float = 60, warning_min: float = 40) -> str:
    if health >= normal_min:
        return "Normal"
    if health >= watch_min:
        return "Watch"
    if health >= warning_min:
        return "Warning"
    return "Critical"
