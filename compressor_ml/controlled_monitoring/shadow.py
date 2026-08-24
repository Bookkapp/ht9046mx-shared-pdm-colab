from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..shared_artifact import SharedArtifactBundle
from ..anomaly import StandardScaler3D, reconstruction_error
from ..features import engineer_features
from ..prepare_dataset import safe_group_name
from ..preprocessing import validate_and_filter
from ..windowing import make_windows
from .config import ControlledMonitoringConfig
from .types import Evidence


@dataclass
class LocalShadowCalibration:
    group_name: str
    scaler: StandardScaler3D
    threshold: float
    training_windows: int
    validation_windows: int
    model_version: str


class SharedLSTMShadow:
    """Immutable Shared LSTM evidence; never updates model weights from live data."""

    def __init__(
        self,
        artifact_dir: str | None = None,
        *,
        bundle: SharedArtifactBundle | Any | None = None,
        config: ControlledMonitoringConfig | None = None,
    ) -> None:
        if bundle is None:
            if artifact_dir is None:
                raise ValueError("artifact_dir or bundle is required")
            bundle = SharedArtifactBundle(artifact_dir)
        self.bundle = bundle
        self.config = config or ControlledMonitoringConfig()

    def _raw_windows(
        self,
        raw_frame: pd.DataFrame,
        machine_id: str,
        module_id: int,
    ) -> np.ndarray:
        subset = raw_frame.loc[
            raw_frame["machine_id"].eq(machine_id) & raw_frame["module_id"].eq(module_id)
        ].copy()
        if subset.empty:
            return np.empty(
                (0, self.bundle.config.window_rows, len(self.bundle.config.feature_columns)),
                dtype=np.float32,
            )
        valid, _ = validate_and_filter(subset, self.bundle.config)
        featured = engineer_features(valid, self.bundle.config)
        windows, _ = make_windows(featured, self.bundle.config)
        return windows

    def fit_local_calibration(
        self,
        raw_history: pd.DataFrame,
        machine_id: str,
        module_id: int,
    ) -> LocalShadowCalibration:
        windows = self._raw_windows(raw_history, machine_id, module_id)
        if len(windows) < 20:
            raise ValueError("Local LSTM calibration needs at least 20 valid sequences")
        train_end = max(1, int(len(windows) * 0.70))
        valid_end = max(train_end + 1, int(len(windows) * 0.85))
        scaler = StandardScaler3D().fit(windows[:train_end])
        validation = scaler.transform(windows[train_end:valid_end])
        prediction = self.bundle.model.predict(validation, batch_size=256, verbose=0)
        errors, _ = reconstruction_error(validation, prediction)
        threshold = float(np.quantile(errors, 0.99))
        return LocalShadowCalibration(
            group_name=safe_group_name(machine_id, module_id),
            scaler=scaler,
            threshold=max(threshold, 1e-8),
            training_windows=train_end,
            validation_windows=len(validation),
            model_version=str(self.bundle.model_version),
        )

    def score_bucket(
        self,
        raw_bucket: pd.DataFrame,
        machine_id: str,
        module_id: int,
        *,
        local_calibration: LocalShadowCalibration | None = None,
    ) -> Evidence:
        group_name = safe_group_name(machine_id, module_id)
        windows = self._raw_windows(raw_bucket, machine_id, module_id)
        if len(windows) < self.config.lstm_min_sequences:
            return Evidence(
                model="LSTM",
                active=False,
                reason_codes=["LSTM_SHADOW_INSUFFICIENT_SEQUENCES"],
                details={"sequence_count": int(len(windows)), "configured": True},
            )
        if local_calibration is not None:
            scaled = local_calibration.scaler.transform(windows)
            prediction = self.bundle.model.predict(scaled, batch_size=256, verbose=0)
            errors, feature_errors = reconstruction_error(scaled, prediction)
            threshold = local_calibration.threshold
            calibration_source = "LOCAL_BOOTSTRAP"
        elif self.bundle.has_group(group_name):
            errors, feature_errors = self.bundle.reconstruct(group_name, windows, batch_size=256)
            threshold = self.bundle.threshold(group_name)
            calibration_source = "TRAINED_GROUP"
        else:
            return Evidence(
                model="LSTM",
                active=False,
                reason_codes=["LSTM_SHADOW_GROUP_NOT_CONFIGURED"],
                details={"sequence_count": int(len(windows)), "configured": False},
            )
        score = float(np.quantile(errors, self.config.lstm_bucket_quantile))
        top_index = int(np.argmax(np.mean(feature_errors, axis=0)))
        top_feature = str(self.bundle.config.feature_columns[top_index])
        active = score >= threshold
        return Evidence(
            model="LSTM",
            active=active,
            score=score,
            threshold=float(threshold),
            reason_codes=["LSTM_RECONSTRUCTION_ANOMALY"] if active else [],
            details={
                "sequence_count": int(len(windows)),
                "error_quantile": self.config.lstm_bucket_quantile,
                "exceedance_fraction": float(np.mean(errors >= threshold)),
                "top_error_feature": top_feature,
                "calibration_source": calibration_source,
                "model_version": str(self.bundle.model_version),
            },
        )
