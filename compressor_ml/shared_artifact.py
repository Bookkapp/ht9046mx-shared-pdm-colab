"""Immutable Shared LSTM artifact loader used by the production shadow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .anomaly import StandardScaler3D, reconstruction_error
from .config import PipelineConfig
from .model import require_tensorflow


class SharedArtifactBundle:
    """Load a trained Shared LSTM and its immutable group scalers."""

    def __init__(self, artifact_dir: str | Path, model: Any | None = None) -> None:
        self.root = Path(artifact_dir)
        self.config = PipelineConfig.load(self.root / "config.json")
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.thresholds = json.loads((self.root / "thresholds.json").read_text(encoding="utf-8"))
        self.model_version = str(self.manifest.get("model_version", "shared_lstm_unknown"))
        if model is None:
            tensorflow = require_tensorflow()
            model = tensorflow.keras.models.load_model(self.root / "shared_model.keras")
        self.model = model
        self._scalers: dict[str, StandardScaler3D] = {}

    def has_group(self, group_name: str) -> bool:
        return group_name in self.thresholds and (self.root / "scalers" / f"{group_name}.npz").exists()

    def scaler(self, group_name: str) -> StandardScaler3D:
        if group_name not in self._scalers:
            self._scalers[group_name] = StandardScaler3D.load(
                str(self.root / "scalers" / f"{group_name}.npz")
            )
        return self._scalers[group_name]

    def threshold(self, group_name: str) -> float:
        return float(self.thresholds[group_name]["value"])

    def reconstruct(
        self,
        group_name: str,
        raw_windows: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.scaler(group_name).transform(raw_windows)
        predicted = self.model.predict(scaled, batch_size=batch_size, verbose=0)
        return reconstruction_error(scaled, predicted)
