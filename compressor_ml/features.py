from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def engineer_features(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create only physically interpretable features on already stable samples."""
    data = frame.copy().sort_values(["machine_id", "module_id", "timestamp"]).reset_index(drop=True)
    if data.empty:
        for column in config.feature_columns:
            if column not in data:
                data[column] = pd.Series(dtype=float)
        return data
    data["hp_gap_1"] = data["hp1"] - data["lp1"]
    data["hp_gap_2"] = data["hp2"] - data["lp2"]
    eps = 1e-6
    data["hp_lp_ratio_1"] = data["hp1"] / data["lp1"].abs().clip(lower=eps)
    data["hp_lp_ratio_2"] = data["hp2"] / data["lp2"].abs().clip(lower=eps)
    groups = data.groupby(["machine_id", "module_id", "segment_id"], sort=False)
    for source, target in (("hp1", "delta_hp1"), ("lp1", "delta_lp1"), ("hp2", "delta_hp2"), ("lp2", "delta_lp2"), ("valve", "delta_valve")):
        data[target] = groups[source].diff()
    window = config.window_rows
    for source in ("hp1", "hp2", "lp2", "templo"):
        rolling = groups[source].rolling(window=window, min_periods=window)
        data[f"{source}_mean_60"] = rolling.mean().reset_index(level=[0, 1, 2], drop=True)
        data[f"{source}_std_60"] = rolling.std(ddof=0).reset_index(level=[0, 1, 2], drop=True).fillna(0.0)
    return data.replace([np.inf, -np.inf], np.nan)
