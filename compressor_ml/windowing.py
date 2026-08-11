from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def make_windows(frame: pd.DataFrame, config: PipelineConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Create windows without crossing module, state-cleaning, or time boundaries."""
    columns = list(config.feature_columns)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing engineered feature columns: {sorted(missing)}")
    sequences: list[np.ndarray] = []
    metadata: list[dict] = []
    for (machine_id, module_id, segment_id), group in frame.groupby(["machine_id", "module_id", "segment_id"], sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        values = group[columns].to_numpy(dtype=np.float32)
        stamps = group["timestamp"].to_numpy()
        required = config.window_rows
        for start in range(0, len(group) - required + 1, config.step_rows):
            stop = start + required
            window = values[start:stop]
            expected = pd.Timestamp(stamps[start]) + pd.to_timedelta((required - 1) * config.sampling_interval_sec, unit="s")
            if pd.Timestamp(stamps[stop - 1]) != expected or not np.isfinite(window).all():
                continue
            sequences.append(window)
            metadata.append({"machine_id": machine_id, "module_id": int(module_id), "segment_id": int(segment_id), "window_start": str(stamps[start]), "timestamp": str(stamps[stop - 1])})
    if not sequences:
        return np.empty((0, config.window_rows, len(columns)), dtype=np.float32), pd.DataFrame(metadata)
    return np.stack(sequences), pd.DataFrame(metadata)


def chronological_split(windows: np.ndarray, metadata: pd.DataFrame) -> tuple[tuple[np.ndarray, pd.DataFrame], tuple[np.ndarray, pd.DataFrame], tuple[np.ndarray, pd.DataFrame]]:
    if len(windows) < 10:
        raise ValueError("Need at least 10 valid windows for chronological train/validation/test split")
    order = metadata.assign(_time=pd.to_datetime(metadata["timestamp"])).sort_values("_time").index.to_numpy()
    windows, metadata = windows[order], metadata.iloc[order].reset_index(drop=True)
    train_end = max(1, int(len(windows) * 0.70))
    valid_end = max(train_end + 1, int(len(windows) * 0.85))
    return (windows[:train_end], metadata.iloc[:train_end]), (windows[train_end:valid_end], metadata.iloc[train_end:valid_end]), (windows[valid_end:], metadata.iloc[valid_end:])
