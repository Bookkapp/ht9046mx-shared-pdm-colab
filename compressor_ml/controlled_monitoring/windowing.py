from __future__ import annotations

from datetime import timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ControlledMonitoringConfig
from .types import WindowStatus


SENSOR_COLUMNS = ("hp1", "lp1", "hp2", "lp2", "valve", "temphi", "templo")
PRESSURE_COLUMNS = ("hp1", "lp1", "hp2", "lp2")
TEMPERATURE_COLUMNS = ("temphi", "templo")


def _token(value: object) -> str:
    return str(value).strip().lower()


def _series_mode(values: pd.Series, default: str = "<blank>") -> str:
    cleaned = values.dropna().astype(str).str.strip()
    if cleaned.empty:
        return default
    modes = cleaned.mode()
    return str(modes.iloc[0] if not modes.empty else cleaned.iloc[-1])


def add_com2_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["pressure_gap"] = result["hp2"] - result["lp2"]
    result["pressure_ratio"] = result["hp2"] / result["lp2"].abs().clip(lower=0.1)
    result["temperature_span"] = result["temphi"] - result["templo"]
    return result


def _prepare_raw(frame: pd.DataFrame, config: ControlledMonitoringConfig) -> pd.DataFrame:
    required = {
        "timestamp",
        "machine_id",
        "module_id",
        "global_status",
        "module_status",
        "busy",
        "sv",
        *SENSOR_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Raw long-format frame is missing columns: {sorted(missing)}")
    data = frame.copy().reset_index(drop=True)
    data["_source_order"] = np.arange(len(data))
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    local_zone = ZoneInfo(config.timezone)
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(local_zone, nonexistent="shift_forward", ambiguous="NaT")
    else:
        timestamps = timestamps.dt.tz_convert(local_zone)
    data["event_time"] = timestamps
    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["busy"] = pd.to_numeric(data["busy"], errors="coerce")
    original = data.sort_values(["machine_id", "module_id", "_source_order"])
    source_diff = original.groupby(["machine_id", "module_id"], sort=False)["event_time"].diff()
    original["_out_of_order"] = source_diff.dt.total_seconds().lt(0)
    data = original.sort_values(["machine_id", "module_id", "event_time", "_source_order"]).reset_index(drop=True)
    module_token = data["module_status"].map(_token)
    data["_transition"] = module_token.isin(config.transition_states) | data["busy"].eq(1)
    data["_transition_time"] = data["event_time"].where(data["_transition"])
    last_transition = data.groupby(["machine_id", "module_id"], sort=False)["_transition_time"].ffill()
    elapsed = (data["event_time"] - last_transition).dt.total_seconds()
    data["_settling"] = elapsed.ge(0) & elapsed.lt(config.settling_seconds)
    data["window_start"] = data["event_time"].dt.floor(f"{config.window_seconds}s")
    return data


def build_event_windows(
    raw_frame: pd.DataFrame,
    config: ControlledMonitoringConfig,
    *,
    processing_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate raw event-time points into auditable five-minute COM2 windows."""

    data = _prepare_raw(raw_frame, config)
    if data.empty:
        return pd.DataFrame()
    processing = processing_time or pd.Timestamp.now(tz=timezone.utc)
    rows: list[dict[str, object]] = []
    group_columns = ["machine_id", "module_id", "window_start"]
    for (machine_id, module_id, window_start), group in data.groupby(group_columns, sort=True):
        reasons: list[str] = []
        group = group.sort_values(["event_time", "_source_order"])
        point_count = int(len(group))
        duplicate = bool(group["event_time"].duplicated(keep=False).any())
        out_of_order = bool(group["_out_of_order"].any())
        finite = np.isfinite(group[list(SENSOR_COLUMNS)].to_numpy(dtype=np.float64)).all()
        sentinel = bool(group[list(TEMPERATURE_COLUMNS)].isin(config.sentinel_values).any(axis=None))
        pressure_values = group[list(PRESSURE_COLUMNS)]
        pressure_invalid = bool(
            (~pressure_values.ge(config.pressure_min)).any(axis=None)
            or (~pressure_values.le(config.pressure_max)).any(axis=None)
        )
        temperature_values = group[list(TEMPERATURE_COLUMNS)]
        temperature_invalid = bool(
            (~temperature_values.gt(config.temperature_min)).any(axis=None)
            or (~temperature_values.lt(config.temperature_max)).any(axis=None)
        )
        valve_invalid = bool(group["valve"].lt(0).any())
        unique_times = group["event_time"].drop_duplicates().sort_values()
        intervals = unique_times.diff().dt.total_seconds().dropna()
        positive_intervals = intervals[intervals.gt(0)]
        median_interval = float(positive_intervals.median()) if not positive_intervals.empty else np.nan
        maximum_gap = float(positive_intervals.max()) if not positive_intervals.empty else 0.0
        expected_points = (
            max(1, int(np.floor(config.window_seconds / median_interval)))
            if np.isfinite(median_interval) and median_interval > 0
            else config.minimum_window_points
        )
        coverage = float(point_count / expected_points)

        module_tokens = group["module_status"].map(_token)
        machine_tokens = group["global_status"].map(_token)
        state_not_running = bool(
            ~module_tokens.isin(config.running_states).all()
            or ~machine_tokens.isin(config.running_states).all()
        )
        transition = bool(group["_transition"].any() or group["_settling"].any())

        if not finite:
            reasons.append("NON_FINITE_SENSOR")
        if sentinel:
            reasons.append("SENTINEL_VALUE")
        if pressure_invalid:
            reasons.append("PRESSURE_OUT_OF_RANGE")
        if temperature_invalid:
            reasons.append("TEMPERATURE_OUT_OF_RANGE")
        if valve_invalid:
            reasons.append("NEGATIVE_VALVE")
        if duplicate:
            reasons.append("DUPLICATE_TIMESTAMP")
        if out_of_order:
            reasons.append("OUT_OF_ORDER_TIMESTAMP")

        if reasons:
            status = WindowStatus.DATA_QUALITY_REVIEW.value
        elif point_count < config.minimum_window_points:
            reasons.append("TOO_FEW_POINTS")
            status = WindowStatus.INCOMPLETE_WINDOW.value
        elif maximum_gap > config.max_gap_seconds:
            reasons.append("TIME_GAP_EXCEEDED")
            status = WindowStatus.INCOMPLETE_WINDOW.value
        elif coverage < config.minimum_coverage:
            reasons.append("COVERAGE_BELOW_MINIMUM")
            status = WindowStatus.INCOMPLETE_WINDOW.value
        elif state_not_running or transition:
            if state_not_running:
                reasons.append("OFF_OR_NON_RUNNING_STATE")
            if transition:
                reasons.append("TRANSITION_OR_SETTLING")
            status = WindowStatus.OFF_OR_TRANSITION.value
        else:
            status = WindowStatus.ELIGIBLE.value

        medians = group[list(SENSOR_COLUMNS)].median(numeric_only=True)
        rows.append(
            {
                "event_time": window_start,
                "window_end": window_start + pd.Timedelta(seconds=config.window_seconds),
                "processing_time": processing,
                "machine_id": str(machine_id),
                "module_id": int(module_id),
                **{column: float(medians[column]) for column in SENSOR_COLUMNS},
                "sv": _series_mode(group["sv"]),
                "module_status": _series_mode(group["module_status"]),
                "global_status": _series_mode(group["global_status"]),
                "busy": float(group["busy"].median()),
                "point_count": point_count,
                "median_sampling_interval_seconds": median_interval,
                "maximum_gap_seconds": maximum_gap,
                "coverage": coverage,
                "window_status": status,
                "quality_reasons": reasons,
            }
        )
    return add_com2_features(pd.DataFrame(rows)).sort_values(
        ["event_time", "machine_id", "module_id"]
    ).reset_index(drop=True)
