from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .config import PipelineConfig


RAW_FIELDS = ("Hp_1st", "Lp_1st", "Hp_2nd", "Lp_2nd", "Valve", "TempHi", "TempLo")
STATE_FIELDS = ("Status", "Busy", "SV")


def _date_from_path(path: Path) -> str:
    match = re.search(r"(20\d{2})_(\d{2})_(\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot infer date from filename: {path.name}")
    return "-".join(match.groups())


def _header_row(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            first_field = line.split(",", 1)[0].strip()
            if first_field in {"Time", "Date"}:
                return index
    raise ValueError(f"No tabular header found in {path}")


def read_handler_log(path: str | Path, machine_id: str) -> pd.DataFrame:
    """Read one handler-day export and return a canonical long-format table."""
    source = Path(path)
    frame = pd.read_csv(source, skiprows=_header_row(source), skipinitialspace=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Time" not in frame:
        raise ValueError(f"Time column missing from {source}")
    day = frame["Date"].astype(str) if "Date" in frame else _date_from_path(source)
    stamp = pd.to_datetime(day.astype(str) + " " + frame["Time"].astype(str).str.strip(), errors="coerce") if isinstance(day, pd.Series) else pd.to_datetime(day + " " + frame["Time"].astype(str).str.strip(), errors="coerce")
    rows: list[pd.DataFrame] = []
    for module_id in range(1, 9):
        data: dict[str, pd.Series | int | str] = {
            "timestamp": stamp,
            "machine_id": machine_id,
            "module_id": module_id,
            "global_status": frame.get("Status", "<blank>").astype(str).str.strip(),
        }
        for raw_name, output_name in (("Hp_1st", "hp1"), ("Lp_1st", "lp1"), ("Hp_2nd", "hp2"), ("Lp_2nd", "lp2"), ("Valve", "valve"), ("TempHi", "temphi"), ("TempLo", "templo")):
            data[output_name] = pd.to_numeric(frame.get(f"{raw_name}_{module_id}"), errors="coerce")
        data["module_status"] = frame.get(f"Status_{module_id}", "<blank>").astype(str).str.strip()
        data["busy"] = pd.to_numeric(frame.get(f"Busy_{module_id}"), errors="coerce")
        data["sv"] = frame.get(f"SV_{module_id}", "<blank>").astype(str).str.strip()
        rows.append(pd.DataFrame(data))
    return pd.concat(rows, ignore_index=True).dropna(subset=["timestamp"])


def validate_and_filter(frame: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag context/sentinel rows and retain only stable active observations."""
    data = frame.copy().sort_values(["machine_id", "module_id", "timestamp"]).reset_index(drop=True)
    data["is_excluded_module"] = data["module_id"].isin(config.excluded_modules)
    data["is_transition"] = data["module_status"].isin(config.transition_states) | data["busy"].eq(1)
    data["is_homing"] = data["module_status"].eq("MValveHome") | data["valve"].lt(0)
    data["is_sentinel"] = data["templo"].eq(-200) | data["temphi"].eq(-200)
    data["is_invalid_pressure"] = (~data[["hp1", "lp1", "hp2", "lp2"]].ge(0).all(axis=1)) | (~data[["hp1", "lp1", "hp2", "lp2"]].le(1000).all(axis=1))
    data["is_invalid_temperature"] = (~data[["temphi", "templo"]].gt(-150).all(axis=1)) | (~data[["temphi", "templo"]].lt(250).all(axis=1))
    data["is_active"] = (
        data["module_status"].eq(config.normal_status)
        & data["busy"].eq(0)
        & ~data[["is_excluded_module", "is_transition", "is_homing", "is_sentinel", "is_invalid_pressure", "is_invalid_temperature"]].any(axis=1)
    )
    group = data.groupby(["machine_id", "module_id"], sort=False)["timestamp"]
    data["seconds_since_previous"] = group.diff().dt.total_seconds()
    data["is_time_gap"] = data["seconds_since_previous"].notna() & data["seconds_since_previous"].ne(config.sampling_interval_sec)
    data["is_duplicate_timestamp"] = data.duplicated(["machine_id", "module_id", "timestamp"], keep=False)
    data["active_run_id"] = data.groupby(["machine_id", "module_id"], sort=False)["is_active"].transform(lambda x: x.ne(x.shift()).cumsum()).astype(int)
    stable = data.loc[data["is_active"]].copy()
    quality = data.loc[~data["is_active"]].copy()
    return _resample_stable_runs(stable, config), quality


def _resample_stable_runs(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Resample short logger gaps only within an uninterrupted active-state run."""
    if frame.empty:
        empty = frame.copy()
        empty["is_imputed_short_gap"] = pd.Series(dtype=bool)
        empty["segment_break"] = pd.Series(dtype=bool)
        empty["segment_id"] = pd.Series(dtype=int)
        return empty
    numeric = ["hp1", "lp1", "hp2", "lp2", "valve", "temphi", "templo"]
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby(["machine_id", "module_id", "active_run_id"], sort=False):
        group = group.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        indexed = group.set_index("timestamp")
        target = pd.date_range(indexed.index.min(), indexed.index.max(), freq=f"{config.sampling_interval_sec}s")
        expanded = indexed.reindex(target)
        observed = expanded[numeric].notna().all(axis=1)
        expanded[numeric] = expanded[numeric].interpolate(limit=config.max_interpolation_gap_sec, limit_area="inside")
        # These context fields are constant inside an active run; filling them
        # never crosses a Busy/ChangeValve/MValveHome boundary.
        for column in ("machine_id", "module_id", "global_status", "module_status", "busy", "sv", "active_run_id"):
            expanded[column] = expanded[column].ffill().bfill()
        expanded["timestamp"] = expanded.index
        expanded["is_imputed_short_gap"] = ~observed
        expanded = expanded.dropna(subset=numeric)
        pieces.append(expanded.reset_index(drop=True))
    valid = pd.concat(pieces, ignore_index=True).sort_values(["machine_id", "module_id", "timestamp"]).reset_index(drop=True)
    valid["segment_break"] = valid.groupby(["machine_id", "module_id"], sort=False)["timestamp"].diff().dt.total_seconds().ne(config.sampling_interval_sec)
    valid["segment_id"] = valid.groupby(["machine_id", "module_id"], sort=False)["segment_break"].cumsum().astype(int)
    return valid


def load_and_prepare(paths: Iterable[str | Path], machine_id: str, config: PipelineConfig, module_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [read_handler_log(path, machine_id) for path in paths]
    if not frames:
        raise ValueError("No source logs supplied")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw["module_id"].eq(module_id)]
    return validate_and_filter(raw, config)
