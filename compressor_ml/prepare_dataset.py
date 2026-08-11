from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .features import engineer_features
from .preprocessing import read_handler_log, validate_and_filter
from .windowing import chronological_split, make_windows


DAILY_DATE_PATTERN = re.compile(r"20\d{2}_\d{2}_\d{2}")


@dataclass(frozen=True)
class MachineSource:
    machine_id: str
    directory: Path


def safe_group_name(machine_id: str, module_id: int) -> str:
    safe_machine = re.sub(r"[^A-Za-z0-9_-]+", "_", machine_id)
    return f"{safe_machine}__M{module_id:02d}"


def discover_daily_files(directory: Path, max_files: int | None = None) -> list[Path]:
    candidates: list[Path] = []
    if not directory.exists():
        return candidates
    if directory.is_file():
        lowered = directory.name.lower()
        if (
            directory.suffix.lower() in {".csv", ".txt"}
            and "static" not in lowered
            and ".tmp." not in lowered
            and DAILY_DATE_PATTERN.search(directory.name)
        ):
            return [directory]
        return candidates
    for pattern in ("*.csv", "*.txt"):
        for path in directory.rglob(pattern):
            lowered = path.name.lower()
            if (
                "static" in lowered
                or ".tmp." in lowered
                or lowered.startswith("~$")
                or not DAILY_DATE_PATTERN.search(path.name)
            ):
                continue
            candidates.append(path)
    files = sorted(set(candidates), key=lambda path: path.as_posix())
    return files[-max_files:] if max_files else files


def even_take(
    windows: np.ndarray,
    metadata: pd.DataFrame,
    limit: int | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if limit is None or len(windows) <= limit:
        return windows, metadata.reset_index(drop=True)
    indices = np.linspace(0, len(windows) - 1, limit, dtype=int)
    return windows[indices], metadata.iloc[indices].reset_index(drop=True)


def _save_group(
    output_dir: Path,
    machine_id: str,
    module_id: int,
    windows: np.ndarray,
    metadata: pd.DataFrame,
) -> dict:
    (train_x, train_meta), (validation_x, validation_meta), (test_x, test_meta) = chronological_split(windows, metadata)
    group_name = safe_group_name(machine_id, module_id)
    group_path = output_dir / "groups" / f"{group_name}.npz"
    np.savez_compressed(
        group_path,
        train=train_x.astype(np.float32),
        validation=validation_x.astype(np.float32),
        test=test_x.astype(np.float32),
    )

    metadata_parts = []
    for split_name, split_meta in (
        ("train", train_meta),
        ("validation", validation_meta),
        ("test", test_meta),
    ):
        part = split_meta.copy()
        part.insert(0, "split_index", np.arange(len(part), dtype=int))
        part.insert(0, "split", split_name)
        metadata_parts.append(part)
    metadata_path = output_dir / "metadata" / f"{group_name}.csv"
    pd.concat(metadata_parts, ignore_index=True).to_csv(metadata_path, index=False)
    return {
        "group_name": group_name,
        "machine_id": machine_id,
        "module_id": module_id,
        "windows_file": str(group_path.relative_to(output_dir)).replace("\\", "/"),
        "metadata_file": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "windows": {
            "train": int(len(train_x)),
            "validation": int(len(validation_x)),
            "test": int(len(test_x)),
            "total": int(len(windows)),
        },
    }


def prepare_dataset(
    machine_sources: Iterable[MachineSource],
    module_ids: Iterable[int],
    output_dir: str | Path,
    config: PipelineConfig,
    dataset_version: str,
    max_files_per_machine: int | None = None,
    max_windows_per_group: int | None = None,
) -> dict:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    (destination / "groups").mkdir(parents=True, exist_ok=True)
    (destination / "metadata").mkdir(parents=True, exist_ok=True)

    requested_modules = tuple(sorted(set(int(module) for module in module_ids)))
    forbidden = set(requested_modules).intersection(config.excluded_modules)
    if forbidden:
        raise ValueError(f"Excluded modules requested: {sorted(forbidden)}")
    if not requested_modules:
        raise ValueError("At least one module is required")

    manifest_groups: list[dict] = []
    quality_rows: list[dict] = []
    source_rows: list[dict] = []

    for machine_source in machine_sources:
        paths = discover_daily_files(machine_source.directory, max_files_per_machine)
        if not paths:
            raise FileNotFoundError(f"No daily CSV/TXT logs found for {machine_source.machine_id}: {machine_source.directory}")
        per_file_cap = None
        if max_windows_per_group:
            per_file_cap = max(10, math.ceil(max_windows_per_group / len(paths)))

        group_windows: dict[int, list[np.ndarray]] = {module_id: [] for module_id in requested_modules}
        group_metadata: dict[int, list[pd.DataFrame]] = {module_id: [] for module_id in requested_modules}
        print(f"{machine_source.machine_id}: reading {len(paths)} file(s)", flush=True)

        for path in paths:
            source_rows.append(
                {
                    "machine_id": machine_source.machine_id,
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                }
            )
            raw = read_handler_log(path, machine_source.machine_id)
            for module_id in requested_modules:
                module_raw = raw.loc[raw["module_id"].eq(module_id)].copy()
                valid, rejected = validate_and_filter(module_raw, config)
                featured = engineer_features(valid, config)
                windows, metadata = make_windows(featured, config)
                selected_windows, selected_metadata = even_take(windows, metadata, per_file_cap)
                if len(selected_windows):
                    selected_metadata = selected_metadata.copy()
                    selected_metadata["source_file"] = str(path.resolve())
                    group_windows[module_id].append(selected_windows)
                    group_metadata[module_id].append(selected_metadata)
                quality_rows.append(
                    {
                        "machine_id": machine_source.machine_id,
                        "module_id": module_id,
                        "source_file": str(path.resolve()),
                        "raw_rows": int(len(module_raw)),
                        "accepted_rows": int(len(valid)),
                        "rejected_rows": int(len(rejected)),
                        "complete_windows": int(len(windows)),
                        "selected_windows": int(len(selected_windows)),
                        "imputed_short_gap_rows": int(valid.get("is_imputed_short_gap", pd.Series(dtype=bool)).sum()),
                    }
                )
            del raw

        for module_id in requested_modules:
            if not group_windows[module_id]:
                print(f"Skipping {machine_source.machine_id} module {module_id}: no complete stable windows", flush=True)
                continue
            windows = np.concatenate(group_windows[module_id])
            metadata = pd.concat(group_metadata[module_id], ignore_index=True)
            order = pd.to_datetime(metadata["timestamp"]).sort_values().index.to_numpy()
            windows = windows[order]
            metadata = metadata.iloc[order].reset_index(drop=True)
            windows, metadata = even_take(windows, metadata, max_windows_per_group)
            if len(windows) < 10:
                print(f"Skipping {machine_source.machine_id} module {module_id}: only {len(windows)} windows", flush=True)
                continue
            manifest_groups.append(
                _save_group(destination, machine_source.machine_id, module_id, windows, metadata)
            )

    if len(manifest_groups) < 2:
        raise ValueError("Prepared dataset needs at least two valid machine-module groups")

    config.save(destination / "config.json")
    pd.DataFrame(quality_rows).to_csv(destination / "data_quality_summary.csv", index=False)
    manifest = {
        "dataset_version": dataset_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "state_cleaned_unscaled_windows_v1",
        "window_shape": [config.window_rows, len(config.feature_columns)],
        "feature_columns": list(config.feature_columns),
        "split_method": "chronological_70_15_15_within_machine_module",
        "scaling": "not_applied_fit_each_group_on_train_partition_only",
        "groups": manifest_groups,
        "sources": source_rows,
        "parameters": {
            "module_ids": list(requested_modules),
            "max_files_per_machine": max_files_per_machine,
            "max_windows_per_group": max_windows_per_group,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "# Prepared HT-9046MX dataset\n\n"
        "This bundle contains state-cleaned, unscaled 60-second windows. "
        "Train/validation/test are chronological inside each machine-module group.\n\n"
        "Fit a separate scaler on each group's `train` array only. Pool only the "
        "scaled train arrays for the shared model. Calibrate each threshold from "
        "that group's validation array.\n",
        encoding="utf-8",
    )
    return manifest


def load_prepared_dataset(dataset_dir: str | Path) -> tuple[PipelineConfig, dict, dict[tuple[str, int], dict[str, np.ndarray]]]:
    root = Path(dataset_dir)
    config = PipelineConfig.load(root / "config.json")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["feature_columns"] != list(config.feature_columns):
        raise ValueError("Prepared dataset feature list does not match config.json")
    datasets: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for group in manifest["groups"]:
        with np.load(root / group["windows_file"], allow_pickle=False) as arrays:
            datasets[(group["machine_id"], int(group["module_id"]))] = {
                "train": arrays["train"].astype(np.float32),
                "validation": arrays["validation"].astype(np.float32),
                "test": arrays["test"].astype(np.float32),
            }
    return config, manifest, datasets


def parse_machine_source(value: str) -> MachineSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use MACHINE_ID=PATH")
    machine_id, raw_path = value.split("=", 1)
    if not machine_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Use MACHINE_ID=PATH")
    return MachineSource(machine_id.strip(), Path(raw_path.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a state-cleaned HT-9046MX dataset bundle before Colab training")
    parser.add_argument("--machine-dir", action="append", type=parse_machine_source, required=True, help="Repeat MACHINE_ID=PATH")
    parser.add_argument("--modules", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-version", default="ht9046mx_shared_v1")
    parser.add_argument("--config")
    parser.add_argument("--max-files-per-machine", type=int)
    parser.add_argument("--max-windows-per-group", type=int)
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.machine_dir,
        args.modules,
        args.output_dir,
        PipelineConfig.load(args.config),
        args.dataset_version,
        args.max_files_per_machine,
        args.max_windows_per_group,
    )
    print(json.dumps({
        "dataset_version": manifest["dataset_version"],
        "groups": len(manifest["groups"]),
        "output_dir": str(Path(args.output_dir).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
