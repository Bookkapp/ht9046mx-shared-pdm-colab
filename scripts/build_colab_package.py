from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT = ARTIFACT_DIR / "ht9046mx_colab_package.zip"
DEFAULT_DATASET = PROJECT_ROOT / "prepared_dataset" / "shared_smoke_v2"


def included_files(sources: tuple[Path, ...]):
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            yield source
        else:
            for path in sorted(source.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    yield path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Colab upload package")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".building.zip")
    sources = (
        PROJECT_ROOT / "compressor_ml",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "ADAPTIVE_SYSTEM.md",
        dataset_dir,
    )
    dataset_entry = f"prepared_dataset/{dataset_dir.name}/manifest.json"
    required_entries = {
        "compressor_ml/adaptive.py",
        "compressor_ml/adaptive_runner.py",
        "configs/adaptive_calibration.json",
        "scripts/run_adaptive_cycle.ps1",
        dataset_entry,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in included_files(sources):
            if path.is_relative_to(dataset_dir):
                relative = Path("prepared_dataset") / dataset_dir.name / path.relative_to(dataset_dir)
            else:
                relative = path.relative_to(PROJECT_ROOT)
            archive.write(path, relative.as_posix())

    with zipfile.ZipFile(temporary) as archive:
        bad_entry = archive.testzip()
        entries = set(archive.namelist())
    if bad_entry is not None:
        raise RuntimeError(f"Corrupt ZIP entry: {bad_entry}")
    missing = sorted(required_entries.difference(entries))
    if missing:
        raise RuntimeError(f"Package is missing required entries: {missing}")
    temporary.replace(output)
    print(
        {
            "output": output.name,
            "dataset": dataset_dir.name,
            "size_mb": round(output.stat().st_size / 1024**2, 1),
            "entries": len(entries),
        }
    )


if __name__ == "__main__":
    main()
