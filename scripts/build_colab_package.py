from __future__ import annotations

from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT = ARTIFACT_DIR / "ht9046mx_colab_package.zip"
TEMPORARY = ARTIFACT_DIR / "ht9046mx_colab_package.building.zip"

INCLUDE = (
    PROJECT_ROOT / "compressor_ml",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "ADAPTIVE_SYSTEM.md",
    PROJECT_ROOT / "prepared_dataset" / "shared_smoke_v2",
)

REQUIRED_ENTRIES = {
    "compressor_ml/adaptive.py",
    "compressor_ml/adaptive_runner.py",
    "configs/adaptive_calibration.json",
    "scripts/run_adaptive_cycle.ps1",
    "prepared_dataset/shared_smoke_v2/manifest.json",
}


def included_files():
    for source in INCLUDE:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            yield source
        else:
            for path in sorted(source.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    yield path


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if TEMPORARY.exists():
        TEMPORARY.unlink()
    with zipfile.ZipFile(
        TEMPORARY, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in included_files():
            archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())

    with zipfile.ZipFile(TEMPORARY) as archive:
        bad_entry = archive.testzip()
        entries = set(archive.namelist())
    if bad_entry is not None:
        raise RuntimeError(f"Corrupt ZIP entry: {bad_entry}")
    missing = sorted(REQUIRED_ENTRIES.difference(entries))
    if missing:
        raise RuntimeError(f"Package is missing required entries: {missing}")
    TEMPORARY.replace(OUTPUT)
    print(
        {
            "output": OUTPUT.name,
            "size_mb": round(OUTPUT.stat().st_size / 1024**2, 1),
            "entries": len(entries),
        }
    )


if __name__ == "__main__":
    main()
