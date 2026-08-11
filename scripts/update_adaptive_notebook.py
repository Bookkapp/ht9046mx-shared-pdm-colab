from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


NOTEBOOK = Path(__file__).resolve().parents[1] / "HT9046MX_Shared_Model_Colab.ipynb"
TAG = "adaptive-system"


def tagged(cell):
    return TAG in cell.get("metadata", {}).get("tags", [])


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not tagged(cell)]

    markdown = new_markdown_cell(
        """## Adaptive System Bootstrap

ส่วนนี้สร้าง Golden Calibration และ Frozen Holdout แยกทุก `machine_id + module_id` หลัง Shared Model เทรนเสร็จ จากนั้นบันทึก Adaptive Seed และ ZIP สำหรับนำไปใช้กับ Automatic Scoring บน Windows

หลักความปลอดภัย:

- Shared Model และ train-only scaler จะไม่ถูกปรับอัตโนมัติ
- Golden Profile จะไม่ถูกเขียนทับ
- ระบบปรับเฉพาะ operational calibration หลังผ่าน validation และ shadow observations
- การ validate นี้เป็น unsupervised calibration validation ไม่ใช่การยืนยัน fault accuracy"""
    )
    markdown.metadata["tags"] = [TAG]

    bootstrap = new_code_cell(
        """# @title 11. Build immutable golden profiles and adaptive seed
from compressor_ml.adaptive import AdaptiveConfig
from compressor_ml.adaptive_runner import bootstrap_from_prepared

ADAPTIVE_SEED_DIR = ARTIFACT_DIR / 'adaptive_seed'
adaptive_config = AdaptiveConfig.load(PROJECT_DIR / 'configs' / 'adaptive_calibration.json')

# A rerun may replace only this generated seed inside the current artifact.
if ADAPTIVE_SEED_DIR.exists():
    assert ADAPTIVE_SEED_DIR.parent.resolve() == ARTIFACT_DIR.resolve()
    shutil.rmtree(ADAPTIVE_SEED_DIR)

adaptive_seed_summary = bootstrap_from_prepared(
    PREPARED_DATASET_DIR,
    ARTIFACT_DIR,
    ADAPTIVE_SEED_DIR,
    adaptive_config=adaptive_config,
    model=shared_model,
    batch_size=BATCH_SIZE,
)
display(pd.DataFrame([adaptive_seed_summary]))"""
    )
    bootstrap.metadata["tags"] = [TAG]

    package = new_code_cell(
        """# @title 12. Validate seed coverage and create the Windows runtime ZIP
seed_manifest = json.loads((ADAPTIVE_SEED_DIR / 'seed_manifest.json').read_text(encoding='utf-8'))
seed_groups = {row['group_name'] for row in seed_manifest['groups']}
artifact_groups = set(group_thresholds)

assert seed_groups == artifact_groups, {
    'missing_seed_groups': sorted(artifact_groups - seed_groups),
    'unexpected_seed_groups': sorted(seed_groups - artifact_groups),
}
assert seed_manifest['model_version'] == f'shared_lstm_{RUN_MODE}_v1'
assert seed_manifest['safety_contract']['shared_model_weights'] == 'immutable'
assert seed_manifest['safety_contract']['golden_profiles'] == 'immutable'

archive_base = DRIVE_PROJECT_DIR / 'artifacts' / f'ht9046mx_adaptive_runtime_{RUN_MODE}'
archive_path = Path(shutil.make_archive(
    str(archive_base),
    'zip',
    root_dir=ARTIFACT_DIR.parent,
    base_dir=ARTIFACT_DIR.name,
))

print('Adaptive groups:', len(seed_groups))
print('Runtime ZIP:', archive_path)
print('ZIP size (MB):', round(archive_path.stat().st_size / 1024**2, 1))
print('Safety contract:', seed_manifest['safety_contract'])"""
    )
    package.metadata["tags"] = [TAG]

    next_steps = new_markdown_cell(
        """## Run continuously on Windows

หลัง cell ด้านบนผ่านครบ:

1. ดาวน์โหลด `artifacts/ht9046mx_adaptive_runtime_smoke.zip` จาก Google Drive
2. แตก ZIP ให้ได้โฟลเดอร์ `artifacts/shared_lstm_colab_smoke` ภายในโปรเจกต์
3. รัน `scripts\\initialize_adaptive_runtime.ps1`
4. ทดสอบหนึ่งรอบด้วย `scripts\\run_adaptive_cycle.ps1`
5. เมื่อตรวจผล smoke แล้วจึงรัน `scripts\\install_adaptive_task.ps1` เพื่อตั้งเวลา Automatic Scoring

Smoke model ใช้ยืนยันว่า pipeline ทำงานครบเท่านั้น ก่อนใช้ตัดสินใจด้าน maintenance ให้สร้าง `shared_full_v1`, ตั้ง `RUN_MODE='full'` และรัน Notebook ใหม่"""
    )
    next_steps.metadata["tags"] = [TAG]

    generated_cells = [markdown, bootstrap, package, next_steps]
    if int(notebook.get("nbformat_minor", 0)) < 5:
        for cell in generated_cells:
            cell.pop("id", None)
    notebook.cells.extend(generated_cells)
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print(
        f"Updated {NOTEBOOK.name} with "
        f"{sum(tagged(cell) for cell in notebook.cells)} adaptive cells"
    )


if __name__ == "__main__":
    main()
