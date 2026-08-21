# HT-9046MX Controlled Hybrid Condition Monitoring

The production-candidate path is now **Controlled Hybrid v1**: an explainable
COM2 detector is the primary decision path, while the immutable 30-epoch Shared
LSTM (`shared_lstm_full_v1`) supplies independent shadow evidence and helps
select clean bootstrap history. A new machine can learn a candidate profile
automatically, but activation always stops at `APPROVAL_REQUIRED`; only a named
human approver can create an `ACTIVE_FROZEN` profile.

Start with [CONTROLLED_HYBRID_SYSTEM.md](CONTROLLED_HYBRID_SYSTEM.md). The older
`adaptive_runner` flow remains in the repository for reproducibility and
comparison, but it is no longer the recommended production scheduler because
its bounded auto-approval policy is less conservative than the controlled
profile lifecycle.

This code-only repository also contains a Google Colab workflow for an **unlabelled, state-aware shared LSTM Autoencoder**. One model learns normalized temporal patterns from all handlers, while every `machine_id + module_id` keeps its own scaler and anomaly threshold.

The repository also contains a guarded **adaptive scoring and calibration system**. It keeps the shared model, train-only scalers, and golden baselines immutable while allowing a bounded operational profile to advance through frozen replay, synthetic-regression checks, shadow observations, automatic approval, audit history, and rollback.

Raw machine logs, trained models, virtual environments, and generated analysis outputs are intentionally excluded from GitHub.

## Colab quick start

1. Build the prepared dataset locally from the raw logs with the command below.
2. Create `ht9046mx_colab_full_package.zip` containing the source and prepared bundle.
3. Put `HT9046MX_Shared_Model_Colab.ipynb` in `MyDrive/Data Analysis`, then upload the package into Colab session storage at `/content`.
4. Open the notebook in Google Colab.
5. Select **Runtime → Change runtime type → T4 GPU** and run all cells.

The notebook defaults to `USE_DRIVE = False` because DriveFS can be unavailable even for a signed-in Colab session. In this mode the trained adaptive runtime ZIP is downloaded automatically at the final cell. Set `USE_DRIVE = True` only when Drive mount is working and direct Drive persistence is preferred.

Expected Google Drive layout:

```text
MyDrive/
└── Data Analysis/
    ├── HT9046MX_Shared_Model_Colab.ipynb
    └── ht9046mx_colab_full_package.zip  # optional when USE_DRIVE=True
```

Raw logs remain only on the local machine. They are not uploaded to GitHub or required by the training notebook.

Cells 11–12 create `adaptive_seed` and `ht9046mx_adaptive_runtime_<mode>.zip` for continuous scoring on Windows. See [ADAPTIVE_SYSTEM.md](ADAPTIVE_SYSTEM.md) for the complete setup and safety contract.

## Build the prepared smoke dataset locally

Run this from the local `Data Analysis` directory after installing the requirements:

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.prepare_dataset `
  --machine-dir "MX12=Clean Data MX12\2026_07\2026_07_13_cleaned.csv" `
  --machine-dir "MX25=Clean Data MX25\2026_07\2026_07_12_cleaned.csv" `
  --machine-dir "MX007=MX_007\2026_07_01_clean.csv" `
  --machine-dir "MX017=MX017\2026_08_09.txt" `
  --machine-dir "MX057=MX057\2026_08\2026_08_10.txt" `
  --machine-dir "MX070=MX070\2026_08\2026_08_09.txt" `
  --modules 1 2 3 4 5 6 8 `
  --output-dir "prepared_dataset\shared_smoke_v2" `
  --dataset-version "shared_smoke_v2" `
  --max-windows-per-group 500
```

The bundle stores unscaled `(60, 24)` windows with chronological 70/15/15 splits, metadata, source lineage, and a data-quality summary. Colab fits a separate train-only scaler for each machine-module, pools only normalized training windows, and calibrates thresholds from each group's validation partition.

Create the upload package after the dataset finishes:

```powershell
.\.venv\Scripts\python.exe scripts\build_colab_package.py
```

For the bounded full dataset, replace each explicit representative file with its machine directory, use a new immutable output directory such as `prepared_dataset\shared_full_v2`, set `--max-files-per-machine 10` (MX_007 contains three RTF reports that are audited and skipped), and increase `--max-windows-per-group` to `5000`. Then build a new ZIP and set `RUN_MODE = "full"` in the notebook.

Build the full-only Colab package without duplicating the smoke dataset:

```powershell
.\.venv\Scripts\python.exe scripts\build_colab_package.py `
  --dataset-dir prepared_dataset\shared_full_v2 `
  --output artifacts\ht9046mx_colab_full_package.zip
```

## Data-backed safeguards

- Module 7 is excluded because the available logs show it is almost never active.
- `Status_n` is the module-state field. `Busy_n=1`, `ChangeValve`, `AdjustValve`, `MValveHome`, negative Valve values, and `TempHi/TempLo=-200` are excluded from normal training.
- Windows cannot cross a state transition or time gap.
- Logger gaps up to five seconds are interpolated only inside an unchanged active-state run.
- Train/validation/test splits are chronological inside each machine-module group.
- The shared model uses balanced group weights; scalers and validation-p99 thresholds remain group-specific.
- Output is anomaly detection with temporary pseudo-labels. It is not root-cause diagnosis or RUL.

## Local single-group smoke test

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m compressor_ml.train `
  --input MX017\2026_08_09.txt `
  --machine-id MX017 --module-id 1 `
  --output-dir artifacts\MX017_M1 `
  --max-windows 120 --config configs\smoke_config.json
```

Do not use a smoke artifact for maintenance decisions.

## Adaptive scoring quick start

After downloading and extracting the Colab adaptive runtime ZIP into `artifacts/shared_lstm_colab_smoke`:

```powershell
.\scripts\initialize_adaptive_runtime.ps1
.\scripts\run_adaptive_cycle.ps1
.\.venv\Scripts\python.exe -m compressor_ml.adaptive_runner status --runtime-dir adaptive_runtime
```

Only after reviewing the smoke output, install the scheduled cycle:

```powershell
.\scripts\install_adaptive_task.ps1 -IntervalMinutes 15
```

Automatic approval applies only to bounded per-group calibration profiles. It does not approve shared-model retraining, fault diagnosis, or maintenance actions without maintenance-linked labels. The synthetic regression canary is scaled beyond each machine/module's immutable golden feature threshold, so a group with a wide historical operating range is tested at comparable severity instead of using a fixed shift that may fall inside its learned baseline.

## Offline multi-machine adaptive test (no MySQL)

This test copies an adaptive seed to a temporary runtime, feeds eligible reference-like observations to one machine/module, advances the guarded shadow approvals, and proves that a second machine/module profile and every golden profile remain unchanged:

```powershell
.\.venv\Scripts\python.exe scripts\offline_adaptive_multimachine_test.py `
  --seed-dir artifacts\shared_lstm_colab_full\adaptive_seed `
  --target-group MX017__M02 `
  --control-group MX070__M02
```

The verified full-run test should advance `MX017__M02` through `SHADOW`, `SHADOW`, and `AUTO_APPROVED`, while `MX070__M02`, every golden profile, the train-only scalers, and the shared model remain unchanged. It does not connect to MySQL, retrain the shared model, or claim measured fault accuracy.
