# HT-9046MX Shared Predictive-Maintenance Model

This code-only repository contains a Google Colab workflow for an **unlabelled, state-aware shared LSTM Autoencoder**. One model learns normalized temporal patterns from all handlers, while every `machine_id + module_id` keeps its own scaler and anomaly threshold.

Raw machine logs, trained models, virtual environments, and generated analysis outputs are intentionally excluded from GitHub.

## Colab quick start

1. Build the prepared dataset locally from the raw logs with the command below.
2. Create `ht9046mx_colab_package.zip` containing the source and prepared bundle.
3. Put the ZIP and `HT9046MX_Shared_Model_Colab.ipynb` in `MyDrive/Data Analysis`.
4. Open the notebook in Google Colab.
5. Select **Runtime → Change runtime type → T4 GPU** and run all cells.

Expected Google Drive layout:

```text
MyDrive/
└── Data Analysis/
    ├── HT9046MX_Shared_Model_Colab.ipynb
    └── ht9046mx_colab_package.zip
```

Raw logs remain only on the local machine. They are not uploaded to GitHub or required by the training notebook.

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
tar.exe -a -c -f artifacts\ht9046mx_colab_package.zip `
  compressor_ml configs requirements.txt README.md prepared_dataset\shared_smoke_v2
```

For the bounded full dataset, replace each explicit representative file with its machine directory, use a new output directory such as `prepared_dataset\shared_full_v1`, add `--max-files-per-machine 7`, and increase `--max-windows-per-group` to `5000`. Then build a new ZIP and set `RUN_MODE = "full"` in the notebook.

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
