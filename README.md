# HT-9046MX Shared Predictive-Maintenance Model

This code-only repository contains a Google Colab workflow for an **unlabelled, state-aware shared LSTM Autoencoder**. One model learns normalized temporal patterns from all handlers, while every `machine_id + module_id` keeps its own scaler and anomaly threshold.

Raw machine logs, trained models, virtual environments, and generated analysis outputs are intentionally excluded from GitHub.

## Colab quick start

1. Clone or download this repository into `MyDrive/Data Analysis`.
2. Copy the six raw-data folders into the same directory using the layout below.
3. Open `HT9046MX_Shared_Model_Colab.ipynb` in Google Colab.
4. Select **Runtime → Change runtime type → T4 GPU**.
5. Run all cells with `RUN_MODE = "smoke"` first.

Expected Google Drive layout:

```text
MyDrive/
└── Data Analysis/
    ├── HT9046MX_Shared_Model_Colab.ipynb
    ├── compressor_ml/
    ├── configs/
    ├── requirements.txt
    ├── Clean Data MX12/   # raw data; never committed
    ├── Clean Data MX25/   # raw data; never committed
    ├── MX_007/            # raw data; never committed
    ├── MX017/             # raw data; never committed
    ├── MX057/             # raw data; never committed
    └── MX070/             # raw data; never committed
```

Smoke mode uses the latest daily file per machine, Modules 1–6 and 8, two training epochs, and bounded windows. Full mode defaults to the latest seven files per machine and 30 epochs so Colab memory remains bounded.

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
