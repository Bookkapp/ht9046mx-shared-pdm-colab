# HT9046MX Project Structure (MySQL-only)

```text
Data Analysis\
├── artifacts\shared_lstm_colab_full\       Immutable 30-epoch Shared LSTM
├── compressor_ml\
│   ├── mysql_source.py                       Read-only MySQL → canonical telemetry adapter
│   ├── machine.py                            Machine-code normalization
│   ├── controlled_monitoring\               COM2/LSTM lifecycle and scheduler runner
│   └── preprocessing.py, train.py, ...       Offline/Colab training utilities only
├── compressor_fastapi_react_dashboard\
│   ├── backend\app\                          FastAPI, MySQL-backed model API
│   └── frontend\src\                         React model-monitor UI
├── configs\                                  MySQL runtime and Controlled Hybrid policy templates
├── scripts\                                  State initialization and model scheduler setup
├── tests\                                    Model pipeline and MySQL normalization tests
├── HT9046MX_Shared_Model_Colab.ipynb         Deliberate retraining only
├── README.md                                  Deployment instructions
├── SERVER_FOLDER_LAYOUT.md                   Permanent 10.195.17.69 layout
└── CONTROLLED_HYBRID_SYSTEM.md               Model methodology
```

## Production source of truth

| Data | Owner | Use |
|---|---|---|
| `10.195.17.73 / ht9046mx_iot.ht9046mx_readings` | Database Server, existing MySQL ingestion | Only live telemetry source; this repository is not installed on this machine and application issues read-only queries over TCP. |
| `artifacts\shared_lstm_colab_full` | Git deployment | Immutable Shared LSTM model/scalers/thresholds. |
| `C:\HT9046MX\state\controlled_runtime` | Web/model Server `10.195.17.69` only | Versioned profiles, decisions, audit trail and MySQL cursor; back up separately from the database. |
| `backend\.env` | Web/model Server operator | MySQL credentials and deployment paths; never commit or copy it to the Database Server. |

## Removed from production

There is no SMB sync worker, UNC path, handler registry, source-copy folder,
file signature registry, handler-IP page, or direct runtime call to
`read_handler_log`. The remaining log parsing code is retained solely for
offline dataset preparation and Colab retraining.

## Main commands

```powershell
# Validate schema/access without changing the database
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config "C:\HT9046MX\state\config\controlled_condition_monitoring.json" source-check

# Build a candidate profile from MySQL history
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config "C:\HT9046MX\state\config\controlled_condition_monitoring.json" bootstrap

# Run one five-minute scoring cycle
.\scripts\run_controlled_monitoring_cycle.ps1 `
  -SystemConfig "C:\HT9046MX\state\config\controlled_condition_monitoring.json"
```
