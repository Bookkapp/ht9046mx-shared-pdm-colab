# HT9046MX Model Monitor Dashboard

FastAPI + React dashboard deployed on `10.195.17.69`. It reads telemetry
directly and read-only from MySQL on `10.195.17.73`.

It does not read handler log files, SMB shares, `handlers.json`, or local
copied data. Machines appear automatically from distinct `machine_number`
values in `ht9046mx_iot.ht9046mx_readings`.

## What the dashboard reads

- MySQL telemetry for raw/engineered five-minute charts and comparisons.
- `state\controlled_runtime\predictions` for COM2/LSTM decisions.
- `state\controlled_runtime\profiles` for candidate/active frozen profiles.
- `artifacts\shared_lstm_colab_full` for Shared LSTM training evidence.

`backend\.env` must set `MYSQL_HOST=10.195.17.73`, a read-only MySQL user and
the readings-table mappings. See the root [README](../README.md) for setup.
Set `MYSQL_STALE_AFTER_MINUTES=30` to mark a machine ONLINE only when its
latest MySQL `recorded_at` is at most 30 minutes old; the Fleet page shows the
event timestamp and age for every machine.

Run locally on the Web App server:

```powershell
cd "C:\HT9046MX\app\compressor_fastapi_react_dashboard"
.\install_dashboard.ps1
.\run_dashboard.ps1
```

Open `http://10.195.17.69:8000`.
