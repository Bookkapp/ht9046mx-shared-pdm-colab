# HT9046MX MySQL-only Server Layout

The model application is deployed on **10.195.17.69**. It reads telemetry
read-only from MySQL on **10.195.17.73**; it never connects to SMB shares or
copies handler log files.

```text
C:\HT9046MX\
├── app\                                      Git checkout on 10.195.17.69
│   ├── artifacts\shared_lstm_colab_full\      Immutable 30-epoch Shared LSTM
│   ├── compressor_ml\                         MySQL reader + model pipeline
│   ├── compressor_fastapi_react_dashboard\    FastAPI + React web application
│   ├── configs\                               Versioned policy/template files
│   ├── scripts\                               Initialization and scheduled runner
│   └── .venv\                                 Server Python environment
│
└── state\                                     Persistent operational state
    ├── config\controlled_condition_monitoring.json
    ├── controlled_runtime\                    Profiles, predictions, lifecycle/audit
    └── logs\                                  Optional server task logs
```

`state\controlled_runtime` is the only operational folder that must be backed
up. It contains frozen profiles, model decisions, lifecycle approvals and the
MySQL time cursor. MySQL remains the source of telemetry truth.

## MySQL contract

Credentials belong only in:

```text
C:\HT9046MX\app\compressor_fastapi_react_dashboard\backend\.env
```

The shipped configuration uses these defaults, all overrideable in `.env`:

| Setting | Default |
|---|---|
| Database host | `10.195.17.73` |
| Database | `ht9046mx_iot` |
| Table | `ht9046mx_readings` |
| Machine column | `machine_number` |
| Event timestamp | `recorded_at` |
| Web App | `http://10.195.17.69:8000` |

The reader detects both accepted telemetry layouts:

- **Wide:** one time row with `Hp_1st_1`, `Lp_2nd_1`, …, `TempLo_8` columns.
- **Long:** one machine/module/time row with `module_id`, `hp1`, `lp2`, …

Run the following after creating `.env`. It validates the connection, shows
the discovered machines and returns the actual table columns without writing
to MySQL:

```powershell
cd "C:\HT9046MX\app"
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config "C:\HT9046MX\state\config\controlled_condition_monitoring.json" source-check
```

## Scheduled flow

```text
MySQL 10.195.17.73 (read-only telemetry)
                ↓ every 5 minutes
HT9046MX-Controlled-Monitoring on 10.195.17.69
                ↓
state\controlled_runtime (profiles, decisions, cursor)
                ↓
FastAPI + React Dashboard on 10.195.17.69:8000
```

No SMB task, UNC path, handler registry, `net use`, or local copied source
folder is part of this deployment.
