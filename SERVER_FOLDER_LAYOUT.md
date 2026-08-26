# HT9046MX Two-Server Layout

There are two different machines with two different responsibilities. They do
not share an application folder, Python environment, model state folder, or
Task Scheduler task.

## Database Server — `10.195.17.73`

This project does **not** install, copy, or initialize any folder on this
machine. It contains the existing MySQL service and existing ingestion process:

```text
MySQL service
└── ht9046mx_iot
    └── ht9046mx_readings       Existing live telemetry table
```

The database administrator owns its database backup, table retention,
importer, database user, and firewall policy. The only requirement for this
project is that the Web/model Server `10.195.17.69` has read-only TCP access to
the configured table on port 3306.

## Web/model Server — `10.195.17.69`

Everything below exists only on the Web/model Server. It reads telemetry
read-only from the Database Server; it never connects to SMB shares or copies
handler log files.

```text
C:\HT9046MX\
├── app\                                      Git checkout on Web/model Server
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
up by this project. It exists on `10.195.17.69` and contains frozen profiles,
model decisions, lifecycle approvals and the MySQL time cursor. MySQL remains
the source of telemetry truth and is backed up separately by the Database
Server owner.

## Connection contract between the two servers

Credentials belong only on the Web/model Server in:

```text
C:\HT9046MX\app\compressor_fastapi_react_dashboard\backend\.env
```

The shipped configuration uses these defaults, all overrideable in the
Web/model Server `.env`:

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

## Scheduled flow across separate machines

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
