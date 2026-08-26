# HT9046MX Controlled Hybrid Model Monitor

This is a **MySQL-only** compressor condition-monitoring deployment.

- **Web/model server:** `10.195.17.69`
- **Telemetry database server:** `10.195.17.73`
- **Telemetry source:** `ht9046mx_iot.ht9046mx_readings` (read-only)
- **Model:** immutable 30-epoch Shared LSTM shadow plus Controlled COM2
  profiles, GMM context, Robust Z/MAD, Ridge LP2 residual and Isolation Forest.

The deployment does **not** use SMB, UNC shares, `net use`, handler IP setup,
file synchronization, or direct log-file scanning. MySQL is the only live
telemetry source.

## Runtime flow

```text
MySQL 10.195.17.73
    ht9046mx_iot.ht9046mx_readings
              │ read-only, every five minutes
              ▼
Controlled Hybrid runner on 10.195.17.69
              │
              ├── candidate/active frozen profiles
              ├── append-only model decisions
              └── MySQL time cursor
              ▼
FastAPI + React Dashboard
http://10.195.17.69:8000
```

The runner never retrains the Shared LSTM from live data and never writes back
to MySQL. Profile activation still requires human approval.

## Deploy on 10.195.17.69

### 1. Get the MySQL-only revision

Merge the MySQL-only pull request, then on the Web App server:

```powershell
cd "C:\HT9046MX\app"
git pull
Set-ExecutionPolicy -Scope Process Bypass
if (-not (Test-Path .\.venv\Scripts\python.exe)) { py -3.13 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\compressor_fastapi_react_dashboard\backend\requirements.txt
```

### 2. Create the permanent state configuration

The old SMB config is incompatible with this release. Render the new MySQL
configuration once with `-Force`:

```powershell
cd "C:\HT9046MX\app"
.\scripts\initialize_server_state.ps1 -Force
```

This creates:

```text
C:\HT9046MX\state\config\controlled_condition_monitoring.json
C:\HT9046MX\state\controlled_runtime\
```

If the earlier file-based release was installed, remove its obsolete task once
from an Administrator PowerShell. It is no longer used and would fail after
the SMB scripts are removed:

```powershell
Unregister-ScheduledTask -TaskName "HT9046MX-SMB-Sync" -Confirm:$false -ErrorAction SilentlyContinue
```

### 3. Configure MySQL credentials

Create `C:\HT9046MX\app\compressor_fastapi_react_dashboard\backend\.env`
from `.env.example`. Set credentials supplied by the database administrator;
do not commit this file.

```dotenv
MYSQL_HOST=10.195.17.73
MYSQL_PORT=3306
MYSQL_USER=your_readonly_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=ht9046mx_iot
READINGS_TABLE=ht9046mx_readings
READINGS_MACHINE_COLUMN=machine_number
READINGS_TIMESTAMP_COLUMN=recorded_at
READINGS_MODULE_COLUMN=

MODEL_PROJECT_ROOT=C:\HT9046MX\app
CONTROLLED_SYSTEM_CONFIG=C:\HT9046MX\state\config\controlled_condition_monitoring.json
CONTROLLED_RUNTIME_DIR=C:\HT9046MX\state\controlled_runtime
SHARED_MODEL_ARTIFACT=C:\HT9046MX\app\artifacts\shared_lstm_colab_full
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
```

Leave `READINGS_MODULE_COLUMN` blank for the normal wide source layout such
as `Hp_1st_1` through `TempLo_8`. Set it only when the database has one row per
module.

### 4. Validate MySQL before scoring

```powershell
cd "C:\HT9046MX\app"
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config "C:\HT9046MX\state\config\controlled_condition_monitoring.json" source-check
```

Expected result: JSON showing `"connected": true`, the readings table columns
and machine codes. This command is read-only.

### 5. Create profile candidates from history

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config "C:\HT9046MX\state\config\controlled_condition_monitoring.json" bootstrap
```

The first run reads the configured 120-day history per machine. To reduce the
initial load, run one known machine first:

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config "C:\HT9046MX\state\config\controlled_condition_monitoring.json" bootstrap `
  --machine-id MX057
```

### 6. Schedule every five minutes

Run PowerShell as Administrator:

```powershell
cd "C:\HT9046MX\app"
.\scripts\install_controlled_monitoring_task.ps1 -IntervalMinutes 5 `
  -SystemConfig "C:\HT9046MX\state\config\controlled_condition_monitoring.json"
Start-ScheduledTask -TaskName "HT9046MX-Controlled-Monitoring"
```

The task queries only new telemetry plus one short window of context. Its
cursor is stored in `state\controlled_runtime\state\mysql_cursors.json`.

### 7. Start the Dashboard

```powershell
cd "C:\HT9046MX\app\compressor_fastapi_react_dashboard"
.\install_dashboard.ps1
.\run_dashboard.ps1
```

Open [http://10.195.17.69:8000](http://10.195.17.69:8000). The health endpoint
is [http://10.195.17.69:8000/api/v1/health](http://10.195.17.69:8000/api/v1/health).
Allow inbound TCP 8000 in Windows Firewall according to site policy.

## MySQL source behaviour

The reader only issues `SELECT` / `SHOW COLUMNS`. It accepts:

1. **Wide rows:** timestamp plus module-suffixed fields such as `Hp_1st_1`,
   `Lp_2nd_1`, `Valve_1`, `TempHi_1`, `TempLo_1`, `Status_1`, `Busy_1`, `SV_1`.
2. **Long rows:** timestamp, a module column, and canonical fields such as
   `hp1`, `lp1`, `hp2`, `lp2`, `valve`, `temphi`, `templo`.

`PREDICTION_SENSOR_SCALE` is replaced by the `sensor_scale` policy in the
server config. Its default `auto` divides pressure/temperature by ten only
when integer-style source values make that necessary.

## Safety boundary

- Review levels are not failure probability, fault diagnosis or an automatic
  stop command.
- Candidate profiles must pass shadow validation and human approval before
  becoming `ACTIVE_FROZEN`.
- Back up `C:\HT9046MX\state\controlled_runtime`.
- Never commit `.env` or a database password.

See [SERVER_FOLDER_LAYOUT.md](SERVER_FOLDER_LAYOUT.md) for the permanent
server layout and [CONTROLLED_HYBRID_SYSTEM.md](CONTROLLED_HYBRID_SYSTEM.md)
for model logic.
