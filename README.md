# HT-9046MX Controlled Hybrid Condition Monitoring

Production deployment uses **Controlled Hybrid v1**. The explainable COM2
pipeline is the primary condition-monitoring decision path, while the immutable
30-epoch Shared LSTM Autoencoder (`shared_lstm_full_v1`) supplies independent
shadow evidence and filters bootstrap history.

This is condition monitoring, not a validated failure probability, root-cause
diagnosis, or remaining-useful-life model. A new machine can learn its own
candidate calibration profile automatically, but the first activation always
stops at `APPROVAL_REQUIRED`. A named engineer must approve it before the
versioned profile becomes `ACTIVE_FROZEN`.

For equations and the complete technical design, see
[CONTROLLED_HYBRID_SYSTEM.md](CONTROLLED_HYBRID_SYSTEM.md). For the React +
FastAPI model-monitor details, see
[compressor_fastapi_react_dashboard/README.md](compressor_fastapi_react_dashboard/README.md).

## Production architecture

```text
Handler compressor logs
        |
        | existing SMB/file-sync process
        v
C:\HT9046MX\Comp_log_data_MX###
        |
        | every 5 minutes
        v
quality/state gates -> 5-minute windows -> operating mode -> GMM regime
        |
        +-> COM2 primary evidence
        |     Robust Z/MAD + Ridge LP2 residual + Isolation Forest + trend
        |
        +-> Shared LSTM shadow evidence
        |     one frozen 30-epoch model + per-machine/module calibration
        v
fusion + persistence -> NORMAL / SHADOW / P1_REVIEW / P2_REVIEW
        |
        +-> controlled_runtime/predictions/<machine>.jsonl
        +-> candidate/profile lifecycle and audit history
        v
React dashboard served by FastAPI
```

The production components are:

- `compressor_ml/controlled_monitoring/`: Controlled Hybrid scoring,
  bootstrap, lifecycle, approval, and audit logic.
- `artifacts/shared_lstm_colab_full/`: deployable frozen Shared LSTM model,
  manifest, thresholds, metrics, and train-group scalers.
- `configs/controlled_condition_monitoring.json`: machine log sources and
  runtime paths.
- `configs/controlled_condition_monitoring_policy.json`: quality, profile,
  shadow, persistence, and approval policy.
- `scripts/run_controlled_monitoring_cycle.ps1`: one scoring cycle.
- `scripts/install_controlled_monitoring_task.ps1`: five-minute Windows Task
  Scheduler registration.
- `compressor_fastapi_react_dashboard/`: production React/Vite + FastAPI model
  monitor and handler configuration UI.

The older `compressor_ml.adaptive_runner` and `run_adaptive_cycle.ps1` remain
only for reproducibility and comparison. Do **not** install them as the primary
production scheduler.

## What is already included in GitHub

The repository includes the deployable Shared LSTM artifact:

```text
artifacts/shared_lstm_colab_full/
├── shared_model.keras
├── manifest.json
├── config.json
├── thresholds.json
├── group_metrics.csv
└── scalers/
```

Raw compressor logs, Python/Node environments, and generated
`controlled_runtime` data are intentionally excluded. The server generates the
runtime state after deployment.

## Deploy on a Windows server

### 1. Prerequisites

Install these before cloning:

- Git for Windows.
- 64-bit Python **3.13.14** (recommended for a new Windows server).
- Node.js 20.19+ or 22.12+ with npm.
- Administrator PowerShell only when registering Task Scheduler jobs.

The deployment uses CPU inference on native Windows. This is sufficient for
five-minute scoring. TensorFlow GPU is not supported on native Windows after
TensorFlow 2.10; use WSL2 only if GPU training/inference is explicitly needed.

The examples use `C:\HT9046MX\Data Analysis` because the committed production
configuration already points there.

### 2. Clone the production branch

```powershell
New-Item -ItemType Directory -Force C:\HT9046MX | Out-Null
git clone https://github.com/Bookkapp/ht9046mx-shared-pdm-colab.git `
  "C:\HT9046MX\Data Analysis"
cd "C:\HT9046MX\Data Analysis"
```

For a server that already has the repository:

```powershell
cd "C:\HT9046MX\Data Analysis"
git pull --ff-only origin main
```

Do not copy `.venv` or `node_modules` from another computer. Build them on the
target server.

### 3. Install the model runtime

```powershell
cd "C:\HT9046MX\Data Analysis"
Set-ExecutionPolicy -Scope Process Bypass
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "tensorflow==2.21.0"
```

The final command pins the currently supported TensorFlow wheel for Python
3.13 on Windows x64, rather than relying on a future untested TensorFlow
release. If `.venv` already exists and was created with another Python
version, do not reuse it: preserve it as a rollback copy, then create a fresh
`.venv` with Python 3.13 before continuing.

Confirm that the deployable model is present:

```powershell
Test-Path .\artifacts\shared_lstm_colab_full\shared_model.keras
.\.venv\Scripts\python.exe -c `
  "import tensorflow as tf; print(tf.__version__)"
.\.venv\Scripts\python.exe -c `
  "import tensorflow as tf; m=tf.keras.models.load_model(r'artifacts\shared_lstm_colab_full\shared_model.keras'); print(m.input_shape, m.output_shape)"
```

`Test-Path` must return `True`, TensorFlow must print `2.21.0`, and the final
command must print the model input/output shapes without an exception. This is
the required Python 3.13 smoke test before installing Task Scheduler jobs.

### 4. Prepare local compressor log folders

The model does not read the SMB share directly. It reads synchronized local
folders such as:

```text
C:\HT9046MX\Comp_log_data_MX017
C:\HT9046MX\Comp_log_data_MX057
C:\HT9046MX\Comp_log_data_MX070
```

Keep the existing SMB/file-sync Task Scheduler process responsible for copying
new logs into those folders. A mapped drive or `net use` connection alone does
not copy data into the model input directories.

Review `configs/controlled_condition_monitoring.json`. Its `machine_sources`
must contain the correct local folder for every enabled machine. The committed
file contains the current 14-machine production list.

Adding a handler through the Dashboard with `SYNC_CONTROLLED_SOURCES=true`
adds its generated local destination to this config atomically. It does not
create an SMB synchronization job.

### 5. Run and inspect the first bootstrap

Run one bootstrap manually before installing scheduled jobs:

```powershell
cd "C:\HT9046MX\Data Analysis"
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config configs\controlled_condition_monitoring.json bootstrap
```

Inspect all lifecycle states:

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config configs\controlled_condition_monitoring.json status
```

Expected behavior:

- Insufficient history remains `COLLECTING_DATA`; this is not an installation
  failure.
- Eligible history moves through `LEARNING` and creates a versioned candidate.
- A trained LSTM group uses its immutable training scaler and threshold.
- A new machine/module uses the same frozen Shared LSTM weights but fits a
  local train/validation scaler and reconstruction threshold from its own
  bootstrap history.
- The system automatically enters `SHADOW_VALIDATION` after producing a
  candidate.
- It moves to `APPROVAL_REQUIRED` only after the configured shadow gates pass.
- It never activates the first profile automatically.

Generated state is stored under `controlled_runtime/`. Back up this directory
because it contains candidate profiles, active frozen profiles, lifecycle
state, predictions, and audit history.

### 6. Install five-minute model scoring

Open PowerShell **as Administrator**:

```powershell
cd "C:\HT9046MX\Data Analysis"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install_controlled_monitoring_task.ps1 -IntervalMinutes 5
Start-ScheduledTask -TaskName "HT9046MX-Controlled-Monitoring"
```

Check the latest result and scheduler logs:

```powershell
Get-Content .\controlled_runtime\latest_cycle.json
Get-ChildItem .\controlled_runtime\scheduler_logs | `
  Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

The task ignores overlapping runs and never modifies the Shared LSTM weights.

### 7. Install the React + FastAPI dashboard

```powershell
cd "C:\HT9046MX\Data Analysis\compressor_fastapi_react_dashboard"
Set-ExecutionPolicy -Scope Process Bypass
.\install_dashboard.ps1
notepad .\backend\.env
```

At minimum, verify these values in `backend\.env`:

```dotenv
MODEL_PROJECT_ROOT=C:\HT9046MX\Data Analysis
CONTROLLED_SYSTEM_CONFIG=C:\HT9046MX\Data Analysis\configs\controlled_condition_monitoring.json
CONTROLLED_POLICY_FILE=C:\HT9046MX\Data Analysis\configs\controlled_condition_monitoring_policy.json
CONTROLLED_RUNTIME_DIR=C:\HT9046MX\Data Analysis\controlled_runtime
SHARED_MODEL_ARTIFACT=C:\HT9046MX\Data Analysis\artifacts\shared_lstm_colab_full
HANDLER_DESTINATION_ROOT=C:\HT9046MX
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
API_KEY=replace-with-a-long-random-secret
```

Start it interactively for the first verification:

```powershell
.\run_dashboard.ps1
```

Open `http://SERVER_IP:8000`, then verify from another PowerShell window:

```powershell
cd "C:\HT9046MX\Data Analysis\compressor_fastapi_react_dashboard"
.\verify_dashboard.ps1 -BaseUrl http://127.0.0.1:8000
```

The API documentation is at `http://SERVER_IP:8000/docs`. If another computer
cannot connect, ask the site administrator to allow inbound TCP port 8000 in
Windows Firewall.

### 8. Install dashboard startup

After the interactive check succeeds, stop it with `Ctrl+C`. Open PowerShell
**as Administrator** and register the startup task:

```powershell
cd "C:\HT9046MX\Data Analysis\compressor_fastapi_react_dashboard"
.\install_dashboard_task.ps1 -Port 8000
Start-ScheduledTask -TaskName "HT9046MX-Model-Monitor"
.\verify_dashboard.ps1 -BaseUrl http://127.0.0.1:8000
```

The dashboard and scoring jobs are intentionally separate:

- `HT9046MX-Controlled-Monitoring`: scores new data every five minutes.
- `HT9046MX-Model-Monitor`: starts React + FastAPI at Windows startup.

### 9. Approve a profile

Use the Machine Monitor page only after reviewing its data quality, COM2,
Shared LSTM, regime, residual, and shadow evidence. Enter the engineer name,
review note, and the same `API_KEY` configured in `backend\.env`.

CLI approval is also available:

```powershell
cd "C:\HT9046MX\Data Analysis"
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config configs\controlled_condition_monitoring.json approve `
  --machine-id MX017 --approved-by "Engineer.Name"
```

Approval creates a new versioned `ACTIVE_FROZEN` profile and audit event. It
does not retrain the Shared LSTM, stop the handler, or delete older versions.

## Data source and MySQL boundary

This model-monitor deployment currently reads:

- synchronized compressor log files for raw and engineered charts;
- `controlled_runtime` JSONL/profile/lifecycle outputs for model decisions;
- `artifacts/shared_lstm_colab_full` for immutable model evidence;
- `backend/config/handlers.json` for handler onboarding.

It does **not** read or write MySQL, so `MYSQL_*` variables from the older
server `.env` have no effect on this dashboard. It can run beside the existing
file-sync/MySQL application without altering that database. Connecting the
Controlled Hybrid outputs to MySQL is a separate production integration phase.

## Production validation checklist

Before operational use, confirm all of the following:

- The Shared LSTM file exists and TensorFlow can load it.
- Every configured machine source points to a folder receiving current logs.
- A manual scoring cycle finishes without error.
- `latest_cycle.json` advances after new data arrives.
- Both scheduled tasks run under an account with access to the deployment and
  log folders.
- `/api/v1/health` returns `ready`.
- `verify_dashboard.ps1` reports the expected model version, 30 epochs, and
  handler/data-source counts.
- The production `API_KEY` is not the example value.
- `controlled_runtime` is included in the server backup plan.
- Candidate profiles are reviewed for at least the configured shadow period
  before human approval.

## Developer validation

Run the model and dashboard test suites before releasing a code change:

```powershell
cd "C:\HT9046MX\Data Analysis"
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests

cd .\compressor_fastapi_react_dashboard\backend
.\.venv\Scripts\python.exe -m pytest

cd ..\frontend
npm ci
npm run build
```

## Retraining in Google Colab

Colab is required only when deliberately training a new Shared LSTM version; it
is not required for normal deployment, onboarding, scoring, or local profile
calibration. The current deployed model has already completed 30 epochs and is
committed under `artifacts/shared_lstm_colab_full`.

The retraining workflow is in `HT9046MX_Shared_Model_Colab.ipynb`. A retrained
artifact must be versioned and validated before replacing the production
artifact. The live scheduler must never update Shared LSTM weights by itself.
