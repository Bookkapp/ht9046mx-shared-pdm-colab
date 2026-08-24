# HT9046MX Controlled Hybrid Model Monitor

Production-oriented React/Vite + FastAPI dashboard for inspecting the model and
pipeline rather than presenting an unsupported failure probability.

## What the dashboard reads

- Raw/engineered machine graphs: synchronized handler log files.
- COM2 and LSTM decisions: `controlled_runtime/predictions/<machine>.jsonl`.
- Candidate and Active Frozen Profiles: versioned joblib bundles under
  `controlled_runtime/profiles`.
- Shared LSTM train evidence: the Full artifact manifest, group metrics,
  thresholds, scaler availability, and immutable Keras file.
- Handler onboarding: the persistent `C:\HT9046MX\state\config\handlers.json`
  registry; write operations are atomic, keep a `.bak`, and never return
  password fields. The Dashboard never writes the Git checkout.

MySQL is not required for this model-monitor surface. It can run beside the
existing file-sync/MySQL dashboard without changing that database.

## Pages

- **Fleet**: lifecycle state, data availability, trained LSTM groups, active
  frozen profiles, and P1/P2 review counts.
- **Machine Monitor**: COM1/COM2/thermal/control graphs, correlation matrix,
  Robust Z/MAD, Ridge expected LP2 and residual, Isolation Forest, Shared LSTM,
  profile parameters, equations, and controlled approval actions.
- **Compare**: compare any raw, engineered, COM2, LSTM, or GMM metric across up
  to six machine/module series; Pearson correlation uses aligned event times.
- **Pipeline & Model**: artifact version, 30-epoch train/validation evidence,
  60×24 input features, policy thresholds, equations, data sources, and
  lifecycle.
- **Handlers**: add a handler using only machine code and IP, update/disable it,
  and make it available to both SMB sync and model scoring through the same
  permanent registry.

## Deploy on another Windows server

Deploy the Git checkout to `C:\HT9046MX\app`, then create permanent state
outside it. See [SERVER_FOLDER_LAYOUT.md](../SERVER_FOLDER_LAYOUT.md) for the
full contract.

```text
C:\HT9046MX\
├── app\
│   ├── compressor_ml\
│   ├── configs\
│   ├── artifacts\shared_lstm_colab_full\
│   └── compressor_fastapi_react_dashboard\
├── data\incoming\
└── state\
    ├── config\handlers.json
    └── controlled_runtime\
```

Open PowerShell:

```powershell
cd "C:\HT9046MX\app"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\initialize_server_state.ps1
cd .\compressor_fastapi_react_dashboard
.\install_dashboard.ps1
notepad .\backend\.env
.\run_dashboard.ps1
```

`run_dashboard.ps1` uses `DASHBOARD_HOST` and `DASHBOARD_PORT` from
`backend\.env`. Command-line `-HostAddress` or `-Port` values override them for
that process only.

Required `.env` changes:

1. Set `MODEL_PROJECT_ROOT` and the Controlled Runtime/Artifact paths.
2. Set a non-empty `API_KEY` before exposing write endpoints.
3. Keep `DASHBOARD_HOST=0.0.0.0` and select the required port.
4. Point `HANDLERS_FILE`, `CONTROLLED_SYSTEM_CONFIG`, and
   `CONTROLLED_RUNTIME_DIR` to `C:\HT9046MX\state`.
5. Verify `HANDLER_DESTINATION_ROOT=C:\HT9046MX\data\incoming`.

Open `http://SERVER_IP:8000`. If Windows Firewall blocks the port, an
administrator can create an inbound TCP rule for port 8000 according to the
site's IT policy.

Optional startup task, from an Administrator PowerShell. The task runs as the
local `SYSTEM` service account at Windows startup, so no interactive logon or
stored user password is required:

```powershell
.\install_dashboard_task.ps1 -Port 8000
```

The Dashboard is not an SMB copier. Install the SMB sync task and then the
separate model-scoring task:

```powershell
cd "C:\HT9046MX\app"
.\scripts\install_smb_sync_task.ps1 -IntervalMinutes 5 `
  -SystemConfig "C:\HT9046MX\state\config\controlled_condition_monitoring.json"
.\scripts\install_controlled_monitoring_task.ps1 -IntervalMinutes 5 `
  -SystemConfig "C:\HT9046MX\state\config\controlled_condition_monitoring.json"
```

Verify after both processes are running:

```powershell
cd .\compressor_fastapi_react_dashboard
.\verify_dashboard.ps1
```

API documentation is available at `http://SERVER_IP:8000/docs`.

## Performance and safety

- Daily logs are aggregated to bounded five-minute points. Parsed windows are
  cached by source path/size/mtime under
  `C:\HT9046MX\state\controlled_runtime\dashboard_cache\windows`; changing
  a source file creates a new cache key.
- Prediction JSONL reads are tail-bounded and cached by file signature.
- FastAPI serves the versioned React production assets with immutable cache
  headers and sends `no-store` for model APIs.
- Handler/profile writes can be protected by `X-API-Key`.
- Approve/Reject changes lifecycle and audit history only. It does not stop a
  machine, retrain LSTM weights, or delete older profile versions.
- `NORMAL`, `SHADOW`, `P1_REVIEW`, and `P2_REVIEW` are condition-monitoring
  states, not failure probabilities or root-cause diagnoses.

## Validation

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest

cd ..\frontend
npm run build
```
