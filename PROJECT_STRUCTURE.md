# HT9046MX Project Structure

This repository is the source/application area. On a real Server it is deployed
to `C:\HT9046MX\app`; live data and configuration must remain outside Git. See
[`SERVER_FOLDER_LAYOUT.md`](SERVER_FOLDER_LAYOUT.md) for the permanent server
tree and backup ownership.

## Repository root

```text
Data Analysis\
├── artifacts\                              Immutable Shared LSTM package
├── compressor_ml\                          Python data/model/SMB-sync code
├── compressor_fastapi_react_dashboard\     FastAPI + React Dashboard
├── configs\                                Versioned policies and config templates
├── scripts\                                Setup and Task Scheduler wrappers
├── tests\                                  Automated tests
├── HT9046MX_Shared_Model_Colab.ipynb        Shared LSTM training notebook
├── README.md                                Deployment/operator guide
├── CONTROLLED_HYBRID_SYSTEM.md              Technical pipeline design
├── PROJECT_STRUCTURE.md                     This guide
└── SERVER_FOLDER_LAYOUT.md                  Server folder guide
```

## Deployable folders

| Folder | Responsibility | Important contents | Deploy? |
|---|---|---|---|
| `artifacts\shared_lstm_colab_full` | Immutable 30-epoch Shared LSTM. Production scores with it but never changes weights. | `shared_model.keras`, `manifest.json`, `config.json`, `thresholds.json`, `scalers\`, `group_metrics.csv` | Yes |
| `compressor_ml` | Python log parsing, features, COM2, LSTM shadow, lifecycle, profile interfaces, and SMB copy code. | Details below | Yes |
| `compressor_fastapi_react_dashboard` | Web API/UI: Dashboard views model state and writes handler configuration. | `backend\`, `frontend\`, dashboard scripts | Yes |
| `configs` | Versioned policy and templates. Live runtime config is kept outside this folder. | Policy JSON, server-state template, local dev config | Yes |
| `scripts` | Repeatable initialization, scheduler installation, validation and offline tools. | SMB/model runners and install scripts | Yes |
| `tests` | Regression tests run before a release. | SMB, model and Controlled Hybrid pipeline tests | Optional but recommended |

### `artifacts\shared_lstm_colab_full`

| File/folder | Role |
|---|---|
| `shared_model.keras` | TensorFlow/Keras sequence model trained in Colab for 30 epochs. |
| `manifest.json` | Version, expected shape, and immutable-artifact provenance. |
| `config.json` | Feature/window configuration required for correct inference. |
| `thresholds.json` | Group-specific reconstruction-error thresholds for LSTM shadow scoring. |
| `scalers\` | Input normalizers stored from training. |
| `group_metrics.csv` | Per-group training/validation metrics. |
| `data_quality_summary.csv` | Training quality evidence; not a live runtime input. |
| `inference_*.csv` | Example/offline inference evidence; not live predictions. |

### `compressor_ml`

| File/folder | Role |
|---|---|
| `preprocessing.py` | Parses handler logs, timestamps, types and normalized sensor columns. |
| `prepare_dataset.py` | Discovers dated log files and prepares train/offline data. |
| `features.py`, `windowing.py` | General feature and window helpers. |
| `model.py`, `train.py`, `inference.py` | Shared-model training/inference utilities for the Colab workflow. |
| `smb_sync.py` | Five-minute incremental SMB copier. Reads `handlers.json`, safely copies changed `.txt/.csv/.log` files, and writes `latest_sync.json`. |
| `anomaly.py`, `config.py`, `shared_artifact.py` | Shared LSTM scaling, configuration, reconstruction error, and immutable artifact loading. |
| `controlled_monitoring\` | Current production Controlled Hybrid pipeline. |

Current production files in `compressor_ml\controlled_monitoring\`:

| File | Role |
|---|---|
| `runner.py` | Five-minute CLI/orchestrator. Loads the persistent handler registry and scores each machine's copied logs. |
| `config.py` | Loads/validates Controlled Hybrid policy. |
| `context.py` | Identifies operating context such as SV/valve states and stable regions. |
| `profiles.py` | Frozen profile components: robust baseline, GMM, residual model and Isolation Forest. |
| `engine.py` | Generates COM2 evidence without changing a frozen profile. |
| `shadow.py` | Loads immutable Shared LSTM and produces reconstruction evidence. |
| `fusion.py` | Applies persistence and produces review level, not automatic shutdown. |
| `lifecycle.py` | Manages data collection, candidate, shadow, approval, and active profile states. |
| `windowing.py` | Converts accepted data into contextual/fixed-time windows. |

### `compressor_fastapi_react_dashboard`

```text
compressor_fastapi_react_dashboard\
├── backend\
│   ├── app\                            FastAPI code
│   ├── config\handlers.json            Development seed only
│   ├── .env.example                    Server environment template
│   ├── requirements.txt                Dashboard Python packages
│   └── tests\                         API and handler-store tests
├── frontend\
│   ├── src\                            React source code
│   ├── dist\                           Built static site (`npm run build`)
│   ├── package.json                    Node packages and build command
│   └── .env.example                    Frontend configuration example
├── install_dashboard.ps1               Install dependencies and build React
├── install_dashboard_task.ps1          Optional Dashboard startup task
├── run_dashboard.ps1                   Start FastAPI + built UI
└── verify_dashboard.ps1                Health/API verification
```

Backend `app`:

| File | Role |
|---|---|
| `api.py` | HTTP endpoints: health, handler CRUD, sync status, fleet/model views, comparisons, and profile actions. |
| `settings.py` | Reads `.env`, including paths to permanent Server state/data. |
| `handler_store.py` | Validates and atomically writes the one persistent `handlers.json`; makes `.bak` first and never returns passwords. |
| `model_store.py` | Read-only API layer for predictions, profiles, copied source logs, model artifact and sync status. |
| `catalog.py` | Parameter/signal catalog used by Dashboard. |
| `run_server.py` | Uvicorn start point. |

Frontend `src`:

| File/folder | Role |
|---|---|
| `pages\FleetPage.jsx` | Fleet model/lifecycle summary. |
| `pages\MachinePage.jsx` | Machine/module charts, COM2, LSTM, profile and evidence view. |
| `pages\ComparePage.jsx` | Cross-machine/module signal comparison. |
| `pages\PipelinePage.jsx` | Model equations, gates and lifecycle explanation. |
| `pages\HandlersPage.jsx` | Adds a machine with code + IP, shows permanent paths and last SMB-sync status. |
| `components\` | Shared shell, branding, buttons, badges and chart wrapper. |
| `api.js`, `hooks.js`, `format.js`, `charts.js` | API calls, polling, formatting and chart setup. |
| `styles.css`, `theme.css` | White Analog Devices-style UI. |

### `configs`

| File | Use |
|---|---|
| `controlled_condition_monitoring_policy.json` | Versioned data-quality, context, profile, COM2/LSTM fusion, persistence and approval policy. |
| `controlled_condition_monitoring.server.template.json` | Rendered once by the initialization script into external permanent server config. |
| `controlled_condition_monitoring.json` | Local/dev config. `handlers_file` overrides its legacy `machine_sources` for enabled machines. Do not use it as the permanent server config. |

### `scripts`

| File | Use |
|---|---|
| `initialize_server_state.ps1` | Creates permanent `state`/`data`, seeds persistent config only when absent, and prints the `.env` values. |
| `run_smb_sync_cycle.ps1` | Runs one SMB copy cycle; use it to validate shares before scheduling. |
| `install_smb_sync_task.ps1` | Creates `HT9046MX-SMB-Sync` every five minutes. |
| `run_controlled_monitoring_cycle.ps1` | Runs one model-scoring cycle from local copied logs. |
| `install_controlled_monitoring_task.ps1` | Creates `HT9046MX-Controlled-Monitoring` every five minutes. |
| `build_colab_package.py` | Builds a Colab upload/training package. |

### `tests`

| File | Protects |
|---|---|
| `test_smb_sync.py` | Handler precedence, disabled handler behavior, incremental/nested copy, and oversize handling. |
| `test_controlled_monitoring.py` | Lifecycle, profiles, shadow scoring, fusion, and runner behavior. |
| `test_pipeline.py` | Shared pipeline/training/inference contracts. |

## Local analysis and generated folders

| Folder | What it is | Server guidance |
|---|---|---|
| `Clean Data MX12`, `Clean Data MX25`, `MX_007`, `MX017`, `MX057`, `MX070` | Local raw/cleaned compressor logs for analysis and smoke tests. | Kept out of Git; do not make them a production runtime dependency. |
| `prepared_dataset` | Intermediate training datasets. | Local/Colab preparation output. |
| `analysis_output` | Exploratory results, charts and metrics. | Local/reporting output. |
| `controlled_runtime` | Local development profiles/predictions/cache. | Git-ignored; production uses external `state\controlled_runtime`. |
| `.venv`, `node_modules`, `__pycache__`, `.pytest_cache` | Local packages and caches. | Never commit; recreate on each Server. |

## Permanent production folders (outside Git)

```text
C:\HT9046MX\
├── app\                                  Git checkout only
├── data\incoming\Comp_log_data_MX###\   SMB-copied log files
└── state\
    ├── config\handlers.json              Dashboard-owned machine registry
    ├── config\controlled_condition_monitoring.json
    ├── controlled_runtime\               Profiles, predictions, lifecycle/cache
    ├── sync_state\                       Copy signatures and latest_sync.json
    └── logs\                             Scheduled-task logs
```

`handlers.json` is the one registry shared by all live components:

```text
Dashboard Add/Update
        ↓
state\config\handlers.json
        ├── HT9046MX-SMB-Sync → data\incoming\Comp_log_data_MX###
        └── HT9046MX-Controlled-Monitoring → state\controlled_runtime
```

As a result, a `git pull` updates application code/templates under `app` but
does not overwrite machine configuration, source logs, profiles, or prior
predictions. Back up both `C:\HT9046MX\data` and `C:\HT9046MX\state`.
