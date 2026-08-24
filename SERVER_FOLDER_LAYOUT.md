# HT-9046MX Server Folder Layout

This is the required permanent layout for a production Windows server. It
separates Git-managed application files from logs, profiles, and configuration
that the Dashboard changes while the system is running.

```text
C:\HT9046MX\
├── app\                                  Git checkout; replaceable by git pull
│   ├── artifacts\shared_lstm_colab_full\ Immutable Shared LSTM artifact
│   ├── compressor_ml\                     Python condition-monitoring and SMB sync code
│   ├── compressor_fastapi_react_dashboard\ React + FastAPI application
│   ├── configs\                           Versioned policy and server-state template
│   ├── scripts\                           Scheduler and initialization scripts
│   └── .venv\                             Python environment built on this server only
│
├── data\
│   └── incoming\
│       ├── Comp_log_data_MX057\           Local copies from SMB handler MX057
│       ├── Comp_log_data_MX012\           Local copies after MX012 is added
│       └── ...                             Never edited by Dashboard or Git
│
└── state\                                 Persistent operational state; include in backup
    ├── config\
    │   ├── handlers.json                  Single machine registry owned by Dashboard
    │   └── controlled_condition_monitoring.json
    │                                        Runtime paths, policy/model references, and sync policy
    ├── controlled_runtime\                Model outputs, profiles, caches, and scoring audit
    ├── sync_state\                        File-copy signatures and latest SMB sync result
    └── logs\                              Scheduler logs for SMB sync and model scoring
```

## Folder ownership

| Folder | Owner | Contents | Backup | May `git pull` change it? |
|---|---|---|---|---|
| `C:\HT9046MX\app` | Git deployment | Source code, model artifact, scripts, policy template | Optional; Git is the recovery source | Yes |
| `C:\HT9046MX\app\.venv` | Server install | Python packages built for that server | No; recreate from requirements | No |
| `C:\HT9046MX\data\incoming` | SMB sync worker | Copied `.txt`, `.csv`, `.log` handler logs | Yes, according to log-retention policy | No |
| `C:\HT9046MX\state\config` | Dashboard/operator | Handler registry and server runtime config | Yes, essential | No |
| `C:\HT9046MX\state\controlled_runtime` | Model runner | Predictions, profile versions, lifecycle, audit, dashboard cache | Yes, essential | No |
| `C:\HT9046MX\state\sync_state` | SMB sync worker | Incremental-copy registry and last sync summary | Yes; small but useful | No |
| `C:\HT9046MX\state\logs` | Scheduled tasks | Text logs for investigation | Retain/archive by site policy | No |

## Persistent configuration files

### `state\config\handlers.json`

This is the **single source of truth for machines**. The Dashboard is allowed
to atomically create, update, disable, or remove records here; it first writes
a `.bak` copy. One record contains:

```json
{
  "name": "MX057",
  "enabled": true,
  "ip": "10.196.132.182",
  "share_path": "\\\\10.196.132.182\\Comp_log_data_MX057",
  "source_subfolder": "",
  "destination": "C:\\HT9046MX\\data\\incoming\\Comp_log_data_MX057",
  "timezone": "Asia/Bangkok",
  "username": "",
  "password_env": "",
  "notes": "Managed from the Controlled Hybrid model monitor"
}
```

Adding `MX012` in the Dashboard creates the UNC share and destination path
from the machine code and IP. It does not store a password. The next SMB sync
cycle discovers the new record; the next model cycle reads its local
destination. Disabling a handler excludes it from future sync and model
scoring but does not delete copied logs or historical profiles.

### `state\config\controlled_condition_monitoring.json`

This is a small server runtime file, created once from
`app\configs\controlled_condition_monitoring.server.template.json`. It holds
absolute paths to the policy, immutable model, persistent runtime directory,
and the same `handlers.json` registry. Its `machine_sources` is intentionally
empty in the standard deployment: enabled handler destinations are resolved
from `handlers.json` at runtime. The field remains only for offline replay
workstations that have no persistent handler registry.

The `sync` section controls safe incremental copying: allowed extensions,
per-machine file limit, maximum file size, copy buffer, and SMB connection
mode. Default `direct` mode accesses the UNC path using the Windows account
running the scheduled task. Use `guest` only when the site has explicitly
enabled unauthenticated Guest SMB access.

## Operational output folders

### `state\controlled_runtime`

The model runner creates and updates these files. Do not edit them manually.

```text
controlled_runtime\
├── latest_cycle.json                         Most recent five-minute model cycle
├── predictions\MX057.jsonl                   Append-only COM2/LSTM decisions
├── profiles\                                 Candidate/active frozen profile bundles
├── state\processed_files.json                Files already processed by model scoring
├── state\persistence.json                    COM2/LSTM persistence state
├── monitoring\source_errors.jsonl            Parse/read errors from log sources
├── dashboard_cache\windows\*.joblib          Rebuildable dashboard window cache
└── scheduler_logs\cycle_*.log                Model task output
```

`predictions` and `profiles` are audit evidence. `dashboard_cache` is
rebuildable, but it is safe to retain. The whole `controlled_runtime` folder
must be backed up before an OS migration or a deployment rollback.

### `state\sync_state`

```text
sync_state\
├── sync_state.json                            Source size/mtime signature per copied file
└── latest_sync.json                           Per-handler result from the last sync cycle
```

The sync worker copies a file only when source size or modified time changes,
or when its local destination is missing. It writes to a temporary `.partial`
file and atomically replaces the destination only after the source stayed
stable throughout the copy. It never deletes an existing local log file.

## Initialization and task flow

Run once after cloning the Git checkout to `C:\HT9046MX\app`:

```powershell
cd C:\HT9046MX\app
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\initialize_server_state.ps1
```

Then configure Dashboard `backend\.env` to point at the `state` and `data`
paths printed by that script. Install two independent five-minute tasks:

```text
HT9046MX-SMB-Sync
    handlers.json -> SMB shares -> data\incoming -> state\sync_state

HT9046MX-Controlled-Monitoring
    handlers.json + data\incoming -> controlled_runtime\predictions/profiles
```

Both tasks must run under a Windows account that can read the SMB shares. The
Dashboard task only serves the web application; it does not copy data.
