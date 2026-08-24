param(
    [string]$SystemConfig = "configs\controlled_condition_monitoring.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if ([System.IO.Path]::IsPathRooted($SystemConfig)) {
    $configPath = [System.IO.Path]::GetFullPath($SystemConfig)
}
else {
    $configPath = Join-Path $projectRoot $SystemConfig
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "SMB sync config not found: $configPath"
}

$system = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
$stateDirectory = [string]$system.sync.state_dir
if (-not $stateDirectory) {
    throw "sync.state_dir is missing from $configPath"
}
if (-not [System.IO.Path]::IsPathRooted($stateDirectory)) {
    $stateDirectory = Join-Path (Split-Path -Parent $configPath) $stateDirectory
}
$logDirectory = Join-Path (Split-Path -Parent $stateDirectory) "logs\smb_sync"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "sync_$stamp.log"

Push-Location $projectRoot
try {
    & $pythonPath -m compressor_ml.smb_sync --system-config $configPath 2>&1 |
        Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "SMB sync cycle failed with exit code $LASTEXITCODE. See $logPath"
    }
}
finally {
    Pop-Location
}
