param(
    [string]$SystemConfig = "configs\controlled_condition_monitoring.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot $SystemConfig
$logDirectory = Join-Path $projectRoot "controlled_runtime\scheduler_logs"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Controlled monitoring config not found: $configPath"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "cycle_$stamp.log"

Push-Location $projectRoot
try {
    & $pythonPath -m compressor_ml.controlled_monitoring.runner `
        --system-config $SystemConfig cycle 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Controlled monitoring cycle failed with exit code $LASTEXITCODE. See $logPath"
    }
}
finally {
    Pop-Location
}
