param(
    [string]$SystemConfig = "configs\adaptive_system.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot $SystemConfig
$logDirectory = Join-Path $projectRoot "adaptive_runtime\scheduler_logs"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Adaptive system config not found: $configPath"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "cycle_$stamp.log"

Push-Location $projectRoot
try {
    # Windows PowerShell 5 may turn TensorFlow's harmless stderr warnings into
    # terminating NativeCommandError records when ErrorActionPreference=Stop.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $pythonPath -m compressor_ml.adaptive_runner cycle --system-config $configPath 2>&1 |
            Tee-Object -FilePath $logPath
        $cycleExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($cycleExitCode -ne 0) {
        throw "Adaptive cycle failed with exit code $cycleExitCode. See $logPath"
    }
}
finally {
    Pop-Location
}
