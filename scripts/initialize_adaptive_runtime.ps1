param(
    [string]$ArtifactDirectory = "artifacts\shared_lstm_colab_smoke",
    [string]$RuntimeDirectory = "adaptive_runtime"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$artifactPath = Join-Path $projectRoot $ArtifactDirectory
$seedPath = Join-Path $artifactPath "adaptive_seed"
$runtimePath = Join-Path $projectRoot $RuntimeDirectory

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $artifactPath "shared_model.keras"))) {
    throw "Shared model artifact not found. Download and extract the Colab adaptive bundle into $artifactPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $seedPath "seed_manifest.json"))) {
    throw "Adaptive seed not found: $seedPath"
}
if (Test-Path -LiteralPath $runtimePath) {
    $existing = Get-ChildItem -LiteralPath $runtimePath -Force
    if ($existing.Count -gt 0) {
        throw "Runtime directory is not empty: $runtimePath"
    }
}

Push-Location $projectRoot
try {
    & $pythonPath -m compressor_ml.adaptive_runner init --seed-dir $seedPath --runtime-dir $runtimePath
    if ($LASTEXITCODE -ne 0) {
        throw "Adaptive runtime initialization failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
