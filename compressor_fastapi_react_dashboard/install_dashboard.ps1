param(
    [string]$PythonCommand = "py"
)

$ErrorActionPreference = "Stop"
$dashboardRoot = $PSScriptRoot
$backendRoot = Join-Path $dashboardRoot "backend"
$frontendRoot = Join-Path $dashboardRoot "frontend"
$venvPython = Join-Path $backendRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($PythonCommand -eq "py") {
        & py -3 -m venv (Join-Path $backendRoot ".venv")
    }
    else {
        & $PythonCommand -m venv (Join-Path $backendRoot ".venv")
    }
    if ($LASTEXITCODE -ne 0) { throw "Unable to create backend virtual environment" }
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $backendRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed" }

Push-Location $frontendRoot
try {
    if (Test-Path -LiteralPath (Join-Path $frontendRoot "package-lock.json")) {
        & npm ci
    }
    else {
        & npm install
    }
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed" }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw "React production build failed" }
}
finally {
    Pop-Location
}

$environmentFile = Join-Path $backendRoot ".env"
if (-not (Test-Path -LiteralPath $environmentFile)) {
    Copy-Item -LiteralPath (Join-Path $backendRoot ".env.example") -Destination $environmentFile
    Write-Warning "Created backend\.env from the example. Set API_KEY and deployment paths before production use."
}

Push-Location $backendRoot
try {
    & $venvPython -c "from app.settings import settings; settings.controlled_runtime_dir.mkdir(parents=True, exist_ok=True)"
    if ($LASTEXITCODE -ne 0) { throw "Unable to initialize the Controlled Hybrid runtime directory" }
}
finally {
    Pop-Location
}

Write-Output "Dashboard installation completed."
Write-Output "Next: edit backend\.env, then run .\run_dashboard.ps1"
