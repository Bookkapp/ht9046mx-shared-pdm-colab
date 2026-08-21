param(
    [string]$HostAddress = "",
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$backendRoot = Join-Path $PSScriptRoot "backend"
$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $projectPython = Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $projectPython) {
        $python = $projectPython
    }
    else {
        throw "Dashboard virtual environment is missing. Run .\install_dashboard.ps1 first."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "frontend\dist\index.html"))) {
    throw "React production build is missing. Run .\install_dashboard.ps1 first."
}

Push-Location $backendRoot
try {
    if ($HostAddress) { $env:DASHBOARD_HOST = $HostAddress }
    if ($Port -gt 0) { $env:DASHBOARD_PORT = [string]$Port }
    & $python -m app.run_server
}
finally {
    Pop-Location
}
