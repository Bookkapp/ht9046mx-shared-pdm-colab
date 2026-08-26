param(
    [string]$StateRoot = "C:\HT9046MX\state",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$stateRootPath = [System.IO.Path]::GetFullPath($StateRoot)
$configDirectory = Join-Path $stateRootPath "config"
$runtimeDirectory = Join-Path $stateRootPath "controlled_runtime"
$logDirectory = Join-Path $stateRootPath "logs"
$monitoringPath = Join-Path $configDirectory "controlled_condition_monitoring.json"

foreach ($directory in @($configDirectory, $runtimeDirectory, $logDirectory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
if ($Force -or -not (Test-Path -LiteralPath $monitoringPath)) {
    $templatePath = Join-Path $projectRoot "configs\controlled_condition_monitoring.server.template.json"
    $template = Get-Content -Raw -LiteralPath $templatePath
    $jsonProjectRoot = $projectRoot.Replace("\", "\\")
    $jsonStateRoot = $stateRootPath.Replace("\", "\\")
    $rendered = $template.Replace("__APP_ROOT__", $jsonProjectRoot).Replace("__STATE_ROOT__", $jsonStateRoot)
    $null = $rendered | ConvertFrom-Json
    [System.IO.File]::WriteAllText($monitoringPath, $rendered + [Environment]::NewLine, $utf8NoBom)
    Write-Output "Created MySQL-only monitoring config: $monitoringPath"
}
else {
    Write-Output "Kept existing monitoring config: $monitoringPath"
    Write-Output "Use -Force once to replace the earlier SMB/file-source configuration."
}

Write-Output ""
Write-Output "Persistent state is ready. Configure backend\.env with MySQL access, then run:"
Write-Output ".\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner --system-config $monitoringPath source-check"
Write-Output "CONTROLLED_SYSTEM_CONFIG=$monitoringPath"
Write-Output "CONTROLLED_RUNTIME_DIR=$runtimeDirectory"
