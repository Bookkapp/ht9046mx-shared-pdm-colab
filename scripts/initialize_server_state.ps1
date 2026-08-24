param(
    [string]$StateRoot = "C:\HT9046MX\state",
    [string]$DataRoot = "C:\HT9046MX\data\incoming"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$stateRootPath = [System.IO.Path]::GetFullPath($StateRoot)
$dataRootPath = [System.IO.Path]::GetFullPath($DataRoot)
$configDirectory = Join-Path $stateRootPath "config"
$runtimeDirectory = Join-Path $stateRootPath "controlled_runtime"
$syncStateDirectory = Join-Path $stateRootPath "sync_state"
$logDirectory = Join-Path $stateRootPath "logs"
$handlersPath = Join-Path $configDirectory "handlers.json"
$monitoringPath = Join-Path $configDirectory "controlled_condition_monitoring.json"

foreach ($directory in @(
    $configDirectory,
    $runtimeDirectory,
    $syncStateDirectory,
    $logDirectory,
    $dataRootPath
)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

if (-not (Test-Path -LiteralPath $handlersPath)) {
    $sourceHandlers = Join-Path $projectRoot "compressor_fastapi_react_dashboard\backend\config\handlers.json"
    $handlers = Get-Content -Raw -LiteralPath $sourceHandlers | ConvertFrom-Json
    foreach ($handler in $handlers) {
        $handler.destination = Join-Path $dataRootPath ("Comp_log_data_{0}" -f $handler.name)
    }
    [System.IO.File]::WriteAllText(
        $handlersPath,
        (($handlers | ConvertTo-Json -Depth 10) + [Environment]::NewLine),
        $utf8NoBom
    )
    Write-Output "Created persistent handler registry: $handlersPath"
}
else {
    Write-Output "Kept existing persistent handler registry: $handlersPath"
}

if (-not (Test-Path -LiteralPath $monitoringPath)) {
    $templatePath = Join-Path $projectRoot "configs\controlled_condition_monitoring.server.template.json"
    $template = Get-Content -Raw -LiteralPath $templatePath
    # Paths are inserted into a JSON string, so literal Windows separators
    # must be escaped before rendering the template.
    $jsonProjectRoot = $projectRoot.Replace("\", "\\")
    $jsonStateRoot = $stateRootPath.Replace("\", "\\")
    $rendered = $template.Replace("__APP_ROOT__", $jsonProjectRoot).Replace("__STATE_ROOT__", $jsonStateRoot)
    $null = $rendered | ConvertFrom-Json
    [System.IO.File]::WriteAllText($monitoringPath, $rendered + [Environment]::NewLine, $utf8NoBom)
    Write-Output "Created persistent monitoring config: $monitoringPath"
}
else {
    Write-Output "Kept existing persistent monitoring config: $monitoringPath"
}

Write-Output ""
Write-Output "Persistent state is ready. Configure the Dashboard .env with:"
Write-Output "HANDLERS_FILE=$handlersPath"
Write-Output "CONTROLLED_SYSTEM_CONFIG=$monitoringPath"
Write-Output "CONTROLLED_RUNTIME_DIR=$runtimeDirectory"
Write-Output "HANDLER_DESTINATION_ROOT=$dataRootPath"
