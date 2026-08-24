param(
    [string]$TaskName = "HT9046MX-SMB-Sync",
    [int]$IntervalMinutes = 5,
    [string]$SystemConfig = "configs\controlled_condition_monitoring.json"
)

$ErrorActionPreference = "Stop"
if ($IntervalMinutes -lt 5) {
    throw "IntervalMinutes must be at least 5"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$runnerPath = Join-Path $PSScriptRoot "run_smb_sync_cycle.ps1"
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
    throw "SMB sync config is missing: $configPath"
}

$powerShell = (Get-Command powershell.exe).Source
$actionArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runnerPath`" -SystemConfig `"$configPath`""
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $actionArguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Copies incremental HT9046MX handler logs from SMB shares to permanent local data folders" -Force

Write-Output "Installed scheduled task: $TaskName"
Write-Output "Interval: every $IntervalMinutes minutes"
Write-Output "System config: $configPath"
Write-Output "Run the task under a Windows account permitted to read the configured SMB shares."
