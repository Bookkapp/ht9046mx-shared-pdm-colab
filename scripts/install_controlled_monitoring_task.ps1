param(
    [string]$TaskName = "HT9046MX-Controlled-Monitoring",
    [int]$IntervalMinutes = 5
)

$ErrorActionPreference = "Stop"
if ($IntervalMinutes -lt 5) {
    throw "IntervalMinutes must be at least 5"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$runnerPath = Join-Path $PSScriptRoot "run_controlled_monitoring_cycle.ps1"
$fullModelPath = Join-Path $projectRoot "artifacts\shared_lstm_colab_full\shared_model.keras"
$systemConfig = Join-Path $projectRoot "configs\controlled_condition_monitoring.json"

if (-not (Test-Path -LiteralPath $fullModelPath)) {
    throw "Full Shared LSTM model is not ready: $fullModelPath"
}
if (-not (Test-Path -LiteralPath $systemConfig)) {
    throw "Controlled monitoring config is missing: $systemConfig"
}

$powerShell = (Get-Command powershell.exe).Source
$actionArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runnerPath`""
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $actionArguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "COM2-primary condition monitoring with immutable Shared-LSTM shadow and human-approved frozen profiles" -Force

Write-Output "Installed scheduled task: $TaskName"
Write-Output "Interval: every $IntervalMinutes minutes"
Write-Output "Model: artifacts\shared_lstm_colab_full\shared_model.keras"
