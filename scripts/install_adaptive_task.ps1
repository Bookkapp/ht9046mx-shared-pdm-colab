param(
    [string]$TaskName = "HT9046MX-Adaptive-Scoring",
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Stop"
if ($IntervalMinutes -lt 5) {
    throw "IntervalMinutes must be at least 5"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$runnerPath = Join-Path $PSScriptRoot "run_adaptive_cycle.ps1"
$artifactPath = Join-Path $projectRoot "artifacts\shared_lstm_colab_smoke\shared_model.keras"
$runtimeManifest = Join-Path $projectRoot "adaptive_runtime\seed_manifest.json"

if (-not (Test-Path -LiteralPath $artifactPath)) {
    throw "Shared model artifact is not ready: $artifactPath"
}
if (-not (Test-Path -LiteralPath $runtimeManifest)) {
    throw "Initialize adaptive_runtime before installing the task"
}

$powerShell = (Get-Command powershell.exe).Source
$actionArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runnerPath`""
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $actionArguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Automatic HT-9046MX scoring and guarded calibration validation" -Force

Write-Output "Installed scheduled task: $TaskName"
Write-Output "Interval: every $IntervalMinutes minutes"
