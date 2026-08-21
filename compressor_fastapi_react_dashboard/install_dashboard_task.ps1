param(
    [string]$TaskName = "HT9046MX-Model-Monitor",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_dashboard.ps1"
$powerShell = (Get-Command powershell.exe).Source
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -Port $Port"
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "HT9046MX React + FastAPI Controlled Hybrid model monitor" -Force
Write-Output "Installed startup task: $TaskName"
Write-Output "Dashboard URL after startup: http://SERVER_IP:$Port"
