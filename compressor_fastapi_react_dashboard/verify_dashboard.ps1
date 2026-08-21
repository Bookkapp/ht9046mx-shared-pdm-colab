param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$health = Invoke-RestMethod -Uri "$BaseUrl/api/v1/health"
$artifact = Invoke-RestMethod -Uri "$BaseUrl/api/v1/model/artifact"
$fleet = Invoke-RestMethod -Uri "$BaseUrl/api/v1/model/fleet"

Write-Output "API status: $($health.status)"
Write-Output "Model: $($artifact.model_version) · groups $($artifact.group_count) · epochs $($artifact.epochs_completed)"
Write-Output "Handlers: $($fleet.summary.configured_handlers) · data sources $($fleet.summary.data_sources_available)"
if ($health.status -ne "ready") {
    throw "Dashboard is running but one or more readiness checks failed. Inspect /api/v1/health."
}
