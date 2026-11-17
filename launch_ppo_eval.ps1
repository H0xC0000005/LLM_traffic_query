# Existing-checkpoint evaluation; does not schedule training.
param(
    [string]$Checkpoint = "",
    [string]$ExtractorPath = "",
    [string]$LocalConfig = "",
    [string[]]$Variable = @(),
    [string]$PythonExe = "",
    [switch]$Run
)
if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) { $Variable += "checkpoint=$Checkpoint" }
if (-not [string]::IsNullOrWhiteSpace($ExtractorPath)) {
    $Extractor = @{ path = $ExtractorPath; symbol = "tsc_isolated_intersection_feature_vector" } | ConvertTo-Json -Compress
    $Variable += "extractor=$Extractor"
}
& (Join-Path $PSScriptRoot "launch_experiment.ps1") -Config tsc_existing_eval `
    -LocalConfig $LocalConfig -Variable $Variable -PythonExe $PythonExe -Run:$Run
exit $LASTEXITCODE
