# Compatibility entry point: named TSC training jobs, with read-only planning by default.
param(
    [string[]]$Select = @("tsc-train-*"),
    [string]$LocalConfig = "",
    [string[]]$Variable = @(),
    [string]$PythonExe = "",
    [int]$Jobs = 1,
    [switch]$Run
)
& (Join-Path $PSScriptRoot "launch_experiment.ps1") -Config tsc -Select $Select `
    -LocalConfig $LocalConfig -Variable $Variable -PythonExe $PythonExe -Jobs $Jobs -Run:$Run
exit $LASTEXITCODE
