# run_topic_formation.ps1

param(
    [string]$Template = "templates/template.yaml",
    [string]$TaskVars = "templates/tsc.yaml",
    [string]$RunDir = "runs/openai_run_001",
    [int]$NumExperts = 4,
    [int]$MaxRounds = 5,
    [string]$Model = "gpt-4.1-mini",
    [switch]$DryRun,
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"

# Move to the directory containing this launcher script.
Set-Location $PSScriptRoot

# Optional: activate local virtual environment if present.
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

$argsList = @(
    "run_topic_formation.py",
    "--template", $Template,
    "--task-vars", $TaskVars,
    "--run-dir", $RunDir,
    "--num-experts", "$NumExperts",
    "--max-rounds", "$MaxRounds",
    "--model", $Model
)

if ($DryRun) {
    $argsList += "--dry-run"
}

if ($NonInteractive) {
    $argsList += "--non-interactive"
}

python @argsList