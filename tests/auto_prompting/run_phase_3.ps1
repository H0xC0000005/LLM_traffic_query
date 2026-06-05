# run_phase_3.ps1

param(
    [string]$Phase12RunDir = "E:\repos\LLM_traffic_query\tests\auto_prompting\runs\debug_single_agent_phase_1_2_web_5.1_assembweb_3232sk_1",
    [string]$ExpertStatsCsv = "E:\repos\LLM_traffic_query\tests\sumo_traci\tensorboard_logs_all\test\3232sk\sumo_ppo_seed172_base_ploop+_ub_enc_bounded_v2+expert_reward_unbiased_simple_v1_1780630987\expert_feature_reports\J0_expert_features_step52350.csv",
    [string]$BaselineStatsCsv = "E:\repos\LLM_traffic_query\tests\sumo_traci\tensorboard_logs_all\test\3232sk\sumo_ppo_seed172_base_ploop+_ub_enc_bounded_v2+expert_reward_unbiased_simple_v1_1780630987\baseline_feature_reports\J0_baseline_features_step52350.csv",
    [string]$BaselineFeatureDescription = "templates/tsc_scene_encoder_feature_blocks_semantics.yaml",
    [string]$EvaluatorTemplate = "templates/template_phase3_evaluator_general.yaml",
    [string]$ExpertTemplate = "templates/template_phase3_expert_correction.yaml",
    [string]$AssemblerTemplate = "templates/template_v3.yaml",
    [string]$TaskVars = "templates/tsc_phase2_improved_signal_context.yaml",
    [string]$OutputDir = "E:\repos\LLM_traffic_query\tests\auto_prompting\runs\phase3_test",
    [string]$ModelMeta = "gpt5.1",
    [string]$Model = "",
    [int]$MaxOutputTokens = 160000,
    [switch]$AssemblerWebSearch,
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

if ([string]::IsNullOrWhiteSpace($ExpertStatsCsv)) {
    throw "ExpertStatsCsv must be provided."
}
if ([string]::IsNullOrWhiteSpace($BaselineStatsCsv)) {
    throw "BaselineStatsCsv must be provided."
}

$argsList = @(
    "run_phase_3.py",
    "--phase12-run-dir", $Phase12RunDir,
    "--expert-stats-csv", $ExpertStatsCsv,
    "--baseline-stats-csv", $BaselineStatsCsv,
    "--baseline-feature-description", $BaselineFeatureDescription,
    "--evaluator-template", $EvaluatorTemplate,
    "--expert-template", $ExpertTemplate,
    "--assembler-template", $AssemblerTemplate,
    "--task-vars", $TaskVars,
    "--output-dir", $OutputDir,
    "--model-meta", $ModelMeta,
    "--max-output-tokens", "$MaxOutputTokens"
)

if (-not [string]::IsNullOrWhiteSpace($Model)) {
    $argsList += @("--model", $Model)
}

if ($AssemblerWebSearch) {
    $argsList += "--assembler-web-search"
}

if ($DryRun) {
    $argsList += "--dry-run"
}

if ($NonInteractive) {
    $argsList += "--non-interactive"
}

python @argsList
