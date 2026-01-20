# launcher_eval_ppo_tsc.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$WorkDir    = "E:\repos\LLM_traffic_query"   # matches VS Code cwd: ${workspaceFolder}
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\eval_ppo_tsc.py"

# Match VS Code env
$env:SUMO_HOME = "E:\Sumo"

# Match VS Code cwd
Set-Location $WorkDir

$EvalArgs = @(
  "--checkpoint", "E:\repos\LLM_traffic_query\tests\sumo_traci\ppo_tsc_models\sumo_ppo_seed518_1768538101__J2.pt",
  "--log-tag", "nh3",
  "--log-dir", "tests\sumo_traci\eval_results\perdim",
  "--episodes", "20",
  "--episode-len", "3600",
  "--sumo-seed", "10086",
  "--deterministic",
#   "--zero-expert",
  "--zero-expert-dims", " ",
  "--noise-expert-dims", "15,16,17,18,19,20",
  "--noise-sigma", "0.15"
)

& $PythonExe $ScriptPath @EvalArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

