# launcher_eval_ppo_tsc.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$WorkDir    = "E:\repos\LLM_traffic_query"   # matches VS Code cwd: ${workspaceFolder}
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\eval_ppo_tsc.py"

# Match VS Code env
$env:SUMO_HOME = "E:\Sumo"

# Match VS Code cwd
Set-Location $WorkDir

$EvalArgs = @(
#   "--checkpoint", "E:\repos\LLM_traffic_query\tests\sumo_traci\models_3LR23LR2\1\sumo_ppo_seed172_base1_600ep_2048roll_5step_1771590042__J1.pt",
#   "--checkpoint", "E:\repos\LLM_traffic_query\tests\sumo_traci\models_3LR23LR2\1\sumo_ppo_seed172_exp1_600ep_2048roll_5step_1771581502__J1.pt",
  "--checkpoint", "E:\repos\LLM_traffic_query\tests\sumo_traci\models_22+22+_2\sumo_ppo_seed172_exp1_600ep_1_1771926979__J1.pt",
  "--log-tag", "zeroblk_1_2",
  "--log-dir", "tests\sumo_traci\eval_results\22+22+_1",
  "--episodes", "200",
  "--episode-len", "3600",
  "--sumo-seed", "10086",
  "--deterministic",
#   "--zero-expert",
#  "0,1,2,3,4", "5,6,7,8,9,10,11,12,13,14,15,16", "17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32", "33,34,35,36"
  "--zero-expert-dims", "5,6,7,8,9,10,11,12,13,14,15,16",
  "--noise-expert-dims", " ",
  "--noise-sigma", "0.15"
)

& $PythonExe $ScriptPath @EvalArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

