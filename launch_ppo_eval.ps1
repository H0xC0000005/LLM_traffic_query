# launcher_eval_ppo_tsc.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$WorkDir    = "E:\repos\LLM_traffic_query"   # matches VS Code cwd: ${workspaceFolder}
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\eval_ppo_tsc.py"

# Match VS Code env
$env:SUMO_HOME = "E:\Sumo"

# Match VS Code cwd
Set-Location $WorkDir

& $PythonExe $ScriptPath `
  --checkpoint "tests\sumo_traci\ppo_tsc_models\sumo_ppo_seed517_1768478919__J2.pt" `
  --log-tag "zero" `
  --log-dir "tests\sumo_traci\eval_results" `
  --episodes 20 `
  --episode-len 3600 `
  --sumo-seed 10086 `
  --deterministic `
  --zero-expert

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
