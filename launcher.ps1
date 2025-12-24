
$ErrorActionPreference = "Stop"

# $PythonExe  = "E:/anaconda3/envs/sumo/python.exe"
# $ScriptPath = "E:\repos\LLM_traffic_query\tests\crazy_traci_agent.py"
# $SumoCfg    = "E:\Sumo\sumo_maps\simple_4leg_intersection\simple_single_intersection.sumocfg"

# & $PythonExe $ScriptPath `
#   -c $SumoCfg `
#   --gui `
#   --max-time 300 `
#   --decision-interval 1 `
#   --seed 123 `
#   --delay-ms 200

# launcher.ps1

$PythonExe  = "E:/anaconda3/envs/sumo/python.exe"
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\run_dqn_tsc.py"
$SumoCfg    = "E:\Sumo\sumo_maps\simple_4leg_intersection\simple_single_intersection.sumocfg"

& $PythonExe $ScriptPath `
  -c $SumoCfg `
  --gui `
  --max-time 3000 `
  --seed 123 `
  --delay-ms 50 `
  --hold 5.0 `
  --epsilon 0.1 `
  --device "cuda" `
  --gamma 0.99 `
  --lr 0.001 `
  --buffer 50000 `
  --train-start 2000 `
  --batch 8 `
  --target-update 1000

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
