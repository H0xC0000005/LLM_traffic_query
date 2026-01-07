# launcher.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\run_ppo_tsc.py"
$SumoCfg    = "E:\Sumo\sumo_maps\simple_5leg_intersection\simple_single_intersection.sumocfg"

# Match VS Code env
$env:SUMO_HOME = "E:\Sumo"

& $PythonExe $ScriptPath `
  -c $SumoCfg `
  --max-time 600000 `
  --episode-len 3600 `
  --warmup 150 `
  --episodes 300 `
  --seed 121 `
  --sumo-seed 514 `
  --delay-ms 0 `
  --hold 5.0 `
  --device cuda `
  --gamma 0.99 `
  --hidden-dim 256 `
  --n-layer 2 `
  --lr 0.0001 `
  --traffic-scale 0.60 `
  --tb-logdir "E:\repos\LLM_traffic_query\tests\sumo_traci\tensorboard_logs" `
  --save-dir "E:\repos\LLM_traffic_query\tests\sumo_traci\ppo_tsc_models" `
  --rollout-steps 1024 `
  --ppo-epochs 2 `
  --minibatch 128 `
  --clip-eps 0.2 `
  --gae-lambda 0.95 `
  --ent-coef 0.01 `
  --vf-coef 0.5 `
  --thr-ref 1.50 `
  --queue-ref 1.0 `
  --w-thr 1.0 `
  --w-queue 1.0 `
  --queue-power 1.0 `
  --top2-w1 0.7 `
  --top2-w2 0.3 `
  --reward-clip-lo -3.0 `
  --reward-clip-hi 3.0

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
