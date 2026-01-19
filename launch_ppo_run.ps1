# launcher_ppo_gated.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$WorkDir    = "E:\repos\LLM_traffic_query\tests\sumo_traci"
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\run_ppo_tsc.py"
$SumoCfg    = "E:\Sumo\sumo_maps\simple_5leg_intersection\simple_single_intersection.sumocfg"

# Match VS Code env
$env:SUMO_HOME = "E:\Sumo"

# Match VS Code cwd
Set-Location $WorkDir

& $PythonExe $ScriptPath `
  -c $SumoCfg `
  --max-time 600000 `
  --episode-len 3600 `
  --warmup 100 `
  --episodes 300 `
  --seed 519 `
  --sumo-seed 514 `
  --delay-ms 0 `
  --hold 5.0 `
  --device cuda `
  --gamma 0.98 `
  --hidden-dim 256 `
  --n-layer 6 `
  --actor-lr 0.00005 `
  --critic-lr 0.0001 `
  --traffic-scale-mean 0.65 `
  --traffic-scale-std 0.05 `
  --tb-logdir "E:\repos\LLM_traffic_query\tests\sumo_traci\tensorboard_logs" `
  --save-dir "E:\repos\LLM_traffic_query\tests\sumo_traci\ppo_tsc_models" `
  --rollout-steps 1024 `
  --ppo-epochs 4 `
  --minibatch 128 `
  --clip-eps 0.2 `
  --vf-clip-eps 0.2 `
  --gae-lambda 0.90 `
  --ent-coef 0.01 `
  --vf-coef 1.0 `
  --thr-ref 2.00 `
  --queue-ref 1.0 `
  --w-thr 1.0 `
  --w-queue 1.0 `
  --queue-power 1.0 `
  --top2-w1 0.7 `
  --top2-w2 0.3 `
  --reward-clip-lo -3.0 `
  --reward-clip-hi 3.0 `
#   --use-expert-features

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }