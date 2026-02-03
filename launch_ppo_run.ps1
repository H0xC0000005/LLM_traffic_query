# launcher_run_ppo_tsc_gated.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$WorkDir    = "E:\repos\LLM_traffic_query\tests\sumo_traci"   # matches VS Code cwd
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\run_ppo_tsc.py"

# Match VS Code env
$env:SUMO_HOME = "E:\Sumo"

# Match VS Code cwd
Set-Location $WorkDir

$RunArgs = @(
  "-c", "E:\Sumo\sumo_maps\4leg_3LR23LR2\4leg_3LR23LR2.sumocfg",
  "--max-time", "600000",
  "--episode-len", "7200",
  "--warmup", "100",
  "--episodes", "200",
  "--seed", "114",
  "--sumo-seed", "514",
  "--delay-ms", "0",
  "--hold", "5.0",
  "--device", "cuda",
  "--gamma", "0.99",
  "--hidden-dim", "256",
  "--n-layer", "6",
  "--actor-lr", "0.00005",
  "--critic-lr", "0.0001",
  "--traffic-scale-mean", "01.0",
  "--traffic-scale-std", "0.04",
  # logdir and save dir suffix should be consistent with sumocfg name: tsb_... and models_...
  # e.g., 4leg_3232skewed.sumocfg  -->  tsb_3232skewed  and  models_3232skewed
  "--tb-logdir", "E:\repos\LLM_traffic_query\tests\sumo_traci\tsb_3LR23LR2\",
  "--save-dir", "E:\repos\LLM_traffic_query\tests\sumo_traci\models_3LR23LR2\",
  "--rollout-steps", "2048",
  "--ppo-epochs", "3",
  "--minibatch", "256",
  "--clip-eps", "0.2",
  "--vf-clip-eps", "0.2",
  "--gae-lambda", "0.95",
  "--ent-coef", "0.04",
  "--ent-coef-end", "0.005",
  "--ent-coef-decay-updates", "200",
  "--vf-coef", "1.0",
  "--explore-alpha-start", "0.02",
  "--explore-alpha-end", "0.001",
  "--explore-alpha-decay-updates", "200",
  "--target-kl", "0.04",
  "--adv-clip", "3.0",
  "--thr-ref", "2.00",
  "--queue-ref", "1.0",
  "--w-thr", "01.0",
  "--w-queue", "01.0",
  "--w-delta-queue", "03.0",
  "--w-wait", "01.0",
  "--wait-ref", "60",
  "--wait-barrier-start", "20",
  "--softmax-wait-beta", "5",
  "--softmax-queue-beta", "3.0",
  "--queue-power", "1.0",
  "--reward-clip-lo", "-5.0",
  "--reward-clip-hi", "5.0",
  "--log-tag", "highdq_514",
  "--use-expert-features",
)

& $PythonExe $ScriptPath @RunArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }