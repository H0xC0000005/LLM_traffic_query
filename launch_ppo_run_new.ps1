# launcher_run_ppo_tsc_reward_encoder.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$WorkDir    = "E:\repos\LLM_traffic_query\tests\sumo_traci"
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\run_ppo_tsc.py"

$env:SUMO_HOME = "E:\Sumo"
Set-Location $WorkDir

# -----------------------------
# Select reward here
# -----------------------------
$RewardName = "pressure"   # "universal_v2" | "queue" | "pressure" | "unbiased_simple_v1"

# -----------------------------
# Select encoder composition here
# -----------------------------
# "adlight_state" "frap_state"
$CoreEncoderName  = "bounded_v2"   # "bounded_v2" | "pressure_state" | "ats"
$AddonEncoderName = "pressure_state"         # "none" | "expert" | "pressure_state" | "ats"

# -----------------------------
# Common PPO / scenario args
# -----------------------------
$RunArgs = @(
    # "E:\Sumo\sumo_maps\4leg_22+22+\4leg_22+22+.sumocfg", "E:\Sumo\sumo_maps\4leg_3LR23LR2\4leg_3LR23LR2.sumocfg",
    # "E:\Sumo\sumo_maps\4leg_3232skewed\4leg_3232skewed.sumocfg",
    # "E:\Sumo\sumo_maps\4leg_2L3S23O\4leg_2L3S23O.sumocfg",
  "-c", "E:\Sumo\sumo_maps\4leg_2L3S23O\4leg_2L3S23O.sumocfg",
  "--max-time", "6000000",
  "--episode-len", "7200",
  "--warmup", "100",
  "--episodes", "600",
  "--seed", "172",
  "--sumo-seed", "514",
  "--delay-ms", "0",
  "--hold", "5.0",
  "--device", "cuda",
  "--gamma", "0.99",
  "--hidden-dim", "256",
  "--n-layer", "6",
  "--actor-lr", "0.00005",
  "--critic-lr", "0.0001",
  "--traffic-scale-mean", "1.0",
  "--traffic-scale-std", "0.05",
  # "E:\repos\LLM_traffic_query\tests\sumo_traci\tensorboard_logs_all\tsb_3232skewed_2",
  # "E:\repos\LLM_traffic_query\tests\sumo_traci\tensorboard_logs_all\tsb_3LR23LR2_5",
  "--tb-logdir", "E:\repos\LLM_traffic_query\tests\sumo_traci\tensorboard_logs_all\tsb_2L3S23O_1",
  # "E:\repos\LLM_traffic_query\tests\sumo_traci\models_all\models_3232skewed_2",
  # "E:\repos\LLM_traffic_query\tests\sumo_traci\models_all\models_3LR23LR2\5",
  "--save-dir", "E:\repos\LLM_traffic_query\tests\sumo_traci\models_all\models_2L3S23O_1",
  "--rollout-steps", "2048",
  "--ppo-epochs", "5",
  "--minibatch", "256",
  "--clip-eps", "0.2",
  "--vf-clip-eps", "0.2",
  "--gae-lambda", "0.95",
  "--ent-coef", "0.04",
  "--ent-coef-end", "0.005",
  "--ent-coef-decay-updates", "400",
  "--vf-coef", "1.0",
  "--explore-alpha-start", "0.02",
  "--explore-alpha-end", "0.005",
  "--explore-alpha-decay-updates", "400",
  "--target-kl", "0.04",
  "--adv-clip", "3.0",
  "--reward-name", $RewardName,
  "--core-encoder-name", $CoreEncoderName,
  "--addon-encoder-name", $AddonEncoderName
)

# -----------------------------
# Reward-specific args
# -----------------------------
$RewardArgs = @()
$RewardTag  = ""

switch ($RewardName) {
  "universal_v2" {
    $RewardArgs += @(
      "--thr-ref", "2.00",
      "--queue-ref", "1.0",
      "--w-thr", "0.90",
      "--w-queue", "0.70",
      "--w-delta-queue", "1.5",
      "--w-wait", "1.3",
      "--w-queue-zone", "0.5",
      "--wait-ref", "60",
      "--wait-barrier-start", "20",
      "--softmax-wait-beta", "5",
      "--softmax-queue-beta", "3.0",
      "--queue-power", "1.0",
      "--reward-clip-lo", "-5.0",
      "--reward-clip-hi", "5.0"
    )
    $RewardTag = "reward_universal_v2"
  }

  "queue" {
    $RewardArgs += @(
      "--queue-ref", "1.0",
      "--queue-power", "1.0",
      "--softmax-queue-beta", "3.0"
    )
    $RewardTag = "reward_queue"
  }

  "pressure" {
    $RewardArgs += @(
      # enable these if your parser supports them
      # "--pressure-upstream-key", "count_ratio_norm",
      # "--pressure-aggregate", "presslight"
    )
    $RewardTag = "reward_pressure"
  }

  "unbiased_simple_v1" {
    $RewardArgs += @(
      "--thr-ref", "2.00",
      "--queue-ref", "1.0",
      "--w-thr", "0.90",
      "--w-queue", "0.70",
      "--w-wait", "1.0",
      "--wait-ref", "40",
      "--wait-barrier-start", "10",
      "--softmax-wait-beta", "5",
      "--softmax-queue-beta", "3.0",
      "--queue-power", "1.0",
      "--reward-clip-lo", "-5.0",
      "--reward-clip-hi", "5.0"
    )
    $RewardTag = "reward_unbiased_simple_v1"
  }

  default {
    throw "Unsupported reward name: $RewardName"
  }
}

# -----------------------------
# Encoder tag
# -----------------------------
$EncoderTag = "enc_${CoreEncoderName}"
if ($AddonEncoderName -ne "none") {
  $EncoderTag = "${EncoderTag}+${AddonEncoderName}"
}

# -----------------------------
# Log tag
# -----------------------------
$LogTag = "base_ploop+_ub_${EncoderTag}_${RewardTag}"

$RunArgs += $RewardArgs
$RunArgs += @("--log-tag", $LogTag)

Write-Host "Running reward:" $RewardName
Write-Host "Running encoder core/addon:" $CoreEncoderName "/" $AddonEncoderName
Write-Host "Command:" $PythonExe $ScriptPath ($RunArgs -join " ")

& $PythonExe $ScriptPath @RunArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }