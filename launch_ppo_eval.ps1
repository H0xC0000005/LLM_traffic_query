# launcher_eval_tsc.ps1

$PythonExe  = "E:\anaconda3\envs\sumo\python.exe"
$WorkDir    = "E:\repos\LLM_traffic_query"
$ScriptPath = "E:\repos\LLM_traffic_query\tests\sumo_traci\eval_ppo_tsc.py"

$env:SUMO_HOME = "E:\Sumo"
Set-Location $WorkDir

# --------------------------------------------------
# Select what to run
# --------------------------------------------------
$ControllerName = "ppo"            # "ppo" | "fully_actuated" | "max_pressure"
$ModelKey       = "base1"           # only used when ControllerName = "ppo"

# --------------------------------------------------
# PPO model presets
# Keep switching models easy by editing ModelKey only
# --------------------------------------------------
$ModelTable = @{
    "base1" = @{
        Checkpoint = "E:\repos\LLM_traffic_query\tests\sumo_traci\models_all\models_3LR23LR2\1\sumo_ppo_seed172_base1_600ep_2048roll_5step_1771590042__J1.pt"
        LogTag     = "base1_comparison"
    }
    "exp1" = @{
        Checkpoint = "E:\repos\LLM_traffic_query\tests\sumo_traci\models_all\models_22+22+_2\sumo_ppo_seed172_exp1_600ep_1_1771926979__J1.pt"
        LogTag     = "zeroblk_1_4"
    }
}

# --------------------------------------------------
# Scenario must be explicit for non-PPO controllers
# It may also override PPO meta if you want
# --------------------------------------------------
$Scenario = @{
    Sumocfg = "E:\Sumo\sumo_maps\4leg_3LR23LR2\4leg_3LR23LR2.sumocfg"
    TlsId   = "J1"
}

# --------------------------------------------------
# Common eval args
# --------------------------------------------------
$EvalArgs = @(
    "--controller-name", $ControllerName,
    "--log-dir", ".\tests\sumo_traci\eval_results\3LR23LR2\rule_based",
    "--episodes", "200",
    "--episode-len", "3600",
    "--sumo-seed", "10086",
    "--deterministic"
)

# Optional PPO-only ablation knobs
$PpoExtraArgs = @(
    # "--zero-expert-dims", "33,34,35,36"
    # "--noise-expert-dims", "33,34"
    # "--noise-sigma", "0.15"
)

# --------------------------------------------------
# Controller-specific argument packing
# --------------------------------------------------
switch ($ControllerName) {
    "ppo" {
        if (-not $ModelTable.ContainsKey($ModelKey)) {
            throw "Unknown ModelKey '$ModelKey'. Available keys: $($ModelTable.Keys -join ', ')"
        }

        $Checkpoint = $ModelTable[$ModelKey].Checkpoint
        $LogTag     = $ModelTable[$ModelKey].LogTag

        $EvalArgs += @(
            "--checkpoint", $Checkpoint,
            "--log-tag", $LogTag
        )

        # Optional: expose scenario override for PPO too.
        # If omitted, evaluator will use the checkpoint-side meta json.
        if ($Scenario.Sumocfg) { $EvalArgs += @("--sumocfg", $Scenario.Sumocfg) }
        if ($Scenario.TlsId)   { $EvalArgs += @("--tls-id",  $Scenario.TlsId)   }

        $EvalArgs += $PpoExtraArgs
    }

    "fully_actuated" {
        if (-not $Scenario.Sumocfg) { throw "Scenario.Sumocfg is required for fully_actuated" }
        if (-not $Scenario.TlsId)   { throw "Scenario.TlsId is required for fully_actuated" }

        $EvalArgs += @(
            "--sumocfg", $Scenario.Sumocfg,
            "--tls-id",  $Scenario.TlsId,
            "--log-tag", "fully_actuated"
        )
    }

    "max_pressure" {
        if (-not $Scenario.Sumocfg) { throw "Scenario.Sumocfg is required for max_pressure" }
        if (-not $Scenario.TlsId)   { throw "Scenario.TlsId is required for max_pressure" }

        $EvalArgs += @(
            "--sumocfg", $Scenario.Sumocfg,
            "--tls-id",  $Scenario.TlsId,
            "--log-tag", "max_pressure"
        )
    }

    default {
        throw "Unsupported ControllerName '$ControllerName'"
    }
}

Write-Host "Running:" $PythonExe $ScriptPath ($EvalArgs -join " ")
& $PythonExe $ScriptPath @EvalArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }