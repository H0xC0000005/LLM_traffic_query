param(
    [string]$Config = "tsc",
    [string[]]$Select = @(),
    [string]$LocalConfig = "",
    [string[]]$Variable = @(),
    [int]$Jobs = 1,
    [string]$OutputRoot = "",
    [string]$PythonExe = "",
    [switch]$Run,
    [switch]$Plan,
    [switch]$List
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = if ($env:EXPERIMENT_LAUNCHER_PYTHON) { $env:EXPERIMENT_LAUNCHER_PYTHON } else { "python" }
}
if (($Run -and $Plan) -or ($Run -and $List) -or ($Plan -and $List)) {
    throw "Choose only one of -Run, -Plan, or -List. The default is a plan."
}

$ConfigPath = if (Test-Path -LiteralPath $Config -PathType Leaf) {
    (Resolve-Path -LiteralPath $Config).Path
} else {
    Join-Path $PSScriptRoot "launchers\configs\$Config.yaml"
}
$LauncherArgs = @((Join-Path $PSScriptRoot "launchers\run.py"), "--config", $ConfigPath, "--jobs", "$Jobs")
foreach ($Selection in $Select) { $LauncherArgs += @("--select", $Selection) }
foreach ($Assignment in $Variable) { $LauncherArgs += @("--var", $Assignment) }
if ([string]::IsNullOrWhiteSpace($LocalConfig)) {
    $DefaultLocal = Join-Path $PSScriptRoot "launchers\configs\local.yaml"
    if (Test-Path -LiteralPath $DefaultLocal -PathType Leaf) { $LocalConfig = $DefaultLocal }
}
if (-not [string]::IsNullOrWhiteSpace($LocalConfig)) { $LauncherArgs += @("--local", $LocalConfig) }
if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) { $LauncherArgs += @("--output-root", $OutputRoot) }
if ($Run) { $LauncherArgs += "--run" }
elseif ($List) { $LauncherArgs += "--list" }
else { $LauncherArgs += "--plan" }

& $PythonExe @LauncherArgs
exit $LASTEXITCODE
