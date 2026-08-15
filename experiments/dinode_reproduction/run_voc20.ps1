param(
    [int]$MaxSamples = 0,
    [string]$OutputName = "eval_voc20_no_cache"
)

$ErrorActionPreference = "Stop"

$EnvName = "dinode-repro"
$Upstream = Join-Path $PSScriptRoot "vendor\DINOde"
$Existing = conda env list --json | ConvertFrom-Json
$TargetEnvPath = $Existing.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
$EnvPython = Join-Path $TargetEnvPath "python.exe"
if (-not (Test-Path -LiteralPath $EnvPython)) {
    throw "Run setup_env.ps1 first; '$EnvName' is not ready."
}

# The upstream collate function is local to main(), which works with Linux
# fork workers but cannot be pickled by Windows spawn workers. Setting workers
# to zero changes only data loading, not model computation or metrics.
$RuntimeDir = Join-Path $PSScriptRoot "runtime"
$WindowsConfig = Join-Path $RuntimeDir "dinode_eval_windows.json"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$Config = Get-Content -LiteralPath (Join-Path $Upstream "configs\dinode_eval.json") -Raw | ConvertFrom-Json
$Config.data.num_workers = 0
$ConfigJson = $Config | ConvertTo-Json -Depth 20
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($WindowsConfig, $ConfigJson, $Utf8NoBom)

Push-Location $Upstream
try {
    $EvalArgs = @(
        "eval.py",
        "--config", $WindowsConfig,
        "--checkpoint", "checkpoints/eccv26_dinode_coco_stuff.pth",
        "--output_dir", "outputs/$OutputName",
        "--val_dataset", "voc20",
        "--pascal_voc_data_dir", "./data/voc_processed",
        "--no_cache",
        "--seed", "42"
    )
    if ($MaxSamples -gt 0) {
        $EvalArgs += @("--max_samples", $MaxSamples.ToString())
    }
    & $EnvPython @EvalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "DINOde evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
