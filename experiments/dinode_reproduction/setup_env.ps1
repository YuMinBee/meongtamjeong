$ErrorActionPreference = "Stop"

$EnvName = "dinode-repro"
$Python = "3.10"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH."
}

$Existing = conda env list --json | ConvertFrom-Json
$TargetEnvPath = $Existing.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
if (-not $TargetEnvPath) {
    conda create -n $EnvName "python=$Python" -y
    $Existing = conda env list --json | ConvertFrom-Json
    $TargetEnvPath = $Existing.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
} else {
    Write-Host "Conda environment '$EnvName' already exists; completing/validating its packages."
}

$EnvPython = Join-Path $TargetEnvPath "python.exe"
if (-not (Test-Path -LiteralPath $EnvPython)) {
    throw "Python executable not found in Conda environment: $EnvPython"
}

& $EnvPython -m pip install `
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 `
    --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed with exit code $LASTEXITCODE." }
& $EnvPython -m pip install -r `
    "$PSScriptRoot\vendor\DINOde\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "DINOde dependency installation failed with exit code $LASTEXITCODE." }

& $EnvPython -c "import torch, torchvision, transformers; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('transformers', transformers.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if ($LASTEXITCODE -ne 0) { throw "Environment validation failed with exit code $LASTEXITCODE." }
