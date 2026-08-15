$ErrorActionPreference = "Stop"

$EnvName = "dinode-repro"
$Upstream = Join-Path $PSScriptRoot "vendor\DINOde"
$DataDir = Join-Path $Upstream "data"
$Archive = Join-Path $DataDir "VOCtrainval_11-May-2012.tar"
$Checkpoint = Join-Path $Upstream "checkpoints\eccv26_dinode_coco_stuff.pth"
$ExpectedVocMd5 = "6cd6e144f989b92b3379bac3b3de84fd"

$Existing = conda env list --json | ConvertFrom-Json
$TargetEnvPath = $Existing.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
$EnvPython = Join-Path $TargetEnvPath "python.exe"
if (-not (Test-Path -LiteralPath $EnvPython)) {
    throw "Run setup_env.ps1 first; '$EnvName' is not ready."
}

if (-not (Test-Path -LiteralPath $Upstream)) {
    throw "Missing upstream checkout at $Upstream"
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Checkpoint) | Out-Null

if (-not (Test-Path -LiteralPath $Archive)) {
    $ArchivePart = "$Archive.part"
    $Sources = @(
        "https://huggingface.co/datasets/DerrickUnleashed/Pascal_VOC/resolve/main/VOCtrainval_11-May-2012.tar?download=true",
        "http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar"
    )
    foreach ($Source in $Sources) {
        Write-Host "Downloading VOC 2012 from $Source"
        curl.exe -L --fail --retry 2 --continue-at - $Source -o $ArchivePart
        if ($LASTEXITCODE -eq 0) {
            Move-Item -LiteralPath $ArchivePart -Destination $Archive
            break
        }
        Write-Warning "VOC mirror failed; trying the next source."
    }
    if (-not (Test-Path -LiteralPath $Archive)) {
        throw "All configured VOC 2012 mirrors failed."
    }
}

$ActualVocMd5 = (Get-FileHash -LiteralPath $Archive -Algorithm MD5).Hash.ToLowerInvariant()
if ($ActualVocMd5 -ne $ExpectedVocMd5) {
    throw "VOC archive checksum mismatch: expected $ExpectedVocMd5, got $ActualVocMd5"
}

if (-not (Test-Path -LiteralPath (Join-Path $DataDir "VOCdevkit\VOC2012"))) {
    tar -xf $Archive -C $DataDir
}

if (-not (Test-Path -LiteralPath $Checkpoint)) {
    & $EnvPython -m pip install gdown==5.2.0
    & $EnvPython -m gdown `
        "https://drive.google.com/uc?id=1W0aw1JAY8oqN5NaZCYkdCF1BvETvYo_T" `
        -O $Checkpoint
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint download failed with exit code $LASTEXITCODE." }
}

Push-Location $Upstream
try {
    & $EnvPython processing/pascal_voc/pascal_voc_processor.py `
        --root_dir ./data `
        --output_dir ./data/voc_processed
    if ($LASTEXITCODE -ne 0) { throw "VOC preprocessing failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}

Write-Host "VOC20 data and official checkpoint are ready."
