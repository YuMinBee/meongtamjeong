[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string]$OutputDirectory,

    [string]$SofficePath = (
        Join-Path $env:TEMP (
            'codex-libreoffice-26.2.5\portable\' +
            'LibreOffice_26.2.5_Win_x86-64\SourceDir\' +
            'LibreOffice\program\soffice.com'
        )
    )
)

$ErrorActionPreference = 'Stop'

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
if ([IO.Path]::GetExtension($resolvedInput) -ne '.docx') {
    throw 'InputPath must point to a DOCX file.'
}
if (-not (Test-Path -LiteralPath $SofficePath -PathType Leaf)) {
    throw "Portable LibreOffice renderer not found: $SofficePath"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $resolvedOutput = Split-Path -Parent $resolvedInput
} else {
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $resolvedOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path
}

$profile = Join-Path $env:TEMP (
    'meongtamjeong-lo-profile-' + [guid]::NewGuid().ToString('N')
)
New-Item -ItemType Directory -Path $profile | Out-Null
$profileUri = 'file:///' + ($profile -replace '\\', '/')

& $SofficePath `
    "-env:UserInstallation=$profileUri" `
    --headless `
    --convert-to 'pdf:writer_pdf_Export' `
    --outdir $resolvedOutput `
    $resolvedInput

if ($LASTEXITCODE -ne 0) {
    throw "LibreOffice conversion failed with exit code $LASTEXITCODE."
}

$pdfName = [IO.Path]::GetFileNameWithoutExtension($resolvedInput) + '.pdf'
$pdfPath = Join-Path $resolvedOutput $pdfName
if (-not (Test-Path -LiteralPath $pdfPath -PathType Leaf)) {
    throw "LibreOffice did not create the expected PDF: $pdfPath"
}

Write-Output $pdfPath
