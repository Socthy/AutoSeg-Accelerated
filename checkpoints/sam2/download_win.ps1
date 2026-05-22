Param(
    [string]$OutputDir = "."
)

Set-Location -Path $PSScriptRoot
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$baseUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
$urls = @(
    "${baseUrl}sam2_hiera_tiny.pt",
    "${baseUrl}sam2_hiera_small.pt",
    "${baseUrl}sam2_hiera_base_plus.pt",
    "${baseUrl}sam2_hiera_large.pt"
)

foreach ($url in $urls) {
    $fileName = Split-Path $url -Leaf
    $outPath = Join-Path $OutputDir $fileName
    Write-Host "Downloading $fileName ..."
    Invoke-WebRequest -Uri $url -OutFile $outPath
}

Write-Host "All SAM2 checkpoints downloaded successfully."
