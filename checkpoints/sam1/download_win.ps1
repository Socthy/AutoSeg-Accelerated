Param(
    [string]$OutputDir = "."
)

Set-Location -Path $PSScriptRoot
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$urls = @(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)

foreach ($url in $urls) {
    $fileName = Split-Path $url -Leaf
    $outPath = Join-Path $OutputDir $fileName
    Write-Host "Downloading $fileName ..."
    Invoke-WebRequest -Uri $url -OutFile $outPath
}

Write-Host "All SAM1 checkpoints downloaded successfully."
