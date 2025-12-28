# Sync Training Data from Raspberry Pi to Windows
# Usage: .\sync_data_from_pi.ps1 -PiHost 192.168.1.100 -PiUser pi

param(
    [Parameter(Mandatory=$true)]
    [string]$PiHost,
    
    [string]$PiUser = "pi",
    
    [string]$LocalDir = ".\rlds_data",
    
    [string]$RemoteDir = "/opt/continuonos/brain/rlds/episodes"
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Sync Training Data from Pi" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pi Host: $PiUser@$PiHost"
Write-Host "Remote: $RemoteDir"
Write-Host "Local: $LocalDir"
Write-Host ""

# Create local directory
if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Path $LocalDir | Out-Null
    Write-Host "Created local directory: $LocalDir"
}

# Check if scp is available
$scpPath = Get-Command scp -ErrorAction SilentlyContinue
if (-not $scpPath) {
    Write-Host "ERROR: scp not found. Install OpenSSH or use an alternative method." -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Use WinSCP or FileZilla to manually copy:" -ForegroundColor Yellow
    Write-Host "  From: $PiUser@$PiHost`:$RemoteDir"
    Write-Host "  To:   $LocalDir"
    exit 1
}

# Sync data
Write-Host ""
Write-Host "Syncing data (this may take a while)..." -ForegroundColor Yellow
Write-Host ""

# Use scp with recursive copy
$scpCommand = "scp -r ${PiUser}@${PiHost}:${RemoteDir}/* $LocalDir/"
Write-Host "Running: $scpCommand"
Write-Host ""

try {
    Invoke-Expression $scpCommand
    
    # Count files
    $fileCount = (Get-ChildItem -Path $LocalDir -Recurse -File).Count
    $totalSize = (Get-ChildItem -Path $LocalDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
    
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Sync Complete!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Files synced: $fileCount"
    Write-Host "Total size: $([math]::Round($totalSize, 2)) MB"
    Write-Host ""
    Write-Host "Now run training:" -ForegroundColor Cyan
    Write-Host "  python scripts\gpu_training\train_gpu.py --data-dir $LocalDir"
}
catch {
    Write-Host "ERROR: Sync failed. Check SSH connection to Pi." -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

