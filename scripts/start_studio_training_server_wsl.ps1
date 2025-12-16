param(
  [string]$Distro = "Ubuntu",
  [int]$Port = 8081,
  [string]$RuntimeRoot = "/opt/continuonos/brain",
  [string]$ConfigDir = "/tmp/continuonbrain_demo",
  [string]$RepoPathWsl = "/mnt/c/Users/CraigM/source/repos/ContinuonXR",
  [string]$LogPath = "/tmp/studio_training_server.log",
  [string]$LauncherStdoutWin = "$PSScriptRoot\\..\\.tmp\\wsl_studio_training_launcher.out.log",
  [string]$LauncherStderrWin = "$PSScriptRoot\\..\\.tmp\\wsl_studio_training_launcher.err.log"
)

$wslArgs = @(
  "-d", $Distro, "--", "bash", "$RepoPathWsl/scripts/wsl/studio_training_server.sh",
  "$Port", "$RuntimeRoot", "$ConfigDir", "$RepoPathWsl", "$LogPath"
)

Write-Host "Starting Studio training server in WSL distro '$Distro' on port $Port ..."
Write-Host "This will spawn a background wsl.exe process that keeps WSL alive."

New-Item -ItemType Directory -Force (Split-Path -Parent $LauncherStdoutWin) | Out-Null
$p = Start-Process -FilePath "wsl.exe" -ArgumentList $wslArgs -WindowStyle Minimized -PassThru `
  -RedirectStandardOutput $LauncherStdoutWin -RedirectStandardError $LauncherStderrWin
Write-Host "Launcher PID: $($p.Id)"
Start-Sleep -Seconds 2

Write-Host "Attempting to reach http://localhost:$Port/api/status ..."
try {
  $resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://localhost:$Port/api/status"
  Write-Host "OK ($($resp.StatusCode))"
} catch {
  Write-Host "Not reachable yet. Check WSL log: wsl -d $Distro -- tail -n 200 $LogPath"
}


