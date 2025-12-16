param(
  [string]$Distro = "Ubuntu"
)

Write-Host "Stopping studio_training_server inside WSL distro '$Distro' ..."

# Stop the python server inside WSL.
wsl -d $Distro -- bash -lc "pkill -f 'continuonbrain.studio_training_server' || true; rm -f /tmp/studio_training_server.pid || true"

# Also stop any lingering wsl.exe processes started with tail -f /dev/null.
# (This is intentionally broad but scoped to WSL; you can close all via 'wsl --shutdown' if needed.)
Write-Host "If the WSL instance is still kept alive by a background wsl.exe, you can run: wsl --shutdown"


