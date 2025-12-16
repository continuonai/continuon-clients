param(
  [string]$Distro = "Ubuntu",
  [int]$Port = 8081
)

Write-Host "Checking http://localhost:$Port/api/status ..."
try {
  $resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://localhost:$Port/api/status"
  Write-Host "OK ($($resp.StatusCode))"
  Write-Host $resp.Content
} catch {
  Write-Host "Not reachable on localhost."
}

Write-Host ""
Write-Host "Recent WSL log:"
wsl -d $Distro -- bash -lc "tail -n 80 /tmp/studio_training_server.log 2>/dev/null || echo 'no log yet'"


