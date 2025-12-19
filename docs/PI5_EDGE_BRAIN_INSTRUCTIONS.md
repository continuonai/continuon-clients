# Pi 5 Edge Brain: Remote Development Guide

This guide explains how to set up, discover, and remotely manage a Raspberry Pi 5 running ContinuonBrain.

## 1. Initial Setup

### On the Pi 5
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ContinuonXR/ContinuonXR.git
    cd ContinuonXR
    ```
2.  **Run the Setup Script:**
    This script installs Avahi (mDNS), configures the `_continuon._tcp` service, and prepares the user groups.
    ```bash
    bash scripts/setup_remote.sh
    ```
    *   **Note:** Reboot the Pi after this script to ensure all group permissions apply.

## 2. Discovery

From your development machine (laptop/desktop):

### Automatic Discovery (CLI)
Use the discovery utility to find robots on your local network.
```bash
python scripts/find_robot.py
```
*   **Output:** `✅ Found robot: continuon-pi.local at 192.168.1.X:8080`
*   **Caching:** The IP is cached in `.continuon_config/discovery_cache.json` for subsequent commands.

### Mobile App Discovery
Open the **ContinuonAI** Flutter app.
1.  Navigate to the **Discovery Screen**.
2.  The app will automatically scan for robots advertising `_continuon._tcp`.
3.  Tap a robot to pair or connect.

## 3. Remote Management (SSH)

### Setup SSH Keys
Avoid typing passwords by deploying an SSH key.
```bash
python scripts/setup_ssh.py <robot-hostname-or-ip> --user pi
```
*   Follow the prompts to generate/deploy keys.
*   **Optional:** Disable password authentication for security.

### Running Commands
Execute commands on the robot without logging in interactively.
```bash
python scripts/remote_runner.py "ls -la /home/pi"
```
*   The script uses the cached IP from the discovery step if `--host` is omitted.

## 4. Development Loop

### Sync Code
Push your local `continuonbrain/` changes to the robot.
```bash
python scripts/sync_robot.py
```

### Watch Mode (Auto-Sync)
Automatically push changes whenever you save a file.
```bash
python scripts/sync_robot.py --watch
```

### Stream Logs
View the live logs from the robot's API server.
```bash
python scripts/log_stream.py
```

### Port Forwarding (Tunneling)
Access the Brain Studio (Web UI) securely from your local browser.
```bash
python scripts/tunnel_robot.py
```
*   Then open: `http://localhost:8080/ui`

## 5. Troubleshooting

*   **Discovery fails:** Ensure both devices are on the same Wi-Fi. Check if `avahi-daemon` is running on the Pi (`systemctl status avahi-daemon`).
*   **SSH permission denied:** Run `python scripts/setup_ssh.py` again or check `~/.ssh/authorized_keys` on the Pi.
*   **Sync fails:** Ensure `rsync` is installed on both machines (use Git Bash on Windows).
