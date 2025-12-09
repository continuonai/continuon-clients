# ContinuonBrain Debian Package

This directory contains the Debian package structure for ContinuonBrain.

## Building the Package

```bash
./build-package.sh
```

This will create `build/continuonbrain_1.0.0_arm64.deb`

## Package Structure

```
packaging/continuonbrain/
├── DEBIAN/
│   ├── control              # Package metadata
│   ├── postinst            # Post-installation script
│   ├── prerm               # Pre-removal script
│   └── postrm              # Post-removal script
├── opt/continuonbrain/
│   ├── app/                # Application code (copied during build)
│   └── config/             # Default configuration
├── etc/
│   ├── systemd/system/
│   │   ├── continuonbrain.service
│   │   └── continuonbrain-watchdog.service
│   └── continuonbrain/
│       └── config.yaml     # System configuration
└── usr/bin/
    ├── continuonbrain      # CLI wrapper
    └── continuonbrain-dev-mode
```

## Installation

```bash
sudo apt install ./build/continuonbrain_1.0.0_arm64.deb
```

## Post-Installation

The package will:
1. Create a `continuon` system user
2. Set up Python virtual environment in `/opt/continuonbrain/venv`
3. Install all Python dependencies
4. Enable and start systemd services
5. Configure kiosk mode (auto-login)

## Usage

```bash
# Start service
continuonbrain start

# View logs
continuonbrain logs

# Enable developer mode
continuonbrain dev-mode enable

# Check status
continuonbrain status
```

## Web UI

After installation, access the web UI at:
- http://localhost:8080/ui (local)
- http://<robot-ip>:8080/ui (network)

## Developer Mode

Press `Ctrl+Alt+F2` to access terminal, or run:
```bash
continuonbrain dev-mode enable
```

## Uninstallation

```bash
sudo apt remove continuonbrain
```

User data will be backed up to `/tmp/continuonbrain-backup-*.tar.gz`
