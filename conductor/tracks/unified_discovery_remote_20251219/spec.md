# Specification: Unified Discovery & Remote Conductor Interface

## Overview
This track addresses the critical need for reliable device discovery and remote management between the development environment (Gemini CLI / Conductor), the user interface (ContinuonAI App), and the Raspberry Pi 5 ("robot"). We will implement a unified mDNS-based discovery layer and an SSH-backed remote interface to enable seamless testing and training on the edge.

## User Goals
- **Automatic Discovery:** Eliminate manual IP entry. The app and CLI should automatically find the "robot" on the local network.
- **Remote Orchestration:** Allow the Gemini CLI to push code, run training jobs, and pull results from the Pi 5 as if it were a local environment.
- **Standardized Communication:** Use industry-standard protocols (mDNS, SSH) for maximum compatibility and security.

## Functional Requirements
### 1. Unified Discovery Layer (mDNS/Avahi)
- **Robot Broadcaster:** Configure the Pi 5 to advertise `_continuon._tcp` and `_ssh._tcp` services using Avahi.
- **Flutter Client (App):** Integrate an mDNS discovery package (e.g., `nsd`) into the ContinuonAI app to resolve "robot.local".
- **Python Client (CLI):** Implement a discovery utility in `scripts/` using `zeroconf` to find and cache the robot's IP.

### 2. Remote Conductor Interface (SSH-backed)
- **Command Runner:** A Python-based wrapper (using `fabric` or `paramiko`) to execute commands on the Pi 5 remotely.
- **FileSync Utility:** A tool to synchronize the local `continuonbrain/` directory with the `/home/pi/continuonbrain/` directory on the robot.
- **Port Forwarding:** A mechanism to tunnel the Brain Studio (port 8081/8082) from the Pi 5 to the local dev machine.
- **Log Streamer:** Capability to `tail -f` remote logs directly into the Conductor session.

### 3. App Bug Fix (Discovery)
- **Address Selection:** Ensure the Flutter app uses the resolved mDNS hostname rather than a hardcoded or cached IP.
- **Connection Retry Logic:** Implement robust retry/refresh logic in the app's robot discovery screen.

## Non-Functional Requirements
- **Security:** Remote access MUST use SSH keys (no password-based auth for the CLI).
- **Fallback Ready:** The discovery code should be architected to support UDP beacons or a central registry in the future.
- **Low Overhead:** The discovery service on the Pi 5 must consume negligible CPU/RAM.

## Acceptance Criteria
- [ ] Running a discovery script on the dev machine returns the Pi 5's current local IP.
- [ ] The ContinuonAI Flutter app successfully lists "robot" on the discovery screen without manual input.
- [ ] A file modified locally can be synced to the Pi 5 with a single command.
- [ ] A training job can be started on the Pi 5 and its output viewed live in the Gemini CLI.
- [ ] Brain Studio on the Pi 5 is accessible via `localhost` on the dev machine through a tunnel.

## Out of Scope
- Global/Cloud-based discovery (Discovery beyond the local subnet).
- Implementing the "Reverse Agent" (Option C) fully in this track.
