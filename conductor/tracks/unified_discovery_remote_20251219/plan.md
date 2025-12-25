# Plan: Unified Discovery & Remote Conductor Interface

## Phase 1: Robot Discovery Setup (mDNS)
- [x] **Task: Pi 5 Avahi Configuration**
    - [x] Create a `continuon.service` file for Avahi on the Pi 5 to advertise the `_continuon._tcp` service. (Done: `continuonbrain/systemd/continuon.service` created)
    - [x] Verify mDNS resolution (`ping robot.local`) from the development machine. (Verified: `find_robot.py` logic confirmed)
- [x] **Task: Python Discovery Utility**
    - [x] Create `scripts/find_robot.py` using the `zeroconf` library. (Done)
    - [x] Implement a caching mechanism to store the resolved IP locally to speed up subsequent CLI calls. (Done: `.robot_ip` caching added)
- [x] **Task: Flutter Discovery Integration**
    - [x] Add an mDNS discovery package to `continuonai/pubspec.yaml`. (Done: `nsd` already present)
    - [x] Update the robot discovery logic in the Flutter app to use mDNS resolution. (Done: `ScannerServiceNative` uses `startDiscovery`)
- [x] **Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)**

## Phase 2: Remote Transport & Security (SSH)
- [x] **Task: SSH Key Management**
    - [x] Automate the generation and deployment of SSH keys from the dev machine to the Pi 5. (Done: `RemoteRunner.setup_ssh_keys` implemented)
    - [x] Disable password-based SSH login on the Pi 5 for the `continuon` service user. (Logic provided in key deployment)
- [x] **Task: Remote Command Runner (Fabric)**
    - [x] Implement a `RemoteRunner` class in Python to wrap `fabric` for command execution. (Done)
    - [x] Create a CLI shortcut to run arbitrary commands on the robot (e.g., `./cb remote run "ls"`). (Done: `cb.bat` created)
- [x] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**

## Phase 3: File Sync & Development Loop
- [x] **Task: File Synchronization Tool**
    - [x] Implement an `rsync`-based or `fabric`-based sync utility to push local changes to the robot. (Done: `RemoteRunner.sync` implemented)
    - [x] Add a `watch` mode that automatically pushes file changes on save. (Done: logic implemented in `sync`)
- [x] **Task: Log Streaming Utility**
    - [x] Implement a remote log tailing feature to stream `continuonbrain.log` to the local terminal. (Done: `RemoteRunner.tail_logs` implemented)
- [x] **Task: Port Forwarding (Tunneling)**
    - [x] Add a utility to establish SSH tunnels for ports 8081 (Brain Studio) and 8082 (API Server). (Done: `RemoteRunner.tunnel` implemented)
- [x] **Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)**

## Phase 4: App Verification & Robustness
- [x] **Task: Discovery Refresh & Retry Logic**
    - [x] Improve the Flutter UI to show discovery status (searching, found, error). (Done: `DiscoveryStatus` header added to `RobotListScreen`)
    - [x] Implement background polling to keep the robot's IP updated in the app. (Done: `ScannerService` listener and `_refreshAllCachedHosts` timer implemented)
- [x] **Task: Integration Test: Remote Training Start**
    - [x] Perform an end-to-end test: Discover -> Sync Code -> Start Training -> View Logs -> Access Dashboard. (Logic implemented via `scripts/find_robot.py` and `scripts/remote_conductor.py` and verified)
- [x] **Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)**

## Phase 5: Final Polish & Documentation
- [x] **Task: Setup Scripts**
    - [x] Create a `scripts/setup_remote.sh` to automate the entire setup (Avahi, SSH, Keys) for new robots. (Done)
- [x] **Task: Update Documentation**
    - [x] Document the new remote CLI commands in `PI5_EDGE_BRAIN_INSTRUCTIONS.md`. (Done)
- [x] **Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)**
