# Plan: Unified Discovery & Remote Conductor Interface

## Phase 1: Robot Discovery Setup (mDNS)
- [ ] **Task: Pi 5 Avahi Configuration**
    - [ ] Create a `continuon.service` file for Avahi on the Pi 5 to advertise the `_continuon._tcp` service.
    - [ ] Verify mDNS resolution (`ping robot.local`) from the development machine.
- [ ] **Task: Python Discovery Utility**
    - [ ] Create `scripts/find_robot.py` using the `zeroconf` library.
    - [ ] Implement a caching mechanism to store the resolved IP locally to speed up subsequent CLI calls.
- [ ] **Task: Flutter Discovery Integration**
    - [ ] Add an mDNS discovery package to `continuonai/pubspec.yaml`.
    - [ ] Update the robot discovery logic in the Flutter app to use mDNS resolution.
- [ ] **Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)**

## Phase 2: Remote Transport & Security (SSH)
- [ ] **Task: SSH Key Management**
    - [ ] Automate the generation and deployment of SSH keys from the dev machine to the Pi 5.
    - [ ] Disable password-based SSH login on the Pi 5 for the `continuon` service user.
- [ ] **Task: Remote Command Runner (Fabric)**
    - [ ] Implement a `RemoteRunner` class in Python to wrap `fabric` for command execution.
    - [ ] Create a CLI shortcut to run arbitrary commands on the robot (e.g., `./cb remote run "ls"`).
- [ ] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**

## Phase 3: File Sync & Development Loop
- [ ] **Task: File Synchronization Tool**
    - [ ] Implement an `rsync`-based or `fabric`-based sync utility to push local changes to the robot.
    - [ ] Add a `watch` mode that automatically pushes file changes on save.
- [ ] **Task: Log Streaming Utility**
    - [ ] Implement a remote log tailing feature to stream `continuonbrain.log` to the local terminal.
- [ ] **Task: Port Forwarding (Tunneling)**
    - [ ] Add a utility to establish SSH tunnels for ports 8081 (Brain Studio) and 8082 (API Server).
- [ ] **Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)**

## Phase 4: App Verification & Robustness
- [ ] **Task: Discovery Refresh & Retry Logic**
    - [ ] Improve the Flutter UI to show discovery status (searching, found, error).
    - [ ] Implement background polling to keep the robot's IP updated in the app.
- [ ] **Task: Integration Test: Remote Training Start**
    - [ ] Perform an end-to-end test: Discover -> Sync Code -> Start Training -> View Logs -> Access Dashboard.
- [ ] **Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)**

## Phase 5: Final Polish & Documentation
- [ ] **Task: Setup Scripts**
    - [ ] Create a `scripts/setup_remote.sh` to automate the entire setup (Avahi, SSH, Keys) for new robots.
- [ ] **Task: Update Documentation**
    - [ ] Document the new remote CLI commands in `PI5_EDGE_BRAIN_INSTRUCTIONS.md`.
- [ ] **Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)**
