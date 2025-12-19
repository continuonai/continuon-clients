# Specification: Flutter Web Discovery & UX Polish

## Overview
Automatic mDNS discovery is technically restricted in web browsers (Chrome/Safari) due to security policies. This track focuses on improving the user experience for ContinuonAI Web users by providing clear manual connection paths, better error state communication, and UI refinements that acknowledge platform-specific limitations.

## User Goals
- **Clear Guidance:** Web users should immediately understand why a scan might not find their robot.
- **Low Friction Manual Entry:** Provide a prominent and easy-to-use manual IP/Port entry field.
- **Persistent Access:** Allow users to "Quick Connect" from anywhere in the UI without navigating back to the robot list.

## Functional Requirements
### 1. Web-Specific UI Guidance
- Detect if the app is running on Web and display a "Discovery Tip" banner.
- Explain the need for the `cors` headers or the `tunnel_robot.py` utility for web-to-pi communication.

### 2. Enhanced Manual Connection
- Add a "Manual Connect" section to the Discovery screen that is always visible.
- Implement a "Quick Connect" text field in the side navigation or top bar for power users.

### 3. Connection State Visualization
- Improve the "Scanning" animation to be more informative on Web (e.g., "Web scan limited to known hosts...").
- Provide a "Connection Diagnostics" tool that tests both gRPC and HTTP ports and reports specific errors (CORS, Timeout, etc.).

## Acceptance Criteria
- [ ] Discovery screen shows a helpful banner when running on Web.
- [ ] Manual IP entry is the primary call-to-action on Web.
- [ ] A "Quick Connect" bar is available in the main layout.
- [ ] The app provides a link to the `tunnel_robot.py` documentation for Web users.
