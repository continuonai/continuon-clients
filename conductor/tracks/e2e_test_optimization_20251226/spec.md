# Track Specification: End-to-End Test & Entry Point Optimization

## 1. Overview
This track focuses on creating a comprehensive End-to-End (E2E) test suite that verifies the entire ContinuonXR stack running locally on a development machine (Windows). It involves initiating the `continuonbrain` runtime, optimizing its boot configuration, and verifying the availability and content of both the Web Server and the Flutter Web App using a multi-layered testing approach.

## 2. Functional Requirements

### 2.1 Brain Runtime Initialization
*   **Entry Point:** The test MUST utilize `python -m continuonbrain.startup_manager` to launch the full brain runtime.
*   **Optimization:** The configuration mechanism for entry points at boot MUST be optimized to streamline how the shell (robot hardware) initializes the brain.
*   **HAL Configuration:** The system MUST use a "Hardware Detect" strategy. It should attempt to load real hardware drivers first and seamlessly fall back to mock implementations if hardware is missing. This ensures the test environment closely mirrors production capabilities.

### 2.2 UI & Server Verification
The test suite MUST perform verification at three distinct layers:
1.  **Network/API Level:** Verify that all HTTP/gRPC endpoints are reachable and return correct status codes (using tools like `curl` or Python `requests`).
2.  **Content Inspection:** Download and parse the serving HTML/JS to ensure key artifacts (e.g., the Flutter app container) are present.
3.  **Full Browser Automation:** Use a headless browser automation tool (e.g., Selenium or Playwright) to render the application, execute JavaScript, and verify the UI state interactively.

## 3. Non-Functional Requirements
*   **Platform Support:** The test MUST run successfully on the `win32` development environment.
*   **Reliability:** The test should be deterministic and not flaky, handling the asynchronous nature of service startup gracefully.
*   **Logging:** Detailed logs MUST be captured for the Brain startup, Web Server requests, and Browser automation steps to aid in debugging.

## 4. Acceptance Criteria
*   [ ] `startup_manager` successfully launches the Brain runtime on Windows without crashing.
*   [ ] HAL correctly identifies missing hardware and falls back to mocks, logging the "Hardware Detect" process.
*   [ ] Boot entry point configuration is refactored/optimized for better maintainability.
*   [ ] Network tests pass: Web Server and API endpoints return 200 OK.
*   [ ] Content tests pass: HTML response contains expected Flutter app tags.
*   [ ] Browser tests pass: Headless browser successfully loads the page and verifies a visual element (e.g., "Continuon" title or logo).

## 5. Out of Scope
*   Testing on physical robot hardware (Raspberry Pi/Jetson) for this specific track (focus is on local dev environment).
*   Deep functional testing of the Flutter app features (focus is on loading/rendering and basic connectivity).
