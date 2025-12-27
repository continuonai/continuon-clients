# ContinuonXR End-to-End Testing

This document describes how to run the full-stack End-to-End (E2E) smoke tests. These tests verify that the Brain Runtime, API Server, and Web UI are all functioning together correctly.

## Prerequisites

1.  **Python Dependencies:**
    ```bash
    pip install -r continuonbrain/requirements.txt
    pip install pytest requests playwright pytest-playwright
    ```

2.  **Browser Setup (for UI tests):**
    ```bash
    playwright install chromium
    ```

## Running the E2E Test Suite

The E2E tests are located in `tests/e2e/test_full_stack_smoke.py`.

To run the tests:

```bash
pytest tests/e2e/test_full_stack_smoke.py -s
```

### What the test does:
1.  **Phase 1 (Startup):** Launches `continuonbrain.startup_manager` in a background process with Mock Hardware enabled.
2.  **Layer 1 (Network):** Pings the API and UI endpoints to ensure they are reachable.
3.  **Layer 2 (Content):** Inspects the HTML response from the UI server to ensure the Flutter app is being served.
4.  **Layer 3 (UI/Browser):** Uses Playwright to launch a headless browser, navigate to the UI, and verify that the page renders correctly.
5.  **Teardown:** Safely shuts down the brain services and cleans up temporary config files.

## Troubleshooting

- **Port Conflict:** Ensure port 8080 is not being used by another process before running the tests.
- **Hardware Drivers:** The tests use `CONTINUON_MOCK_HARDWARE=1` to allow running on development machines without physical sensors/actuators.
- **Headless Mode:** Browser tests run in headless mode by default. To see the browser window, modify `headless=False` in the test script.
