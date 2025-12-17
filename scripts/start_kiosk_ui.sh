#!/bin/bash
# start_kiosk_ui.sh
# Waits for ContinuonBrain UI to be available, then launches Kiosk browser.
# Add this to your Desktop Session Startup (e.g., ~/.config/autostart).

TARGET_URL="http://localhost:8081/ui"
MAX_RETRIES=60
SLEEP_SEC=2

echo "Waiting for Robot UI at $TARGET_URL..."

for ((i=1; i<=MAX_RETRIES; i++)); do
    if curl -s --head "$TARGET_URL" | grep "200 OK" > /dev/null; then
        echo "UI is up! Launching Kiosk..."
        
        # Launch using the python logic which handles flags nicely,
        # or call browser directly. Calling python is safer to keep logic in one place.
        # However, we can just call the browser directly for speed/simplicity here.
        
        /usr/bin/chromium-browser \
            --kiosk \
            --password-store=basic \
            --no-default-browser-check \
            --no-first-run \
            --noerrdialogs \
            --disable-infobars \
            --check-for-update-interval=31536000 \
            --simulated-keyring \
            "$TARGET_URL" &
            
        exit 0
    fi
    echo "Retry $i/$MAX_RETRIES..."
    sleep $SLEEP_SEC
done

echo "Timed out waiting for UI."
exit 1
