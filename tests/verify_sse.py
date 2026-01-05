import requests
import time
import threading
import json
import sys

def listen_for_events():
    print("Connecting to SSE stream...")
    try:
        response = requests.get('http://localhost:8081/api/events', stream=True, timeout=5)
        if response.status_code != 200:
            print(f"Failed to connect: {response.status_code}")
            return

        print("Connected. Waiting for events...")
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    try:
                        data = json.loads(data_str)
                        print(f"RECEIVED EVENT: {data.get('event_type', 'unknown')} - {data.get('message', '')}")
                        if data.get('event_type') == 'test_event':
                            print("SUCCESS: Received test event!")
                            sys.exit(0)
                    except json.JSONDecodeError:
                        print(f"Received raw data: {data_str}")
    except Exception as e:
        print(f"Stream error: {e}")

def trigger_event():
    time.sleep(2)
    print("Triggering test event via log...")
    # We can trigger by sending a request that causes a log, or by writing to the log if we can't hit an endpoint easily.
    # But wait, we modified server.py to listen to SystemEventLogger. 
    # SystemEventLogger writes to file AND calls listeners.
    # The server instance has the listener.
    # So if we assume the server is running, any log entry created by the server process will work.
    # Accessing /api/ping causes no log.
    # Accessing /api/mode/manual_control might cause a log if it changes mode?
    # Or /api/admin/factory_reset? (Too dangerous).
    # Let's try /api/mode/manual_control - usually logs mode changes.
    
    # Alternatively, we can rely on "server_ready" log if we start the server fresh? 
    # But the script connects AFTER server start.
    
    # Getting /api/status doesn't log.
    
    # Triggering a known log:
    # ui_launch logs "ui_launch".
    # But that's desktop only.
    
    # Let's try sending a POST to /api/mode/manual_control.
    print("Sending mode change to trigger log...")
    try:
        requests.get('http://localhost:8081/api/mode/manual_control')
    except Exception as e:
        print(f"Trigger error: {e}")

# We will run this script separately while server is running.
if __name__ == "__main__":
    t = threading.Thread(target=trigger_event)
    t.start()
    listen_for_events()
