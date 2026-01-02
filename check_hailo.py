import requests
import json
import time

print("Checking API Status for VisionCore/Hailo...")
for i in range(10):
    try:
        resp = requests.get("http://127.0.0.1:8081/api/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            vision = data.get("vision_core", False)
            print(f"Attempt {i+1}: VisionCore Available = {vision}")
            if vision:
                print("✅ Hailo/VisionCore is READY.")
                break
        else:
             print(f"Attempt {i+1}: API Error {resp.status_code}")
    except Exception as e:
        print(f"Attempt {i+1}: Connection failed ({e})")
    time.sleep(2)
