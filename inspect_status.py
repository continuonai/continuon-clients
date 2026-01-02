import requests
import json

try:
    resp = requests.get("http://127.0.0.1:8081/api/status", timeout=5)
    data = resp.json()
    print("Top Level Keys:", list(data.keys()))
    print("\nCapabilities:", data.get("capabilities"))
    print("\nDetected Hardware:", data.get("detected_hardware"))
    print("\nVision Core Key?", data.get("vision_core"))
    print("\nHardware Mode:", data.get("hardware_mode"))
except Exception as e:
    print(e)
