import psutil
import requests
import sys
import os

print("=== DIAGNOSTIC REPORT ===")

# 1. Memory
print("\n[1] Memory & Swap (psutil):")
mem = psutil.virtual_memory()
swap = psutil.swap_memory()
print(f"Memory: {mem.percent}% ({mem.available // (1024*1024)} MB free)")
print(f"Swap:   {swap.percent}% (Used: {swap.used // (1024*1024)} MB / Total: {swap.total // (1024*1024)} MB)")

is_mem_ok = mem.percent < 90 and swap.percent < 85
print(f"Status for Training: {'✅ OK' if is_mem_ok else '❌ FAIL (Will trigger safety pause)'}")

# 2. API Server
print("\n[2] API Server Status:")
try:
    resp = requests.get("http://127.0.0.1:8081/api/status", timeout=5)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text[:200]}...")
except Exception as e:
    print(f"❌ Connection Failed: {e}")

# 3. Imports
print("\n[3] Import Checks:")
try:
    print("Importing VisionCore...", end=" ")
    from continuonbrain.services.vision_core import create_vision_core
    print("✅ OK")
except Exception as e:
    print(f"❌ FAIL: {e}")

try:
    print("Importing CoreModel...", end=" ")
    from continuonbrain.jax_models.core_model import CoreModel
    print("✅ OK")
except Exception as e:
    print(f"❌ FAIL: {e}")

print("\n=== END REPORT ===")
