import os
from transformers import AutoConfig
import logging

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

model_ids = [
    "google/gemma-3n-2b-it",
    "google/gemma-3n-E2B-it", 
    "google/gemma-3-2b-it",
    "google/gemma-3n-it",
    "google/gemma-2b-it" # Control
]

token = "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"

print("Verifying Model IDs...")
for mid in model_ids:
    try:
        config = AutoConfig.from_pretrained(mid, token=token)
        print(f"✅ SUCCESS: {mid} exists!")
    except Exception as e:
        error_str = str(e)
        if "404" in error_str:
             print(f"❌ FAIL: {mid} not found (404)")
        elif "401" in error_str:
             print(f"❌ FAIL: {mid} unauthorized (401)")
        else:
             print(f"❌ FAIL: {mid} - {e}")
