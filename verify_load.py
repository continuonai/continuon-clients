from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_id = "google/gemma-3n-E2B-it"
token = "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"

print(f"Loading {model_id}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        token=token,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ Success!")
except Exception as e:
    print(f"❌ Error: {e}")
