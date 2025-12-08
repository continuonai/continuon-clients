
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Python executable: {sys.executable}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"sys.path: {sys.path}")

try:
    print("Attempting to import transformers...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Success: transformers imported")
except ImportError as e:
    print(f"FAILURE: transformers import failed: {e}")
except Exception as e:
    print(f"FAILURE: transformers unexpected error: {e}")

try:
    print("Attempting to import torch...")
    import torch
    print("Success: torch imported")
except ImportError as e:
    print(f"FAILURE: torch import failed: {e}")
except Exception as e:
    print(f"FAILURE: torch unexpected error: {e}")

try:
    print("Attempting to import accelerate...")
    import accelerate
    print("Success: accelerate imported")
except ImportError as e:
    print(f"FAILURE: accelerate import failed: {e}")
except Exception as e:
    print(f"FAILURE: accelerate unexpected error: {e}")
