
import os
import sys

# Ensure CWD is in path
sys.path.append(os.getcwd())

os.environ["CONTINUON_USE_LITERT"] = "1"
os.environ["CONTINUON_HEADLESS"] = "0" 
os.environ["CONTINUON_ALLOW_TRANSFORMERS_CHAT"] = "0"

print("Importing build_chat_service...")
try:
    from continuonbrain.gemma_chat import build_chat_service
    print(f"Imported from: {build_chat_service.__module__}")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Calling build_chat_service()...")
service = build_chat_service()
print(f"Service created: {service}")
if service:
    print(f"Type: {type(service)}")
