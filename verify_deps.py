
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import ai_edge_litert
    print(f"ai_edge_litert imported: {ai_edge_litert}")
except ImportError as e:
    print(f"FAIL: ai_edge_litert import failed: {e}")

try:
    import ai_edge_litert.interpreter
    print("ai_edge_litert.interpreter imported")
except ImportError as e:
    print(f"FAIL: ai_edge_litert.interpreter import failed: {e}")

try:
    sys.path.append(os.getcwd())
    from continuonbrain.services.chat.litert_chat import HAS_LITERT
    print(f"HAS_LITERT: {HAS_LITERT}")
except ImportError as e:
    print(f"FAIL: litert_chat import failed: {e}")
except Exception as e:
    print(f"FAIL: litert_chat other error: {e}")
