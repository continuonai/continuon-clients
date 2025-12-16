import sys
import traceback

print("Checking MediaPipe imports...")
try:
    import mediapipe
    print(f"MediaPipe: {mediapipe.__file__}")
except ImportError:
    print("MediaPipe not installed")
    sys.exit(1)

try:
    import mediapipe.tasks
    print(f"MediaPipe Tasks: {mediapipe.tasks.__file__}")
except Exception:
    print("Failed to import mediapipe.tasks")
    traceback.print_exc()

try:
    from mediapipe.tasks.python.genai import llm_inference
    print("Success: llm_inference imported")
except Exception:
    print("Failed to import llm_inference")
    traceback.print_exc()
