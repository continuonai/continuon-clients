
import sys
import os
import logging
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from continuonbrain.gemma_chat import GemmaChat

# Config logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🚀 Starting Integrated VLM Verification...")
    
    # 1. Initialize Chat
    chat = GemmaChat(model_name="google/gemma-3n-E2B-it")
    
    # 2. Load Model
    print("🧠 Loading Model (this may take a while)...")
    if not chat.load_model():
        print("❌ Failed to load model.")
        return
        
    if not chat.is_vlm:
        print("⚠️  Warning: Model loaded but NOT in VLM mode (is_vlm=False).")
    else:
        print("✅ Model loaded in VLM mode!")

    # 3. Get Image (Mock or Camera)
    try:
        # Try to capture from camera if OAK libraries are present
        # For verification speed vs reliability, let's just use a generated test image
        # unless user insists on real camera. 
        # The user prompt says "using the camera". I should try.
        try:
            from continuonbrain.drivers.oak_depth_capture import OAKDepthCapture
            print("📷 Attempting to connect to camera...")
            camera = OAKDepthCapture()
            frame = camera.capture_frame()
            if frame and "rgb" in frame:
                # OAK returns numpy array usually. transformer processor expects PIL or numpy.
                # cv2 image is BGR, need RGB. OAK usually provides RGB if requested properly.
                # Assuming frame["rgb"] is a numpy array.
                image = Image.fromarray(frame["rgb"])
                print("✅ Captured image from camera.")
            else:
                raise RuntimeError("No frame captured")
        except Exception as e:
            print(f"⚠️  Camera capture failed ({e}). Using synthetic test image.")
            # Create a simple red image
            image = Image.new('RGB', (640, 480), color = 'red')
    except Exception as e:
        print(f"Error preparing image: {e}")
        return

    # 4. Chat with VLM
    query = "What do you see in this image?"
    print(f"❓ Query: {query}")
    
    response = chat.chat(query, image=image)
    
    print("\n" + "="*40)
    print(f"🤖 Agent Answer: {response}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
