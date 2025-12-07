import os
import sys
import subprocess
from PIL import Image

# Setup dummy image
img_path = "/tmp/test_red_square.png"
img = Image.new('RGB', (100, 100), color = 'red')
img.save(img_path)

# Run CLI with image
cmd = [
    sys.executable, 
    "continuonbrain/utils/gemini_cli.py", 
    "What color is this image?", 
    "--image", 
    img_path
]

print(f"Running: {' '.join(cmd)}")
env = os.environ.copy()
env["GEMINI_API_KEY"] = "AIzaSyDhXpcXCwhNP6Xl8HGNcAMAWi6TBdBw20A"

try:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if "red" in result.stdout.lower():
        print("✅ VERIFIED: Gemini saw the red image.")
    else:
        print("❌ FAILED: Response did not mention 'red'.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ EXECUTION FAILED: {e}")
    sys.exit(1)
