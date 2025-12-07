
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-3-4b-it"
# Fallback to hardcoded token if env var is missing (detected from startup_manager.py)
TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "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")

def download_model():
    logger.info(f"Starting download for: {MODEL_ID}")
    
    if not TOKEN:
        logger.warning("HUGGINGFACE_TOKEN not found! Download will likely fail for gated models.")

    try:
        # Download Tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
        logger.info("Tokenizer downloaded successfully.")

        # Download Model
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            token=TOKEN,
            # device_map="auto", # Removed to avoid accelerate dependency for download
            # offload_folder="/tmp/offload"
        )
        logger.info("Model downloaded successfully.")
        
        print(f"✅ Successfully downloaded {MODEL_ID}")
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
