from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

def download_model():
    model_name = 'all-MiniLM-L6-v2'
    print(f"Downloading {model_name}...")
    
    # Force download to a specific directory if needed, 
    # but sentence-transformers uses ~/.cache/torch/sentence_transformers by default.
    model = SentenceTransformer(model_name)
    
    print(f"Model {model_name} ready.")

if __name__ == "__main__":
    download_model()
