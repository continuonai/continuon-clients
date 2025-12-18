#!/usr/bin/env python3
"""
Discover and download small models for multi-source learning.

This script:
1. Lists models already in HuggingFace cache
2. Suggests additional small models to download
3. Optionally downloads recommended models
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import argparse

try:
    from huggingface_hub import scan_cache_dir, HfApi, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub")


# Recommended small models by category
RECOMMENDED_MODELS = {
    "lm_ultra_small": [
        "google/gemma-3-270m-it",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2-0.5B-Instruct",
    ],
    "lm_small": [
        "google/gemma-2-2b-it",
        "google/gemma-3n-2b-it",
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
    "vlm": [
        "Salesforce/blip-image-captioning-base",
        "Qwen/Qwen2-VL-2B-Instruct",
    ],
    "embeddings": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2",
        "thenlper/gte-small",
    ],
    "code": [
        "bigcode/starcoder2-3b",
        "Salesforce/codegen-350M-mono",
    ],
    "math": [
        "microsoft/phi-2",  # 2.7B, strong math reasoning
        "microsoft/phi-3-mini-4k-instruct",  # 3.8B, efficient
        "huggingface/SmolLM-1.7B-Instruct",  # 1.7B, math-focused
    ],
    "physics": [
        "microsoft/phi-2",  # Also good for physics
        "microsoft/phi-3-mini-4k-instruct",  # Physics reasoning
    ],
}

RECOMMENDED_DATASETS = {
    "wikipedia": [
        "wikimedia/wikipedia",
        "rahular/simple-wikipedia",
    ],
    "vla": [
        "Open-X-Embodiment/bridge_data_v2",
        "HuggingFaceVLA/community_dataset_v3",
    ],
    "chat": [
        "HuggingFaceH4/ultrachat_200k",
    ],
    "vlm": [
        "liuhaotian/LLaVA-Instruct-150K",
        "philschmid/amazon-product-descriptions-vlm",
    ],
}


def get_hf_cache_dir() -> Path:
    """Get HuggingFace cache directory."""
    env_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hub:
        return Path(env_hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    default = Path.home() / ".cache" / "huggingface" / "hub"
    if default.exists():
        return default
    # Fallback for shared cache
    return Path("/home/craigm26/.cache/huggingface/hub")


def list_cached_models() -> Dict[str, List[str]]:
    """List models and datasets in cache."""
    cache_dir = get_hf_cache_dir()
    models = []
    datasets = []
    
    if not cache_dir.exists():
        return {"models": [], "datasets": []}
    
    for item in cache_dir.iterdir():
        if not item.is_dir():
            continue
        
        name = item.name
        if name.startswith("models--"):
            # Convert "models--google--gemma-2-2b-it" to "google/gemma-2-2b-it"
            model_id = name.replace("models--", "").replace("--", "/")
            models.append(model_id)
        elif name.startswith("datasets--"):
            dataset_id = name.replace("datasets--", "").replace("--", "/")
            datasets.append(dataset_id)
    
    return {
        "models": sorted(set(models)),
        "datasets": sorted(set(datasets))
    }


def scan_cache_with_hf_hub() -> Dict:
    """Use huggingface_hub to scan cache (more detailed)."""
    if not HF_AVAILABLE:
        return {}
    
    try:
        cache_info = scan_cache_dir()
        models = []
        datasets = []
        
        for repo in cache_info.repos:
            repo_id = repo.repo_id
            repo_type = repo.repo_type
            
            size_gb = sum(rev.size_on_disk for rev in repo.revisions) / (1024**3)
            
            info = {
                "id": repo_id,
                "size_gb": round(size_gb, 2),
                "revisions": len(repo.revisions),
            }
            
            if repo_type == "model":
                models.append(info)
            elif repo_type == "dataset":
                datasets.append(info)
        
        return {
            "models": sorted(models, key=lambda x: x["id"]),
            "datasets": sorted(datasets, key=lambda x: x["id"]),
        }
    except Exception as e:
        print(f"⚠️  Error scanning cache: {e}")
        return {}


def print_cached_items(cached: Dict):
    """Print cached models and datasets."""
    print("\n" + "="*70)
    print("📦 CACHED MODELS")
    print("="*70)
    
    if cached.get("models"):
        for model in cached["models"]:
            if isinstance(model, dict):
                print(f"  ✅ {model['id']:<50} ({model['size_gb']:.2f} GB, {model['revisions']} revs)")
            else:
                print(f"  ✅ {model}")
    else:
        print("  (none)")
    
    print("\n" + "="*70)
    print("📚 CACHED DATASETS")
    print("="*70)
    
    if cached.get("datasets"):
        for dataset in cached["datasets"]:
            if isinstance(dataset, dict):
                print(f"  ✅ {dataset['id']:<50} ({dataset['size_gb']:.2f} GB, {dataset['revisions']} revs)")
            else:
                print(f"  ✅ {dataset}")
    else:
        print("  (none)")


def find_missing_models(cached: Dict) -> Dict[str, List[str]]:
    """Find recommended models not yet cached."""
    cached_model_ids = set()
    cached_dataset_ids = set()
    
    for model in cached.get("models", []):
        if isinstance(model, dict):
            cached_model_ids.add(model["id"])
        else:
            cached_model_ids.add(model)
    
    for dataset in cached.get("datasets", []):
        if isinstance(dataset, dict):
            cached_dataset_ids.add(dataset["id"])
        else:
            cached_dataset_ids.add(dataset)
    
    missing = {
        "models": {},
        "datasets": {},
    }
    
    for category, model_list in RECOMMENDED_MODELS.items():
        missing_models = [m for m in model_list if m not in cached_model_ids]
        if missing_models:
            missing["models"][category] = missing_models
    
    for category, dataset_list in RECOMMENDED_DATASETS.items():
        missing_datasets = [d for d in dataset_list if d not in cached_dataset_ids]
        if missing_datasets:
            missing["datasets"][category] = missing_datasets
    
    return missing


def print_recommendations(missing: Dict):
    """Print recommended models to download."""
    print("\n" + "="*70)
    print("💡 RECOMMENDED MODELS TO DOWNLOAD")
    print("="*70)
    
    if not missing.get("models") and not missing.get("datasets"):
        print("  ✨ All recommended models are already cached!")
        return
    
    if missing.get("models"):
        for category, models in missing["models"].items():
            print(f"\n  📌 {category.replace('_', ' ').title()}:")
            for model in models:
                print(f"     - {model}")
    
    if missing.get("datasets"):
        print("\n  📚 Recommended Datasets:")
        for category, datasets in missing["datasets"].items():
            print(f"\n     {category.replace('_', ' ').title()}:")
            for dataset in datasets:
                print(f"       - {dataset}")


def download_model(model_id: str, token: Optional[str] = None) -> bool:
    """Download a model."""
    if not HF_AVAILABLE:
        print("❌ huggingface_hub not available")
        return False
    
    try:
        print(f"⬇️  Downloading {model_id}...")
        snapshot_download(
            repo_id=model_id,
            token=token,
            local_files_only=False,
        )
        print(f"✅ Downloaded {model_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        return False


def download_dataset(dataset_id: str, token: Optional[str] = None) -> bool:
    """Download a dataset."""
    if not HF_AVAILABLE:
        print("❌ huggingface_hub not available")
        return False
    
    try:
        print(f"⬇️  Downloading dataset {dataset_id}...")
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            token=token,
            local_files_only=False,
        )
        print(f"✅ Downloaded dataset {dataset_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {dataset_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Discover and download small models for multi-source learning"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached models and datasets"
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Show recommended models to download"
    )
    parser.add_argument(
        "--download",
        nargs="+",
        metavar="MODEL_ID",
        help="Download specific models or datasets"
    )
    parser.add_argument(
        "--download-all-recommended",
        action="store_true",
        help="Download all recommended models (use with caution!)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace token (or set HUGGINGFACE_TOKEN env var)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use huggingface_hub scan_cache_dir for detailed info"
    )
    
    args = parser.parse_args()
    
    if not args.list and not args.recommend and not args.download and not args.download_all_recommended:
        # Default: list and recommend
        args.list = True
        args.recommend = True
    
    # Get cached items
    if args.detailed and HF_AVAILABLE:
        cached = scan_cache_with_hf_hub()
    else:
        cached = list_cached_models()
    
    if args.list:
        print_cached_items(cached)
    
    if args.recommend:
        missing = find_missing_models(cached)
        print_recommendations(missing)
    
    # Download specific items
    if args.download:
        token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
        for item_id in args.download:
            if "/" in item_id:
                # Try as model first, then dataset
                if not download_model(item_id, token):
                    download_dataset(item_id, token)
            else:
                print(f"⚠️  Invalid model/dataset ID: {item_id}")
    
    # Download all recommended
    if args.download_all_recommended:
        token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
        missing = find_missing_models(cached)
        
        print("\n⚠️  This will download many models. Continue? (y/N): ", end="")
        if input().lower() != 'y':
            print("Cancelled.")
            return
        
        all_models = []
        for models in missing.get("models", {}).values():
            all_models.extend(models)
        
        all_datasets = []
        for datasets in missing.get("datasets", {}).values():
            all_datasets.extend(datasets)
        
        print(f"\n⬇️  Downloading {len(all_models)} models and {len(all_datasets)} datasets...")
        
        for model_id in all_models:
            download_model(model_id, token)
        
        for dataset_id in all_datasets:
            download_dataset(dataset_id, token)
        
        print("\n✅ Download complete!")


if __name__ == "__main__":
    main()
