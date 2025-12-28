"""
Memory-Aware Model Selector

Automatically selects appropriate chat model size based on available system memory.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Profile for a chat model."""
    id: str
    name: str
    min_memory_mb: int  # Minimum system RAM to load this model
    model_memory_mb: int  # Approximate memory the model uses
    priority: int  # Lower = higher priority when memory allows
    model_type: str  # "jax", "transformers", "litert", "mock"
    

# Model profiles ordered by size (smallest to largest)
MODEL_PROFILES: List[ModelProfile] = [
    # Tiny models - work on any system
    ModelProfile(
        id="mock",
        name="Mock Chat (Testing)",
        min_memory_mb=0,
        model_memory_mb=10,
        priority=100,
        model_type="mock"
    ),
    
    # JAX CoreModel - very lightweight
    ModelProfile(
        id="jax-core",
        name="JAX CoreModel",
        min_memory_mb=1024,  # 1GB
        model_memory_mb=200,
        priority=10,
        model_type="jax-core"
    ),
    
    # LiteRT 270M - optimized for edge
    ModelProfile(
        id="gemma-270m-litert",
        name="Gemma 270M (LiteRT)",
        min_memory_mb=2048,  # 2GB
        model_memory_mb=500,
        priority=20,
        model_type="litert"
    ),
    
    # Gemma 370M JAX - small but capable
    ModelProfile(
        id="google/gemma-370m",
        name="Gemma 370M (JAX)",
        min_memory_mb=3072,  # 3GB
        model_memory_mb=800,
        priority=25,
        model_type="jax"
    ),
    
    # Gemma 2B - medium size
    ModelProfile(
        id="google/gemma-2-2b-it",
        name="Gemma 2 2B",
        min_memory_mb=6144,  # 6GB
        model_memory_mb=4000,
        priority=30,
        model_type="transformers"
    ),
    
    # Gemma 3n 2B - larger
    ModelProfile(
        id="google/gemma-3n-E2B-it",
        name="Gemma 3n 2B",
        min_memory_mb=8192,  # 8GB
        model_memory_mb=5000,
        priority=35,
        model_type="transformers"
    ),
    
    # Gemma 3n 4B - requires more memory
    ModelProfile(
        id="google/gemma-3n-E4B-it",
        name="Gemma 3n 4B",
        min_memory_mb=12288,  # 12GB
        model_memory_mb=8000,
        priority=40,
        model_type="transformers"
    ),
]


def get_system_memory_mb() -> int:
    """Get total system memory in MB."""
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        # Fallback: read from /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal is in kB
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
    # Default assumption if we can't detect
    return 4096  # Assume 4GB


def get_available_memory_mb() -> int:
    """Get available memory in MB (total - used)."""
    try:
        import psutil
        return psutil.virtual_memory().available // (1024 * 1024)
    except ImportError:
        # Fallback: estimate as 50% of total
        return get_system_memory_mb() // 2


def get_memory_tier() -> str:
    """
    Classify system into memory tiers.
    
    Returns:
        One of: "tiny", "small", "medium", "large", "xlarge"
    """
    total_mb = get_system_memory_mb()
    
    if total_mb < 2048:
        return "tiny"      # < 2GB - very constrained (mock only)
    elif total_mb < 4096:
        return "small"     # 2-4GB - can run JAX CoreModel
    elif total_mb < 8192:
        return "medium"    # 4-8GB - can run small Gemma models
    elif total_mb < 16384:
        return "large"     # 8-16GB - can run medium models
    else:
        return "xlarge"    # 16GB+ - can run larger models


def select_model_for_memory(
    total_memory_mb: Optional[int] = None,
    reserve_memory_mb: int = 2000,
    prefer_type: Optional[str] = None
) -> ModelProfile:
    """
    Select the best model profile based on available memory.
    
    Args:
        total_memory_mb: Total system RAM (auto-detected if None)
        reserve_memory_mb: Memory to reserve for system/other processes
        prefer_type: Preferred model type ("jax", "transformers", "litert")
        
    Returns:
        Best ModelProfile for the current system
    """
    if total_memory_mb is None:
        total_memory_mb = get_system_memory_mb()
    
    available_for_model = total_memory_mb - reserve_memory_mb
    
    logger.info(f"System memory: {total_memory_mb}MB, available for model: {available_for_model}MB")
    
    # Filter models that fit in memory
    viable_models = [
        p for p in MODEL_PROFILES 
        if p.min_memory_mb <= total_memory_mb and p.model_memory_mb <= available_for_model
    ]
    
    if not viable_models:
        logger.warning(f"No models fit in {available_for_model}MB, using mock")
        return MODEL_PROFILES[0]  # Return mock
    
    # If preferred type specified, try to match
    if prefer_type:
        preferred = [p for p in viable_models if p.model_type == prefer_type]
        if preferred:
            viable_models = preferred
    
    # Sort by priority (lower = better) and return best
    viable_models.sort(key=lambda p: p.priority)
    selected = viable_models[0]
    
    logger.info(f"Selected model: {selected.name} (needs {selected.model_memory_mb}MB)")
    return selected


def get_recommended_models(max_count: int = 3) -> List[ModelProfile]:
    """
    Get list of recommended models for the current system.
    
    Returns models in order of preference (best fit first).
    """
    total_memory_mb = get_system_memory_mb()
    reserve_memory_mb = 2000
    available_for_model = total_memory_mb - reserve_memory_mb
    
    viable = [
        p for p in MODEL_PROFILES
        if p.min_memory_mb <= total_memory_mb and p.model_memory_mb <= available_for_model
    ]
    
    # Sort by priority
    viable.sort(key=lambda p: p.priority)
    return viable[:max_count]


def get_memory_status() -> Dict[str, Any]:
    """Get current memory status for diagnostics."""
    total = get_system_memory_mb()
    available = get_available_memory_mb()
    tier = get_memory_tier()
    recommended = select_model_for_memory(total)
    
    return {
        "total_mb": total,
        "available_mb": available,
        "tier": tier,
        "recommended_model": recommended.name,
        "recommended_model_id": recommended.id,
        "recommended_model_type": recommended.model_type,
        "model_memory_required_mb": recommended.model_memory_mb,
    }


# Convenience function for startup
def print_memory_recommendation():
    """Print memory status and model recommendation to console."""
    status = get_memory_status()
    
    print("=" * 55)
    print("  MEMORY-AWARE MODEL SELECTION")
    print("=" * 55)
    print(f"  System RAM:      {status['total_mb']:,} MB ({status['total_mb']//1024} GB)")
    print(f"  Available:       {status['available_mb']:,} MB")
    print(f"  Memory Tier:     {status['tier'].upper()}")
    print(f"  Recommended:     {status['recommended_model']}")
    print(f"  Model Memory:    {status['model_memory_required_mb']:,} MB")
    print("=" * 55)


if __name__ == "__main__":
    # Test the selector
    print_memory_recommendation()
    
    print("\nAll viable models for this system:")
    for m in get_recommended_models(max_count=10):
        print(f"  - {m.name} ({m.model_type}): needs {m.model_memory_mb}MB")

