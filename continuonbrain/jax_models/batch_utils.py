"""
JAX Batch Handling Utilities

Provides consistent batch dimension management for CoreModel and related modules.
This module centralizes all batch handling logic to prevent the scattered ndim checks
that previously existed across InputEncoder, CMSRead, CMSWrite, and CoreModel.

Usage:
    from continuonbrain.jax_models.batch_utils import (
        ensure_batch_dim, remove_batch_dim, BatchSpec,
        OBS_SPEC, ACTION_SPEC, STATE_SPEC, CMS_MEM_SPEC
    )

    # Normalize inputs
    x_obs, was_unbatched = ensure_batch_dim(x_obs, OBS_SPEC)

    # After processing, restore original shape if needed
    output = remove_batch_dim(output, was_unbatched)
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import jax.numpy as jnp

# Import our exception hierarchy
import sys
from pathlib import Path
# Add parent to path for imports during development
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from core.exceptions import BatchDimensionError
except ImportError:
    # Fallback if exceptions module not available yet
    class BatchDimensionError(Exception):
        def __init__(self, expected_ndim: int, actual_ndim: int, tensor_name: str):
            self.expected_ndim = expected_ndim
            self.actual_ndim = actual_ndim
            self.tensor_name = tensor_name
            super().__init__(
                f"Tensor '{tensor_name}' has {actual_ndim} dimensions, "
                f"expected {expected_ndim}."
            )


@dataclass
class BatchSpec:
    """
    Specification for expected tensor dimensions.

    Attributes:
        name: Name of the tensor (for error messages)
        unbatched_ndim: Expected number of dimensions for unbatched input
        batched_ndim: Expected number of dimensions for batched input (defaults to unbatched_ndim + 1)
    """
    name: str
    unbatched_ndim: int
    batched_ndim: Optional[int] = None

    def __post_init__(self):
        if self.batched_ndim is None:
            self.batched_ndim = self.unbatched_ndim + 1


# Standard tensor specifications for CoreModel
OBS_SPEC = BatchSpec("x_obs", unbatched_ndim=1)          # [obs_dim] -> [B, obs_dim]
ACTION_SPEC = BatchSpec("a_prev", unbatched_ndim=1)     # [action_dim] -> [B, action_dim]
REWARD_SPEC_SCALAR = BatchSpec("r_t", unbatched_ndim=0) # scalar -> [B, 1]
REWARD_SPEC_1D = BatchSpec("r_t", unbatched_ndim=1)     # [1] -> [B, 1]
STATE_SPEC = BatchSpec("state", unbatched_ndim=1)       # [d_s] -> [B, d_s]
CMS_MEM_SPEC = BatchSpec("cms_memory", unbatched_ndim=2, batched_ndim=3)  # [N, d] -> [B, N, d]
CMS_KEY_SPEC = BatchSpec("cms_key", unbatched_ndim=2, batched_ndim=3)     # [N, d_k] -> [B, N, d_k]
OBJECT_FEAT_SPEC = BatchSpec("object_features", unbatched_ndim=2, batched_ndim=3)  # [N_obj, d] -> [B, N_obj, d]


def ensure_batch_dim(
    tensor: jnp.ndarray,
    spec: BatchSpec,
    strict: bool = False,
) -> Tuple[jnp.ndarray, bool]:
    """
    Ensure tensor has batch dimension, adding if necessary.

    Args:
        tensor: Input tensor
        spec: Expected dimension specification
        strict: If True, raise error for unexpected dimensions. If False, try to handle gracefully.

    Returns:
        (tensor_with_batch_dim, was_unbatched)
        - tensor_with_batch_dim: Tensor with leading batch dimension
        - was_unbatched: True if batch dimension was added (for later removal)

    Raises:
        BatchDimensionError: If tensor has unexpected dimensions and strict=True

    Examples:
        >>> x = jnp.zeros((10,))  # unbatched
        >>> x_batched, was_unbatched = ensure_batch_dim(x, OBS_SPEC)
        >>> x_batched.shape
        (1, 10)
        >>> was_unbatched
        True

        >>> x = jnp.zeros((4, 10))  # already batched
        >>> x_batched, was_unbatched = ensure_batch_dim(x, OBS_SPEC)
        >>> x_batched.shape
        (4, 10)
        >>> was_unbatched
        False
    """
    if tensor.ndim == spec.batched_ndim:
        # Already batched
        return tensor, False
    elif tensor.ndim == spec.unbatched_ndim:
        # Add batch dimension
        return tensor[None, ...], True
    else:
        if strict:
            raise BatchDimensionError(
                expected_ndim=spec.unbatched_ndim,
                actual_ndim=tensor.ndim,
                tensor_name=spec.name,
            )
        # Try to handle gracefully
        if tensor.ndim < spec.unbatched_ndim:
            # Need to add dimensions
            while tensor.ndim < spec.batched_ndim:
                tensor = tensor[None, ...]
            return tensor, True
        else:
            # Too many dimensions - just return as is with warning
            return tensor, False


def ensure_batch_dim_reward(
    r_t: jnp.ndarray,
) -> Tuple[jnp.ndarray, bool]:
    """
    Special handling for reward tensor which can be scalar, 1D, or 2D.

    Always returns [B, 1] shape.

    Args:
        r_t: Reward tensor (scalar, [1], or [B, 1])

    Returns:
        (r_t_batched, was_unbatched)
    """
    if r_t.ndim == 0:
        # Scalar -> [1, 1]
        return r_t[None, None], True
    elif r_t.ndim == 1:
        # [1] or [B] -> [1, 1] or [B, 1]
        if r_t.shape[0] == 1:
            # Assume unbatched [1]
            return r_t[None, :], True
        else:
            # Assume batched [B] - add trailing dim
            return r_t[:, None], False
    elif r_t.ndim == 2:
        # Already [B, 1]
        return r_t, False
    else:
        raise BatchDimensionError(
            expected_ndim=1,
            actual_ndim=r_t.ndim,
            tensor_name="r_t",
        )


def remove_batch_dim(
    tensor: jnp.ndarray,
    was_unbatched: bool,
) -> jnp.ndarray:
    """
    Remove batch dimension if it was added.

    Args:
        tensor: Tensor with batch dimension
        was_unbatched: True if batch dimension was added by ensure_batch_dim

    Returns:
        tensor: Original shape if was_unbatched, otherwise unchanged

    Examples:
        >>> x = jnp.zeros((1, 10))
        >>> remove_batch_dim(x, was_unbatched=True).shape
        (10,)

        >>> x = jnp.zeros((4, 10))
        >>> remove_batch_dim(x, was_unbatched=False).shape
        (4, 10)
    """
    if was_unbatched and tensor.ndim > 0:
        return tensor[0]
    return tensor


def remove_batch_dim_from_list(
    tensors: List[jnp.ndarray],
    was_unbatched: bool,
) -> List[jnp.ndarray]:
    """
    Remove batch dimension from a list of tensors.

    Args:
        tensors: List of tensors with batch dimension
        was_unbatched: True if batch dimension was added

    Returns:
        List of tensors with batch dimension removed if was_unbatched
    """
    if was_unbatched:
        return [t[0] for t in tensors]
    return tensors


def remove_batch_dim_from_dict(
    info: Dict[str, Any],
    was_unbatched: bool,
    list_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Remove batch dimensions from info dict values.

    Args:
        info: Dictionary with tensor values
        was_unbatched: True if batch dimension was added
        list_keys: Keys whose values are lists of tensors

    Returns:
        Dictionary with batch dimensions removed from tensor values
    """
    if not was_unbatched:
        return info

    list_keys = list_keys or []
    result = {}

    for k, v in info.items():
        if isinstance(v, jnp.ndarray):
            result[k] = v[0]
        elif isinstance(v, list) and k in list_keys:
            result[k] = [x[0] if isinstance(x, jnp.ndarray) else x for x in v]
        else:
            result[k] = v

    return result


def normalize_inputs(
    x_obs: jnp.ndarray,
    a_prev: jnp.ndarray,
    r_t: jnp.ndarray,
    s_prev: jnp.ndarray,
    w_prev: jnp.ndarray,
    p_prev: jnp.ndarray,
    cms_memories: List[jnp.ndarray],
    cms_keys: List[jnp.ndarray],
    object_features: Optional[jnp.ndarray] = None,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    List[jnp.ndarray], List[jnp.ndarray],
    Optional[jnp.ndarray], bool
]:
    """
    Normalize all CoreModel inputs to batched format.

    This is the single entry point for batch normalization.

    Args:
        x_obs: Observation [obs_dim] or [B, obs_dim]
        a_prev: Previous action [action_dim] or [B, action_dim]
        r_t: Reward (scalar, [1], or [B, 1])
        s_prev: Previous fast state [d_s] or [B, d_s]
        w_prev: Previous wave state [d_w] or [B, d_w]
        p_prev: Previous particle state [d_p] or [B, d_p]
        cms_memories: List of CMS memory matrices [N, d] or [B, N, d]
        cms_keys: List of CMS key matrices [N, d_k] or [B, N, d_k]
        object_features: Optional object features [N_obj, d] or [B, N_obj, d]

    Returns:
        Tuple of normalized inputs plus was_unbatched flag
    """
    # Determine if unbatched by checking x_obs
    was_unbatched = x_obs.ndim == 1

    # Normalize observation and action
    if was_unbatched:
        x_obs = x_obs[None, :]
        a_prev = a_prev[None, :]

    # Normalize reward (special handling)
    r_t, _ = ensure_batch_dim_reward(r_t)

    # Normalize states
    if s_prev.ndim == 1:
        s_prev = s_prev[None, :]
    if w_prev.ndim == 1:
        w_prev = w_prev[None, :]
    if p_prev.ndim == 1:
        p_prev = p_prev[None, :]

    # Normalize CMS memories and keys
    if cms_memories[0].ndim == 2:
        cms_memories = [m[None, :, :] for m in cms_memories]
        cms_keys = [k[None, :, :] for k in cms_keys]

    # Normalize object features if provided
    if object_features is not None and object_features.ndim == 2:
        object_features = object_features[None, :, :]

    return (
        x_obs, a_prev, r_t,
        s_prev, w_prev, p_prev,
        cms_memories, cms_keys,
        object_features, was_unbatched
    )


def denormalize_outputs(
    y_t: jnp.ndarray,
    info: Dict[str, Any],
    was_unbatched: bool,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Denormalize CoreModel outputs back to original shape.

    Args:
        y_t: Output tensor [B, output_dim]
        info: Info dictionary with batched tensors
        was_unbatched: True if inputs were unbatched

    Returns:
        (y_t, info) with batch dimension removed if was_unbatched
    """
    if not was_unbatched:
        return y_t, info

    y_t = y_t[0]
    info = remove_batch_dim_from_dict(
        info,
        was_unbatched=True,
        list_keys=['attention_weights', 'cms_memories', 'cms_keys']
    )

    return y_t, info


def validate_batch_consistency(
    tensors: Dict[str, jnp.ndarray],
    expected_batch_size: Optional[int] = None,
) -> int:
    """
    Validate that all tensors have consistent batch sizes.

    Args:
        tensors: Dictionary of tensor name -> tensor
        expected_batch_size: If provided, validate against this size

    Returns:
        Inferred batch size

    Raises:
        ValueError: If batch sizes are inconsistent
    """
    batch_sizes = {}

    for name, tensor in tensors.items():
        if tensor.ndim > 0:
            batch_sizes[name] = tensor.shape[0]

    unique_sizes = set(batch_sizes.values())

    if len(unique_sizes) > 1:
        details = ", ".join(f"{k}={v}" for k, v in batch_sizes.items())
        raise ValueError(f"Inconsistent batch sizes: {details}")

    inferred_size = next(iter(unique_sizes)) if unique_sizes else 1

    if expected_batch_size is not None and inferred_size != expected_batch_size:
        raise ValueError(
            f"Expected batch size {expected_batch_size}, got {inferred_size}"
        )

    return inferred_size
