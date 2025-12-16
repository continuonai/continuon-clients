"""
RLDS Dataset Loader

Load TFRecord episodes for training and convert to JAX arrays.
Supports batching, shuffling, and prefetching for efficient TPU training.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import hashlib

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


def _tf_is_usable() -> bool:
    """
    TensorFlow is optional. On some dev machines (notably NumPy>=2 with older TF wheels),
    importing TF can throw noisy errors and even crash.

    We avoid importing TF at module import time and only enable TF paths when it is
    plausibly usable.
    """
    if importlib.util.find_spec("tensorflow") is None:
        return False
    try:
        major = int(str(np.__version__).split(".", 1)[0])
        if major >= 2:
            # Most TF builds in the wild still pin to NumPy<2; avoid noisy import failures.
            return False
    except Exception:
        pass
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception:
        return False


TF_AVAILABLE = _tf_is_usable()


def _parse_tfrecord_example(example_proto: "tf.Tensor") -> Dict[str, "tf.Tensor"]:
    """
    Parse a TFRecord example.
    
    Args:
        example_proto: Serialized TFRecord example
    
    Returns:
        Dictionary of parsed features
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for TFRecord parsing")
    
    import tensorflow as tf  # type: ignore

    feature_description = {
        'episode_id': tf.io.FixedLenFeature([], tf.string),
        'step_index': tf.io.FixedLenFeature([], tf.int64),
        'reward': tf.io.FixedLenFeature([], tf.float32),
        'is_terminal': tf.io.FixedLenFeature([], tf.int64),
        'observation': tf.io.FixedLenFeature([], tf.string),
        'action': tf.io.FixedLenFeature([], tf.string),
        'observation_command': tf.io.VarLenFeature(tf.float32),
        'action_command': tf.io.VarLenFeature(tf.float32),
        'step_metadata': tf.io.FixedLenFeature([], tf.string, default_value=b''),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Convert sparse features to dense
    parsed['observation_command'] = tf.sparse.to_dense(parsed['observation_command'])
    parsed['action_command'] = tf.sparse.to_dense(parsed['action_command'])
    
    return parsed


def _deserialize_observation(obs_bytes: bytes) -> Dict[str, Any]:
    """Deserialize observation from JSON bytes."""
    return json.loads(obs_bytes.decode('utf-8'))


def _deserialize_action(action_bytes: bytes) -> Dict[str, Any]:
    """Deserialize action from JSON bytes."""
    return json.loads(action_bytes.decode('utf-8'))


def _extract_pose_vector(pose: Any) -> List[float]:
    if not isinstance(pose, dict):
        return []
    out: List[float] = []
    pos = pose.get("position")
    if isinstance(pos, (list, tuple)):
        out.extend([float(x) for x in pos])
    quat = pose.get("orientation_quat")
    if isinstance(quat, (list, tuple)):
        out.extend([float(x) for x in quat])
    return out


def _extract_image_vector(img: Any, target_dim: int) -> np.ndarray:
    try:
        arr = np.array(img, dtype=np.float32).flatten()
        if arr.size > target_dim:
            return arr[:target_dim]
        if arr.size < target_dim:
            return np.pad(arr, (0, target_dim - arr.size), mode="constant")
        return arr
    except Exception:
        return np.zeros(target_dim, dtype=np.float32)


def _extract_obs_vector(obs: Dict[str, Any], obs_dim: int, feature_spec: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Extract observation vector from observation dictionary.
    
    This is a simplified extractor. In production, you'd want to handle
    the full observation structure (poses, robot state, etc.).
    """
    def pad(vec: np.ndarray) -> np.ndarray:
        if len(vec) < obs_dim:
            vec = np.pad(vec, (0, obs_dim - len(vec)), mode="constant")
        elif len(vec) > obs_dim:
            vec = vec[:obs_dim]
        return vec.astype(np.float32)

    feature_spec = feature_spec or {}
    image_keys = feature_spec.get("image_keys", [])
    image_dim = feature_spec.get("image_dim", obs_dim)
    pose_keys = feature_spec.get("pose_keys", [])

    # Images (flatten)
    for key in image_keys:
        if isinstance(obs, dict) and key in obs:
            return pad(_extract_image_vector(obs[key], image_dim))

    # Poses
    for key in pose_keys:
        if isinstance(obs, dict) and key in obs:
            pose_vec = _extract_pose_vector(obs[key])
            if pose_vec:
                return pad(np.array(pose_vec, dtype=np.float32))

    # Try to extract command vector if available
    if isinstance(obs, dict) and "command" in obs:
        cmd = obs["command"]
        if isinstance(cmd, (list, np.ndarray)):
            return pad(np.array(cmd, dtype=np.float32))

    # Fallback: robot_state.joint_positions
    if isinstance(obs, dict) and "robot_state" in obs and isinstance(obs["robot_state"], dict):
        joints = obs["robot_state"].get("joint_positions")
        if isinstance(joints, (list, np.ndarray)):
            return pad(np.array(joints, dtype=np.float32))

    # Generic fallback: flatten numeric values in dict (depth-first)
    flat: List[float] = []

    def _flatten(val: Any):
        if len(flat) >= obs_dim:
            return
        if isinstance(val, (int, float, np.floating, np.integer)):
            flat.append(float(val))
        elif isinstance(val, (list, tuple)):
            for v in val:
                _flatten(v)
        elif isinstance(val, dict):
            for v in val.values():
                _flatten(v)

    _flatten(obs)
    if flat:
        return pad(np.array(flat, dtype=np.float32))

    # Default: zeros
    return np.zeros(obs_dim, dtype=np.float32)


def _extract_action_vector(action: Dict[str, Any], action_dim: int) -> np.ndarray:
    """
    Extract action vector from action dictionary.
    """
    def _hash_to_vec(text: str, dim: int) -> np.ndarray:
        if dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        reps = int(np.ceil(dim / raw.size))
        tiled = np.tile(raw, reps)[:dim]
        # map to [-1, 1]
        return (tiled / 127.5 - 1.0).astype(np.float32)

    base = None
    if "command" in action:
        cmd = action["command"]
        if isinstance(cmd, (list, np.ndarray)):
            base = np.array(cmd, dtype=np.float32)

    if base is None:
        base = np.zeros(action_dim, dtype=np.float32)

    # Pad/truncate base to action_dim
    if len(base) < action_dim:
        base = np.pad(base, (0, action_dim - len(base)), mode="constant")
    elif len(base) > action_dim:
        base = base[:action_dim]

    # Optional "imagination" targets: encode planner/tool traces into tail dims.
    # This lets the current WaveCore training (MSE on action vector) learn to predict
    # symbolic-plan intent without changing model outputs.
    reserve = min(16, action_dim)
    if reserve > 0:
        plan_k = reserve // 2
        tool_k = reserve - plan_k
        tail = np.zeros((reserve,), dtype=np.float32)

        planner = action.get("planner")
        if isinstance(planner, dict) and plan_k > 0:
            intent = str(planner.get("intent") or "")
            steps = planner.get("plan_steps") or []
            if isinstance(steps, (list, tuple)):
                steps_text = " | ".join([str(s) for s in steps][:16])
            else:
                steps_text = str(steps)
            plan_text = f"intent:{intent}\nsteps:{steps_text}"
            tail[:plan_k] = _hash_to_vec(plan_text, plan_k)

        tool_calls = action.get("tool_calls")
        if isinstance(tool_calls, list) and tool_k > 0:
            names = []
            for tc in tool_calls[:16]:
                if isinstance(tc, dict):
                    names.append(str(tc.get("name") or tc.get("tool") or ""))
                else:
                    names.append(str(tc))
            tool_text = "tools:" + ",".join(names)
            tail[plan_k:] = _hash_to_vec(tool_text, tool_k)

        # Write into the last `reserve` dims
        base[-reserve:] = tail
    
    # Default: return zeros
    return base


def create_tfrecord_dataset(
    tfrecord_paths: Union[str, Path, List[Union[str, Path]]],
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    repeat: bool = True,
    prefetch: int = 4,
    obs_dim: int = 128,
    action_dim: int = 32,
    compression_type: Optional[str] = "GZIP",
    feature_spec: Optional[Dict[str, Any]] = None,
) -> "tf.data.Dataset":
    """
    Create a TensorFlow dataset from TFRecord files.
    
    Args:
        tfrecord_paths: Path(s) to TFRecord file(s) or directory
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        repeat: Whether to repeat the dataset
        prefetch: Number of batches to prefetch
        obs_dim: Observation dimension
        action_dim: Action dimension
        compression_type: Compression type ("GZIP" or None)
    
    Returns:
        TensorFlow dataset yielding batches
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for dataset loading")

    import tensorflow as tf  # type: ignore
    
    # Convert to list of paths
    if isinstance(tfrecord_paths, (str, Path)):
        tfrecord_paths = [Path(tfrecord_paths)]
    else:
        tfrecord_paths = [Path(p) for p in tfrecord_paths]
    
    # Expand directories
    expanded_paths = []
    for path in tfrecord_paths:
        if path.is_dir():
            expanded_paths.extend(path.glob("*.tfrecord*"))
        else:
            expanded_paths.append(path)
    
    if not expanded_paths:
        raise ValueError("No TFRecord files found")
    
    # Create dataset
    dataset = tf.data.TFRecordDataset(
        [str(p) for p in expanded_paths],
        compression_type=compression_type,
    )
    
    # Parse examples
    dataset = dataset.map(_parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Extract and convert to arrays
    def extract_features(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def _obs_bytes_to_vec(x):
            obs = _deserialize_observation(x)
            return _extract_obs_vector(obs, obs_dim, feature_spec)

        def _action_bytes_to_vec(x):
            action = _deserialize_action(x)
            return _extract_action_vector(action, action_dim)

        obs_vec = tf.py_function(_obs_bytes_to_vec, inp=[example["observation"]], Tout=tf.float32)
        action_vec = tf.py_function(_action_bytes_to_vec, inp=[example["action"]], Tout=tf.float32)

        # For backward compatibility, fall back to command vectors if present
        obs_vec = tf.cond(
            tf.shape(obs_vec)[0] > 0,
            lambda: obs_vec,
            lambda: tf.pad(example["observation_command"], [[0, tf.maximum(0, obs_dim - tf.shape(example["observation_command"])[0])]])[
                :obs_dim
            ],
        )
        action_vec = tf.cond(
            tf.shape(action_vec)[0] > 0,
            lambda: action_vec,
            lambda: tf.pad(example["action_command"], [[0, tf.maximum(0, action_dim - tf.shape(example["action_command"])[0])]])[
                :action_dim
            ],
        )

        # Pad/truncate to fixed dimensions (safety)
        obs_vec = tf.pad(obs_vec, [[0, tf.maximum(0, obs_dim - tf.shape(obs_vec)[0])]])[:obs_dim]
        action_vec = tf.pad(action_vec, [[0, tf.maximum(0, action_dim - tf.shape(action_vec)[0])]])[:action_dim]

        return {
            'obs': obs_vec,
            'action': action_vec,
            'reward': example['reward'],
            'done': tf.cast(example['is_terminal'], tf.bool),
            'episode_id': example['episode_id'],
            'step_index': example['step_index'],
        }
    
    dataset = dataset.map(extract_features, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    
    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Repeat
    if repeat:
        dataset = dataset.repeat()
    
    # Prefetch
    dataset = dataset.prefetch(prefetch)
    
    return dataset


def tf_dataset_to_jax_iterator(tf_dataset: "tf.data.Dataset") -> Iterator[Dict[str, "jnp.ndarray"]]:
    """
    Convert TensorFlow dataset to JAX-compatible iterator.
    
    Args:
        tf_dataset: TensorFlow dataset
    
    Yields:
        Batches as dictionaries of JAX arrays
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for dataset iteration")

    import tensorflow as tf  # type: ignore
    
    for batch in tf_dataset:
        # Convert TensorFlow tensors to JAX arrays
        jax_batch = {}
        for key, value in batch.items():
            if isinstance(value, tf.Tensor):
                # Convert to numpy then to JAX array
                numpy_value = value.numpy()
                jax_batch[key] = jnp.array(numpy_value)
            else:
                jax_batch[key] = value
        
        yield jax_batch


def load_rlds_dataset(
    tfrecord_paths: Union[str, Path, List[Union[str, Path]]],
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    repeat: bool = True,
    prefetch: int = 4,
    obs_dim: int = 128,
    action_dim: int = 32,
    compression_type: Optional[str] = "GZIP",
    feature_spec: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """
    Load RLDS dataset and return JAX-compatible iterator.
    
    This is a convenience function that combines create_tfrecord_dataset
    and tf_dataset_to_jax_iterator.
    
    Args:
        tfrecord_paths: Path(s) to TFRecord file(s) or directory
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        repeat: Whether to repeat the dataset
        prefetch: Number of batches to prefetch
        obs_dim: Observation dimension
        action_dim: Action dimension
        compression_type: Compression type ("GZIP" or None)
    
    Yields:
        Batches as dictionaries of JAX arrays
    """
    tf_dataset = create_tfrecord_dataset(
        tfrecord_paths=tfrecord_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        repeat=repeat,
        prefetch=prefetch,
        obs_dim=obs_dim,
        action_dim=action_dim,
        compression_type=compression_type,
        feature_spec=feature_spec,
    )
    
    return tf_dataset_to_jax_iterator(tf_dataset)

