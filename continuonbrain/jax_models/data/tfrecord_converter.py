"""
TFRecord Converter

Convert JSON/JSONL episodes to TFRecord format optimized for TPU ingestion.
Validates schema against proto definitions.
"""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
import numpy as np
import argparse

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Convert bytes to TFRecord feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
    """Convert float to TFRecord feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(value: List[float]) -> tf.train.Feature:
    """Convert list of floats to TFRecord feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: int) -> tf.train.Feature:
    """Convert int to TFRecord feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value: List[int]) -> tf.train.Feature:
    """Convert list of ints to TFRecord feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _string_feature(value: str) -> tf.train.Feature:
    """Convert string to TFRecord feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def _string_list_feature(value: List[str]) -> tf.train.Feature:
    """Convert list of strings to TFRecord feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf-8') for v in value]))


def _serialize_observation(obs: Dict[str, Any]) -> bytes:
    """
    Serialize observation dictionary to bytes.
    
    For now, we'll use JSON serialization. In production, you might want
    to use protobuf serialization matching the proto schema.
    """
    return json.dumps(obs, sort_keys=True).encode('utf-8')


def _serialize_action(action: Dict[str, Any]) -> bytes:
    """Serialize action dictionary to bytes."""
    return json.dumps(action, sort_keys=True).encode('utf-8')


def _load_json_episode(episode_path: Path) -> Dict[str, Any]:
    """Load a JSON episode file."""
    with episode_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_episode(episode_path: Path) -> Iterator[Dict[str, Any]]:
    """Load a JSONL episode file, yielding steps."""
    with episode_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def convert_step_to_tfrecord(
    step: Dict[str, Any],
    episode_id: str,
    step_index: int,
) -> tf.train.Example:
    """
    Convert a single step to TFRecord Example.
    
    Args:
        step: Step dictionary with keys: observation, action, reward, is_terminal, step_metadata
        episode_id: Episode identifier
        step_index: Step index within episode
    
    Returns:
        TFRecord Example
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for TFRecord conversion")
    
    obs = step.get("observation", {})
    action = step.get("action", {})
    reward = step.get("reward", 0.0)
    is_terminal = step.get("is_terminal", False)
    step_metadata = step.get("step_metadata", {})
    
    # Extract common fields
    # For observation: extract command vector if available, or serialize full obs
    obs_command = obs.get("command", [])
    if isinstance(obs_command, (list, np.ndarray)):
        obs_command = [float(x) for x in obs_command]
    else:
        obs_command = []
    
    # For action: extract command vector
    action_command = action.get("command", [])
    if isinstance(action_command, (list, np.ndarray)):
        action_command = [float(x) for x in action_command]
    else:
        action_command = []
    
    # Serialize full observation and action as JSON bytes
    obs_bytes = _serialize_observation(obs)
    action_bytes = _serialize_action(action)
    
    # Build feature dictionary
    feature = {
        'episode_id': _string_feature(episode_id),
        'step_index': _int64_feature(step_index),
        'reward': _float_feature(float(reward)),
        'is_terminal': _int64_feature(1 if is_terminal else 0),
        'observation': _bytes_feature(obs_bytes),
        'action': _bytes_feature(action_bytes),
        'observation_command': _float_list_feature(obs_command),
        'action_command': _float_list_feature(action_command),
    }
    
    # Add step metadata
    if step_metadata:
        metadata_str = json.dumps(step_metadata, sort_keys=True)
        feature['step_metadata'] = _bytes_feature(metadata_str.encode('utf-8'))
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_episode_to_tfrecord(
    episode_path: Path,
    output_path: Optional[Path] = None,
    compress: bool = True,
) -> Path:
    """
    Convert a single episode (JSON or JSONL) to TFRecord format.
    
    Args:
        episode_path: Path to input episode file (.json or .jsonl)
        output_path: Path to output TFRecord file (auto-generated if None)
        compress: Whether to use GZIP compression
    
    Returns:
        Path to output TFRecord file
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for TFRecord conversion")
    
    # Determine output path
    if output_path is None:
        output_path = episode_path.with_suffix('.tfrecord')
        if compress:
            output_path = output_path.with_suffix('.tfrecord.gz')
    
    # Load episode
    if episode_path.suffix == ".json":
        episode_data = _load_json_episode(episode_path)
        metadata = episode_data.get("metadata", {})
        steps = episode_data.get("steps", [])
        episode_id = metadata.get("episode_id", episode_path.stem)
    elif episode_path.suffix == ".jsonl":
        steps = list(_load_jsonl_episode(episode_path))
        # Extract episode_id from first step
        episode_id = steps[0].get("episode_id", episode_path.stem) if steps else episode_path.stem
    else:
        raise ValueError(f"Unsupported file format: {episode_path.suffix}")
    step_count = len(steps)
    print(
        f"Converting {episode_path.name} ({step_count} steps) -> {Path(output_path).name if output_path else 'auto'}",
        flush=True,
    )
    
    # Write TFRecord
    open_fn = gzip.open if compress else open
    mode = 'wb' if compress else 'w'
    
    with open_fn(output_path, mode) as f:
        # TFRecordWriter expects a string path; Path objects trigger a TypeError on Windows.
        writer = tf.io.TFRecordWriter(f.name if compress else str(output_path))
        
        for step_index, step in enumerate(steps):
            example = convert_step_to_tfrecord(step, episode_id, step_index)
            writer.write(example.SerializeToString())
            if step_index and step_index % 500 == 0:
                print(f"  wrote {step_index}/{step_count} steps...", flush=True)
        
        writer.close()
    
    print(f"  done {episode_path.name}: {step_count} steps", flush=True)
    
    return output_path


def convert_directory_to_tfrecord(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    compress: bool = True,
    pattern: str = "*.{json,jsonl}",
) -> List[Path]:
    """
    Convert all episodes in a directory (or a single episode file) to TFRecord.
    
    Args:
        input_dir: Directory containing episodes or a single .json/.jsonl file
        output_dir: Output directory (defaults to input_dir / "tfrecord")
        compress: Whether to use GZIP compression
        pattern: Glob pattern for episode files (directory mode only)
    
    Returns:
        List of output TFRecord file paths
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for TFRecord conversion")
    
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir.parent / "tfrecord" if input_dir.is_file() else input_dir / "tfrecord"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve episode files (single file or directory glob)
    if input_dir.is_file():
        if input_dir.suffix not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported file format: {input_dir}")
        episode_files = [input_dir]
    else:
        json_files = list(input_dir.glob("*.json"))
        jsonl_files = list(input_dir.glob("*.jsonl"))
        episode_files = sorted(json_files + jsonl_files)
    
    if not episode_files:
        print(f"No episode files found in {input_dir}", flush=True)
        return []
    
    total = len(episode_files)
    print(f"Found {total} episode files in {input_dir}, writing to {output_dir}", flush=True)
    
    output_paths = []
    for idx, episode_file in enumerate(episode_files, start=1):
        output_file = output_dir / f"{episode_file.stem}.tfrecord"
        if compress:
            output_file = output_file.with_suffix('.tfrecord.gz')
        
        print(f"[{idx}/{total}] {episode_file.name} -> {output_file.name}", flush=True)
        
        try:
            converted_path = convert_episode_to_tfrecord(
                episode_file,
                output_file,
                compress=compress,
            )
            output_paths.append(converted_path)
        except Exception as e:
            print(f"Error converting {episode_file}: {e}", flush=True)
            continue
    
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JSON/JSONL episodes to TFRecord")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing episodes or a single .json/.jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for TFRecords (defaults to <input>/tfrecord or sibling when input is a file)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable GZIP compression (default: compress)",
    )
    args = parser.parse_args()

    compress = not args.no_compress
    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    outputs = convert_directory_to_tfrecord(
        input_path,
        output_dir=output_dir,
        compress=compress,
    )
    if not outputs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def validate_tfrecord_schema(tfrecord_path: Path) -> bool:
    """
    Validate TFRecord file against expected schema.
    
    Args:
        tfrecord_path: Path to TFRecord file
    
    Returns:
        True if valid, False otherwise
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for TFRecord validation")
    
    required_features = [
        'episode_id',
        'step_index',
        'reward',
        'is_terminal',
        'observation',
        'action',
    ]
    
    try:
        # Open file (handle GZIP)
        if tfrecord_path.suffix == '.gz':
            dataset = tf.data.TFRecordDataset([str(tfrecord_path)], compression_type='GZIP')
        else:
            dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
        
        # Read first example
        for raw_record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            features = example.features.feature
            
            # Check required features
            for feature_name in required_features:
                if feature_name not in features:
                    print(f"Missing required feature: {feature_name}")
                    return False
            
            return True
        
        # No examples found
        print("No examples found in TFRecord file")
        return False
        
    except Exception as e:
        print(f"Error validating TFRecord: {e}")
        return False

