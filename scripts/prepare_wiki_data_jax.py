
import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
import numpy as np
import json
from pathlib import Path

def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

def create_bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf-8') if isinstance(v, str) else v for v in values]))

def convert_to_tfrecord(dataset, output_dir, max_samples=1000):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / "wiki_data.tfrecord"
    
    print(f"Converting dataset to {file_path}...")
    
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(str(file_path), options=options)
    
    count = 0
    for item in dataset:
        if count >= max_samples:
            break
            
        # Map Wikipedia content to "Action" (generation target)
        # We'll treat the title as "Context/Observation" and text as "Action"
        # This is a simplification for the RL-based JAX pipeline.
        
        text = item.get('text', '')
        title = item.get('title', '') # Some datasets have title
        
        # Simple tokenization mock (in real usage, use the model's tokenizer)
        # Here we just store raw bytes or simple mock embeddings
        
        # Create a mock observation (e.g. 128 float vector)
        # The loader expects a JSON string in the 'observation' feature
        obs_dict = {"robot_state": {"joint_positions": np.random.rand(128).astype(np.float32).tolist()}}
        obs_bytes = json.dumps(obs_dict).encode('utf-8')
        
        # Create a mock action
        # The loader expects a JSON string in the 'action' feature
        action_dict = {"command": np.random.rand(32).astype(np.float32).tolist()}
        action_bytes = json.dumps(action_dict).encode('utf-8')
        
        feature = {
            'is_first': create_int_feature([True]), 
            'is_last': create_int_feature([True]),
            'is_terminal': create_int_feature([True]),
            'reward': create_float_feature([0.0]),
            'discount': create_float_feature([1.0]),
            'observation': create_bytes_feature([obs_bytes]), # Correct key is 'observation', NOT 'obs'
            'action': create_bytes_feature([action_bytes]),   # Correct key is 'action', type bytes
            
            # Additional fields required by parse_tfrecord_example or helpful
            'episode_id': create_bytes_feature([str(count).encode('utf-8')]),
            'step_index': create_int_feature([0]),
            
            # Text specific
            'step_metadata': create_bytes_feature([text[:200].encode('utf-8') if text else b""]),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        count += 1
        
    writer.close()
    print(f"Constructed {count} samples.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/opt/continuonos/brain/rlds/wiki_tfrecord")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    
    print("Loading rahular/simple-wikipedia...")
    try:
        # Load small slice
        dataset = load_dataset("rahular/simple-wikipedia", split="train", streaming=True)
        
        convert_to_tfrecord(dataset, args.output_dir, args.max_samples)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
