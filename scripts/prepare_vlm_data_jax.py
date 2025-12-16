
import os
import argparse
import tensorflow as tf
from datasets import load_dataset
import numpy as np
import json
from pathlib import Path
from PIL import Image
import io

def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

def create_bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf-8') if isinstance(v, str) else v for v in values]))

def process_image(image_obj, target_size=(64, 64)):
    """Resize and flatten image."""
    try:
        if isinstance(image_obj, dict) and 'bytes' in image_obj:
            image = Image.open(io.BytesIO(image_obj['bytes']))
        elif isinstance(image_obj, Image.Image):
            image = image_obj
        else:
            return None
            
        image = image.convert('RGB')
        image = image.resize(target_size)
        # Normalize to 0-1
        arr = np.array(image, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

def convert_to_tfrecord(dataset, output_dir, max_samples=1000):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / "vlm_data.tfrecord"
    
    print(f"Converting VLM dataset to {file_path}...")
    
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(str(file_path), options=options)
    
    count = 0
    for item in dataset:
        if count >= max_samples:
            break
        
        # Extract features
        # Dataset fields: 'image', 'text', 'category', etc.
        image_data = item.get('image')
        text = item.get('prediction', item.get('text', '')) # 'prediction' is target in some datasets
        
        image_flat = process_image(image_data)
        if image_flat is None:
            continue
            
        # Create observation dict with image data
        # We store flattened image in 'image_rgb' key
        obs_dict = {
            "image_rgb": image_flat,
            "robot_state": {"joint_positions": [0.0] * 7} # Dummy robot state
        }
        obs_bytes = json.dumps(obs_dict).encode('utf-8')
        
        # Create action dict with text
        # In a real VLM, this text is the target generation
        action_dict = {"command": [0.0] * 32, "text": text} # Include raw text for debugging/hybrid
        action_bytes = json.dumps(action_dict).encode('utf-8')
        
        feature = {
            'is_first': create_int_feature([True]),
            'is_last': create_int_feature([True]),
            'is_terminal': create_int_feature([True]),
            'reward': create_float_feature([0.0]),
            'discount': create_float_feature([1.0]),
            'observation': create_bytes_feature([obs_bytes]),
            'action': create_bytes_feature([action_bytes]),
            
            'episode_id': create_bytes_feature([f"vlm_{count}".encode('utf-8')]),
            'step_index': create_int_feature([0]),
            'step_metadata': create_bytes_feature([text[:200].encode('utf-8')]),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        count += 1
        
        if count % 10 == 0:
            print(f"Processed {count} samples...")
        
    writer.close()
    print(f"Constructed {count} VLM samples.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/opt/continuonos/brain/rlds/vlm_tfrecord")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()
    
    print("Loading philschmid/amazon-product-descriptions-vlm...")
    try:
        # Load dataset
        # Note: 'train' split. Disable streaming for stability on this env.
        # Taking a small slice to ensure it fits in memory/time constraints.
        dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split=f"train[:{args.max_samples}]", streaming=False)
        
        convert_to_tfrecord(dataset, args.output_dir, args.max_samples)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
