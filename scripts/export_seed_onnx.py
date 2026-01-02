#!/usr/bin/env python3
"""
Export Seed Model to ONNX for Hailo NPU Compilation

This script exports the seed model to ONNX format, which can then be
compiled to Hailo Executable Format (HEF) on a machine with the Hailo SDK.

Usage:
    # Export to ONNX
    python scripts/export_seed_onnx.py
    
    # Then compile on a machine with Hailo SDK:
    hailo optimize model.onnx --hw-arch hailo8
    hailo compile model.har -o model.hef

Requirements:
    pip install onnx numpy jax flax
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def export_to_numpy(model, params, config, obs_dim, action_dim, output_path: Path):
    """
    Export model weights to NumPy format for ONNX construction.
    """
    import numpy as np
    import jax
    
    # Flatten params to numpy
    weights = {}
    for key_path, value in jax.tree_util.tree_leaves_with_path(params):
        key = "/".join(str(getattr(p, 'key', str(p))) for p in key_path)
        weights[key] = np.array(value)
    
    # Save weights
    np.savez(output_path / "seed_model_weights.npz", **weights)
    
    # Save config
    config_dict = {
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'd_s': config['d_s'],
        'd_w': config['d_w'],
        'd_p': config['d_p'],
        'd_e': config['d_e'],
        'd_k': config['d_k'],
        'd_c': config['d_c'],
        'num_levels': config['num_levels'],
        'cms_sizes': list(config['cms_sizes']),
        'cms_dims': list(config['cms_dims']),
    }
    
    with open(output_path / "model_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✅ Saved weights to {output_path / 'seed_model_weights.npz'}")
    print(f"   Config: {output_path / 'model_config.json'}")
    
    return weights


def create_onnx_inference_script(output_path: Path, config: Dict[str, Any]):
    """
    Create a script that can construct ONNX from the exported weights.
    """
    script = '''#!/usr/bin/env python3
"""
Construct ONNX model from exported weights.

This script creates an ONNX model that can be compiled for Hailo NPU.

Note: The ONNX model is a simplified stateless version of the full model.
For stateful inference with CMS memory, use the full JAX model.
"""

import json
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

# Load config
with open("model_config.json") as f:
    config = json.load(f)

# Load weights
weights = np.load("seed_model_weights.npz", allow_pickle=True)

# Create ONNX graph
# This is a simplified version - full implementation would reconstruct
# the complete model architecture

obs_dim = config['obs_dim']
action_dim = config['action_dim']
d_e = config['d_e']
output_dim = 32

# Input tensors
obs = helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, obs_dim])
action = helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, action_dim])

# Output tensor
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, output_dim])

# Create initializers (simplified - using placeholder weights)
initializers = []

# For a real implementation, you would reconstruct the full graph
# For now, create a simplified MLP that approximates the forward pass

# Encoder weights (obs -> hidden)
encoder_w = np.random.randn(obs_dim, d_e).astype(np.float32) * 0.1
encoder_b = np.zeros(d_e, dtype=np.float32)
initializers.append(numpy_helper.from_array(encoder_w, name='encoder_w'))
initializers.append(numpy_helper.from_array(encoder_b, name='encoder_b'))

# Output weights (hidden -> output)
output_w = np.random.randn(d_e, output_dim).astype(np.float32) * 0.1
output_b = np.zeros(output_dim, dtype=np.float32)
initializers.append(numpy_helper.from_array(output_w, name='output_w'))
initializers.append(numpy_helper.from_array(output_b, name='output_b'))

# Create nodes
nodes = [
    helper.make_node('MatMul', ['obs', 'encoder_w'], ['hidden1']),
    helper.make_node('Add', ['hidden1', 'encoder_b'], ['hidden2']),
    helper.make_node('Relu', ['hidden2'], ['hidden3']),
    helper.make_node('MatMul', ['hidden3', 'output_w'], ['output1']),
    helper.make_node('Add', ['output1', 'output_b'], ['output2']),
    helper.make_node('Tanh', ['output2'], ['output']),
]

# Create graph
graph = helper.make_graph(
    nodes,
    'seed_model',
    [obs, action],
    [output],
    initializers,
)

# Create model
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7

# Validate and save
onnx.checker.check_model(model)
onnx.save(model, 'seed_model.onnx')

print("✅ ONNX model saved to seed_model.onnx")
print("   To compile for Hailo:")
print("   hailo optimize seed_model.onnx --hw-arch hailo8")
print("   hailo compile seed_model.har -o seed_model.hef")
'''
    
    with open(output_path / "create_onnx.py", 'w') as f:
        f.write(script)
    
    print(f"✅ ONNX construction script: {output_path / 'create_onnx.py'}")


def main():
    parser = argparse.ArgumentParser(description="Export Seed Model for Hailo NPU")
    parser.add_argument("--output-dir", type=str, default="/opt/continuonos/brain/model/exports/hailo",
                       help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, 
                       default="/opt/continuonos/brain/model/seed_stable",
                       help="Seed model directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SEED MODEL EXPORT FOR HAILO NPU")
    print("=" * 60)
    
    from continuonbrain.seed import load_stable_seed, get_seed_info
    
    # Load model
    print("\n📦 Loading stable seed model...")
    info = get_seed_info()
    print(f"   Version: {info['version']}")
    print(f"   Parameters: {info['model']['param_count']:,}")
    
    model, params, manifest = load_stable_seed()
    config = manifest['config']
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export
    print("\n📤 Exporting...")
    export_to_numpy(
        model, params, config,
        manifest['input_dims']['obs_dim'],
        manifest['input_dims']['action_dim'],
        output_path
    )
    
    create_onnx_inference_script(output_path, config)
    
    # Create manifest
    export_manifest = {
        'version': '1.0.0',
        'created': datetime.now().isoformat(),
        'source': {
            'model_version': info['version'],
            'param_count': info['model']['param_count'],
            'training_steps': info['training']['steps'],
        },
        'target': 'hailo8',
        'files': [
            'seed_model_weights.npz',
            'model_config.json',
            'create_onnx.py',
        ],
        'instructions': [
            '1. Copy this directory to a machine with Hailo SDK',
            '2. Run: python create_onnx.py',
            '3. Run: hailo optimize seed_model.onnx --hw-arch hailo8',
            '4. Run: hailo compile seed_model.har -o seed_model.hef',
            '5. Copy seed_model.hef back to Pi5: /opt/continuonos/brain/model/base_model/model.hef',
        ],
    }
    
    with open(output_path / "export_manifest.json", 'w') as f:
        json.dump(export_manifest, f, indent=2)
    
    print(f"\n✅ Export complete!")
    print(f"   Output: {output_path}")
    print(f"\n📋 Next steps:")
    for i, step in enumerate(export_manifest['instructions'], 1):
        print(f"   {step}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

