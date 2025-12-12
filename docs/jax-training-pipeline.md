# JAX Training Pipeline for Cloud TPU Production

This document describes the JAX-based training pipeline for ContinuonBrain that enables development on Raspberry Pi and production training on Google Cloud TPUs.

## Overview

The JAX training pipeline provides:

- **Pi Development**: Model authoring, sanity checks, and RLDS data capture
- **Cloud TPU Training**: Production-scale training on Google Cloud TPUs
- **Dual Inference**: JAX CPU inference and Hailo-compiled models with auto-detection
- **TFRecord Data Format**: Optimized for TPU ingestion

This pipeline coexists with the existing PyTorch trainer (`local_lora_trainer.py`) and is automatically selected based on hardware capabilities (TPU/AI HAT detection).

## Architecture

### CoreModel (v0 Seed Model)

The `CoreModel` is the canonical model implementation in JAX/Flax, inspired by the HOPE architecture:

- **Fast state (s_t)**: Low-level reactive state
- **Wave state (w_t)**: SSM-like global coordination
- **Particle state (p_t)**: Local nonlinear dynamics
- **CMS memory**: Hierarchical memory levels

The same model code runs identically on Pi (sanity checks) and cloud TPU (full training).

**Location**: `continuonbrain/jax_models/core_model.py`

### Data Pipeline

1. **Episode Logger** (`data/episode_logger.py`): Captures RLDS episodes on Pi in JSON/JSONL format
2. **TFRecord Converter** (`data/tfrecord_converter.py`): Converts JSON/JSONL → TFRecord for TPU ingestion
3. **Dataset Loader** (`data/rlds_dataset.py`): Loads TFRecord → JAX arrays with batching and prefetching

### Training

1. **Local Sanity Check** (`train/local_sanity_check.py`): Tiny training loop on Pi CPU to verify model correctness
2. **TPU Training** (`train/cloud/tpu_train.py`): Full training loop for Google Cloud TPU

### Export & Inference

1. **Checkpointing** (`export/checkpointing.py`): Save/load checkpoints using Orbax (supports GCS)
2. **JAX CPU Export** (`export/export_jax.py`): Export for JAX CPU inference on Pi
3. **Hailo Export** (`export/export_hailo.py`): Export pipeline: JAX → TF → ONNX → Hailo compiler
4. **Inference Router** (`export/inference_router.py`): Hardware-aware inference with fallback chain

## Quick Start

### 1. Install Dependencies

```bash
# JAX CPU (for Pi)
pip install "jax[cpu]" jaxlib flax optax

# For TPU training (on cloud)
pip install "jax[tpu]" orbax-checkpoint

# For data pipeline
pip install tensorflow tensorflow-datasets

# For GCS upload
pip install google-cloud-storage
```

### 2. Capture Episodes on Pi

```python
from continuonbrain.jax_models.data.episode_logger import EpisodeLogger

logger = EpisodeLogger("/opt/continuonos/brain/rlds/episodes", format="jsonl")

episode_id = logger.start_episode(
    environment_id="pi5-dev",
    xr_mode="trainer",
)

# Log steps
for step in range(100):
    logger.log_step(
        obs={"command": [0.1, 0.2, ...]},
        action={"command": [0.15, 0.25, ...]},
        reward=0.5,
        done=False,
    )

logger.end_episode()
```

### 3. Convert to TFRecord (dir or single file)

```python
from continuonbrain.jax_models.data.tfrecord_converter import convert_directory_to_tfrecord

# Directory mode (scans for *.json / *.jsonl)
convert_directory_to_tfrecord(
    input_dir="/opt/continuonos/brain/rlds/episodes",
    output_dir="/opt/continuonos/brain/rlds/tfrecord",
    compress=True,
)

# Single-file mode (converter accepts file paths)
convert_directory_to_tfrecord(
    input_dir="/opt/continuonos/brain/rlds/episodes/episode.json",
    output_dir="/opt/continuonos/brain/rlds/tfrecord",
    compress=True,
)
```

### 4. Run Sanity Check on Pi

```bash
python -m continuonbrain.jax_models.train.local_sanity_check \
    --rlds-dir /opt/continuonos/brain/rlds/tfrecord \
    --obs-dim 128 \
    --action-dim 32 \
    --max-steps 10 \
    --batch-size 4
```

### 5. Upload to GCS

```bash
python -m continuonbrain.jax_models.utils.upload_to_gcs \
    --local-dir /opt/continuonos/brain/rlds/tfrecord \
    --gcs-bucket continuon-rlds \
    --gcs-prefix rlds/episodes
```

### 6. Train on TPU

```bash
python -m continuonbrain.jax_models.train.cloud.tpu_train \
    --data-path gs://continuon-rlds/rlds/episodes \
    --output-dir gs://continuon-rlds/checkpoints/core_model_v0 \
    --batch-size 256 \
    --num-steps 10000 \
    --learning-rate 1e-4
```

### 7. Export for Inference

```bash
# Export for JAX CPU
python -m continuonbrain.jax_models.export.export_jax \
    --checkpoint-path gs://continuon-rlds/checkpoints/core_model_v0 \
    --output-path ./models/core_model_inference \
    --quantization fp16

# Export for Hailo (optional)
python -m continuonbrain.jax_models.export.export_hailo \
    --checkpoint-path ./models/core_model_inference \
    --output-dir ./models/core_model_hailo
```

### 8. Orchestrate on Pi (health → TFRecord → local train → export)

```bash
python -m continuonbrain.services.training_manager \
  --episodes-dir /opt/continuonos/brain/rlds/episodes \
  --tfrecord-dir /opt/continuonos/brain/rlds/tfrecord \
  --health \
  --convert-tfrecord \
  --train-local \
  --trainer-data-path /opt/continuonos/brain/rlds/tfrecord \
  --export
```

## Hardware Detection

The pipeline automatically detects available hardware:

```python
from continuonbrain.jax_models.train.hardware_detector_jax import detect_jax_backends

info = detect_jax_backends()
print(f"TPU available: {info['backends']['tpu']['available']}")
print(f"Hailo available: {info['backends']['hailo']['available']}")
print(f"Best for training: {info['best_for_training']}")
print(f"Best for inference: {info['best_for_inference']}")
```

## Trainer Selection

The trainer selector automatically chooses between PyTorch and JAX:

```python
from continuonbrain.jax_models.utils.trainer_selector import select_trainer

trainer_type = select_trainer(prefer_jax=True)
print(f"Selected trainer: {trainer_type}")
```

## Configuration

Model configurations are provided for different deployment scenarios:

- `CoreModelConfig.pi5_optimized()`: Optimized for Raspberry Pi 5
- `CoreModelConfig.development()`: Small sizes for fast iteration
- `CoreModelConfig.tpu_optimized()`: Large dimensions for TPU training

## Integration with Existing Pipeline

The JAX trainer integrates with the existing training infrastructure:

- Uses the same RLDS episode format
- Compatible with existing episode capture code
- Can be selected automatically via `trainer_selector`
- Shares the same checkpoint format (with Orbax)

## File Structure

```
continuonbrain/jax_models/
├── __init__.py
├── core_model.py              # CoreModel (Flax nn.Module)
├── config.py                  # CoreModelConfig
├── data/
│   ├── episode_logger.py     # RLDS capture on Pi
│   ├── tfrecord_converter.py # JSON/JSONL → TFRecord
│   └── rlds_dataset.py       # TFRecord → JAX arrays
├── train/
│   ├── local_sanity_check.py # Pi CPU sanity training
│   ├── hardware_detector_jax.py
│   └── cloud/
│       └── tpu_train.py      # Cloud TPU training
├── export/
│   ├── checkpointing.py      # Orbax checkpointing
│   ├── export_jax.py         # JAX CPU export
│   ├── export_hailo.py       # Hailo export pipeline
│   └── inference_router.py   # Hardware-aware inference
└── utils/
    ├── trainer_selector.py   # PyTorch vs JAX selection
    └── upload_to_gcs.py      # RLDS → GCS upload
```

## Next Steps

1. **Complete Hailo Integration**: Implement actual Hailo runtime integration
2. **Vertex AI Pipeline**: Create Vertex AI Pipeline definition for automated training
3. **Model Evaluation**: Add evaluation metrics and validation sets
4. **Hyperparameter Tuning**: Integrate with Vertex AI Hyperparameter Tuning
5. **Monitoring**: Add training monitoring and logging integration

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Google Cloud TPU](https://cloud.google.com/tpu)
- [Orbax Checkpointing](https://github.com/google/orbax)

