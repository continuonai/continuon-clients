#!/bin/bash
# Minimal TPU VM startup script to run the JAX trainer.
# Expects environment variables:
#   BUCKET (e.g., gs://continuon-rlds)
#   OUTPUT_DIR (e.g., gs://continuon-rlds/checkpoints/core_model_v0)
#   DATA_PATH (e.g., gs://continuon-rlds/rlds/episodes)

set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-pip git
pip install --upgrade pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax tensorflow orbax-checkpoint google-cloud-storage

if [ ! -d /opt/continuonxr ]; then
  git clone https://github.com/continuon/ContinuonXR.git /opt/continuonxr
fi

cd /opt/continuonxr
export PYTHONPATH=$(pwd)

python -m continuonbrain.jax_models.train.cloud.tpu_train \
  --data-path "${DATA_PATH:-${BUCKET}/rlds/episodes}" \
  --output-dir "${OUTPUT_DIR:-${BUCKET}/checkpoints/core_model_v0}" \
  --batch-size 256 \
  --num-steps 10000 \
  --learning-rate 1e-4

