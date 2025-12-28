# GPU Training Setup for ContinuonBrain

This guide sets up JAX-based training on a Windows machine with an NVIDIA GPU.

## Requirements

- Windows 10/11 with NVIDIA GPU (4GB+ VRAM)
- NVIDIA Driver 525+ 
- CUDA 12.x
- Python 3.10 or 3.11
- ~10GB disk space

## Quick Setup

### 1. Install CUDA Toolkit

Download and install CUDA 12.x from NVIDIA:
- https://developer.nvidia.com/cuda-downloads

Verify installation:
```powershell
nvcc --version
nvidia-smi
```

### 2. Clone the Repository

```powershell
git clone <your-repo-url> ContinuonXR
cd ContinuonXR
```

### 3. Run Setup Script

```powershell
# PowerShell (Run as Administrator if needed)
.\scripts\gpu_training\setup_windows_gpu.ps1
```

Or manually:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r scripts/gpu_training/requirements-gpu.txt
```

### 4. Verify GPU Setup

```powershell
python scripts/gpu_training/verify_gpu.py
```

### 5. Copy Training Data from Pi

Copy RLDS episodes from your Pi:
```powershell
# From Pi to Windows (run on Pi or use SCP)
scp -r pi@<pi-ip>:/opt/continuonos/brain/rlds/episodes ./rlds_data/
```

### 6. Run Training

```powershell
python scripts/gpu_training/train_gpu.py --epochs 100 --batch-size 32
```

## Performance Comparison

| Hardware | Time per Epoch | 100 Epochs |
|----------|---------------|------------|
| Pi 5 (CPU) | ~10 min | ~16 hours |
| GTX 1650 4GB | ~30 sec | ~50 min |
| RTX 3060 12GB | ~10 sec | ~17 min |
| RTX 4090 24GB | ~3 sec | ~5 min |

## Export Model to Pi

After training, export the model for deployment on Pi:

```powershell
python scripts/gpu_training/export_model.py --output model_checkpoint.zip
```

Then copy to Pi:
```powershell
scp model_checkpoint.zip pi@<pi-ip>:/opt/continuonos/brain/model/adapters/candidate/
```

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch-size 8`
- Use gradient checkpointing: `--gradient-checkpoint`

### "JAX not using GPU"
- Verify CUDA installation: `nvidia-smi`
- Check JAX GPU: `python -c "import jax; print(jax.devices())"`

### "Module not found"
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements-gpu.txt`

