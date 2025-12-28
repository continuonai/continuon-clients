# GPU Training Setup Script for Windows
# Run in PowerShell (Administrator recommended)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ContinuonBrain GPU Training Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.10 or 3.11" -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Check NVIDIA GPU
Write-Host ""
Write-Host "Checking NVIDIA GPU..." -ForegroundColor Yellow
$nvidiaSmi = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: NVIDIA GPU not detected. Install NVIDIA drivers." -ForegroundColor Red
    exit 1
}
Write-Host "Found GPU: $nvidiaSmi" -ForegroundColor Green

# Check CUDA
Write-Host ""
Write-Host "Checking CUDA..." -ForegroundColor Yellow
$nvccVersion = nvcc --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: CUDA toolkit not found in PATH" -ForegroundColor Yellow
    Write-Host "JAX will attempt to use bundled CUDA. For best performance, install CUDA 12.x" -ForegroundColor Yellow
} else {
    Write-Host "Found CUDA" -ForegroundColor Green
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "Created .venv" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install GPU requirements
Write-Host ""
Write-Host "Installing GPU training requirements..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
pip install -r scripts/gpu_training/requirements-gpu.txt

# Verify installation
Write-Host ""
Write-Host "Verifying JAX GPU installation..." -ForegroundColor Yellow
python -c "import jax; devices = jax.devices(); print(f'JAX devices: {devices}')"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Setup Complete!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Verify GPU: python scripts/gpu_training/verify_gpu.py"
    Write-Host "  2. Copy training data from Pi"
    Write-Host "  3. Run training: python scripts/gpu_training/train_gpu.py"
} else {
    Write-Host ""
    Write-Host "WARNING: JAX may not be using GPU. Check CUDA installation." -ForegroundColor Yellow
}

