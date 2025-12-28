@echo off
REM GPU Training Setup Script for Windows
REM Run this in Command Prompt or PowerShell

echo ============================================
echo   ContinuonBrain GPU Training Setup
echo ============================================
echo.

REM Check Python
echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10 or 3.11
    exit /b 1
)

REM Check NVIDIA GPU
echo.
echo Checking NVIDIA GPU...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if errorlevel 1 (
    echo ERROR: NVIDIA GPU not detected. Install NVIDIA drivers.
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists
) else (
    python -m venv .venv
    echo Created .venv
)

REM Activate and install
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing GPU training requirements...
echo This may take 5-10 minutes...
pip install -r scripts\gpu_training\requirements-gpu.txt

REM Verify
echo.
echo Verifying JAX GPU installation...
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Verify GPU: python scripts\gpu_training\verify_gpu.py
echo   2. Copy training data from Pi
echo   3. Run training: python scripts\gpu_training\train_gpu.py
echo.

pause

