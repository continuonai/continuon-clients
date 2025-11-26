# WaveCore Toy FFT Model (Pi-friendly)

This folder holds a lightweight PyTorch prototype used to validate the FFT-based SpectralBlock and HybridBlock design on Raspberry Pi 5–class hardware.

## Files
- `toy_wave_model.py`: SpectralBlock, HybridBlock, SpectralLanguageModel, and a small synthetic training loop sized for Pi CPUs.

## Quickstart (Pi 5)
1. Install PyTorch CPU build (ARM wheel) and dependencies:
   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   ```
2. Run a short sanity-check training loop:
   ```bash
   python -m continuonbrain.wavecore.toy_wave_model --steps 200 --device cpu
   ```
   You should see the loss decrease every ~50 steps, confirming FFT + spectral decay stability.

## Ablations
- **Wave-only**: Use `SpectralBlock` stack in `SpectralLanguageModel`.
- **Transformer-only**: Swap `SpectralBlock` with attention-only blocks if you want a baseline.
- **Hybrid**: Compose `HybridBlock` in a custom model to fuse spectral and attention paths with the provided layer norms.

## Notes
- Defaults target Pi 5 memory/latency budgets (e.g., `d_model=128`, `seq_len=64`, `batch_size=16`).
- The synthetic task predicts `(t + 1) mod vocab_size` for quick regression testing without external data.
