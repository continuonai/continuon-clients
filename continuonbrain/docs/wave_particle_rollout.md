# HOPE Wave–Particle Rollout Harness

The wave–particle duality described in the HOPE architecture is now wired into a
lightweight rollout loop so you can verify that the spectral **wave** state and
MLP **particle** state co-evolve, gated by the fast state and CMS context.

## Why this exists
- Confirms that the existing HOPECore + CMS modules run end-to-end on constrained
  hardware (Pi-friendly defaults) without the full training stack.
- Surfaces the gate, wave, and particle contributions at every step so you can
  check that the wave path stays stable while the particle path handles sharp,
  local updates.

## Run a quick rollout
```bash
python -m continuonbrain.hope_impl.wave_particle_rollout --steps 8 --device cpu
```

Each step prints a single line like:
```
step=3 gate=0.497 wave_norm=4.120 particle_norm=2.085 fusion_norm=6.013 cms_energy=0.000 lyapunov=3.000
```

### What to look for
- **gate** should stay in `[0,1]`, showing how much the fast state leans on the
  particle vs wave path.
- **wave_norm** should remain bounded even as the window slides (spectral state
  is constant-size SSM-style).
- **particle_norm** can spike briefly on sharp transitions; gating controls how
  much of that reaches the fast state.
- **lyapunov** and **cms_energy** give a quick sanity check on stability and
  memory decay.

## Tuning knobs
- Edit `WaveParticleRolloutConfig` in `continuonbrain/hope_impl/wave_particle_rollout.py`
  to adjust observation/action dims, reward scaling, or swap in `HOPEConfig.pi5_optimized`
  for on-device runs.
- Increase `--steps` for longer traces; the CMS decay defaults ensure the memory
  footprint stays constant while the wave state remains O(1) with respect to
  sequence length.

