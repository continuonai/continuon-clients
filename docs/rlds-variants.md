# RLDS Variants in this Repo (and how to normalize them)

This repo currently produces multiple “RLDS-like” episode shapes. That’s OK (different producers, different fidelity), but **we must be explicit** about:
- which producers emit which variant,
- which validators apply,
- and how to convert everything into a single canonical on-disk layout for replay/training.

## Canonical on-disk layout (episode_dir)
Canonical storage layout used across Pi capture + downstream tooling:

```
<episode_dir>/
  metadata.json
  steps/
    000000.jsonl
  blobs/               # optional (video/depth/audio/etc)
```

Each `steps/000000.jsonl` line is a dict containing:
- `observation`: dict
- `action`: dict
- `reward`: number (optional; default 0.0)
- `is_terminal`: bool (optional; default false)
- `step_metadata`: map<string,string> (optional; recommended for non-schema flags)

## Variants produced today

### Variant A: `episode_dir` (robotics-style capture)
- **Producer(s)**:
  - `continuonbrain/recording/arm_episode_recorder.py` (Pi arm capture)
  - `continuonbrain/rlds/community_dataset_importer.py` (community dataset ingest)
- **Layout**: already canonical `episode_dir/metadata.json + steps/000000.jsonl`.
- **Validator**: `continuonbrain/rlds/validators.py` (strict “studio/mock robotics envelope”).
- **Notes**: best fit for training pipelines; includes placeholders and alignment fields.

### Variant B: `single_json` (HOPE eval / ad-hoc episodes)
- **Producer(s)**:
  - `continuonbrain/eval/hope_eval_runner.py` (writes `<prefix>_<ts>.json`)
  - `continuonbrain/services/brain_service.py::save_episode_rlds` (writes `recordings/episodes/episode_<ts>.json`)
- **Layout**: single JSON file, typically `{"steps":[...]}` (or custom episode envelope).
- **Validator**: none reliable today (does not match `validators.py` strict schema).
- **Action**: normalize to canonical episode_dir before training or archival.

### Variant C: `single_jsonl` (JAX EpisodeLogger)
- **Producer(s)**:
  - `continuonbrain/jax_models/data/episode_logger.py` (writes `<episode_id>.jsonl`)
- **Layout**: JSONL file where each line is a step dict; metadata is in-memory only.
- **Validator**: none reliable today (no metadata.json).
- **Action**: normalize to canonical episode_dir (derive metadata + wrap steps).

### Variant D: `metrics_episode_dir` (background curiosity traces)
- **Producer(s)**:
  - `continuonbrain/services/background_learner.py` (optional; if configured)
- **Layout**: canonical episode_dir; steps carry novelty/lyapunov/etc in `step_metadata`.
- **Validator**: should use a permissive “metrics variant” validator (not strict robotics).

## Recommended development rule
- **All training + cloud export inputs** should be **canonical episode_dir**.
- Producers are allowed to emit non-canonical variants, but must provide a normalizer path.


