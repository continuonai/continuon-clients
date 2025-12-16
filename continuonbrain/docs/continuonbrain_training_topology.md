# ContinuonBrain Training & Deployment Topology (v0.2)

This note captures the current JAX-first, Google-aligned plan for ContinuonBrain training and deployment. It documents how seed models are born on Colab/TPU, embodied on Pi, scaled on Vertex TPU, and continuously personalized while respecting privacy and RLDS conventions.

## Goals & Stack
- **Backend:** JAX/Flax + XLA; TPU is the primary forge.
- **Data format:** RLDS/TFRecord (TFDS-compatible) with episode manifests.
- **Privacy:** Cloud ingestion only accepts PII-safe, opt-in episodes; safety-violation episodes are upload-eligible but still redacted.
- **Edge focus:** Pi 5 runs HOPE loops and idle training with local-only memories after OTA.

## Phase Flow (Local → Edge → Cloud → Edge)
1. **Seed genesis (Google Colab TPU notebook)**
   - Train a small-but-real SSM/WaveCore bundle in JAX.
   - Export Orbax checkpoints and an edge package to GCS.
2. **Seed-down embodiment (Pi 5)**
   - Pull the seed package, run realtime inference, and record RLDS locally.
   - Run all three HOPE loops with constrained trainable surfaces (adapters/head/calibration/local skill memory).
   - Apply PII-safe transforms before any episode upload.
3. **Cloud forge scale (Vertex AI TPU)**
   - Ingest PII-safe episodes (opt-in + safety escalation) to `gs://continuon-episodes-raw/` with sidecar metadata.
   - Build curated datasets (manifest + sharded TFRecords) and train the full backbone on TPU.
   - Evaluate, bless, and publish OTA channel pointers (`stable/beta/nightly/rollback`).
4. **OTA + edge persistence (Pi 5 idle training)**
   - Robots poll channel pointers, run canary eval, and activate.
   - Idle/sleep training runs only the slow loop on local memories, bounded by thermal/battery/time budgets with rollback to last-good deltas.

## HOPE Loop Responsibilities on Pi
- **Fast loop:** Online/near-online adaptation of tiny adapters + calibration; bounded steps.
- **Medium loop:** Idle consolidation using local RLDS slices.
- **Slow loop (local):** Deepest on-device consolidation; still limited to adapters/head/local skill memory.
- **Post-OTA rule:** Only the slow loop runs, and only against local memories.

## RLDS & Privacy Rules
- **Upload eligibility:** Operator opt-in per episode; safety violations may auto-flag for upload.
- **PII-safe transform (edge):** Mask faces/plates, prefer audio features over raw speech, hash robot identifiers, and coarsen/remove precise location.
- **Cloud layout:**
  - Raw: `gs://continuon-episodes-raw/rlds_v0/robot_hash=<...>/date=YYYY-MM-DD/episode_id=<...>/`
  - Curated: `gs://continuon-datasets-curated/<dataset>/vX.Y.Z/{train,val}/shard-*.tfrecord` + `manifest.json`
  - Artifacts: `gs://continuon-model-artifacts/<family>/<run_id>/checkpoints` + `export/<target>/`
  - Channels: `gs://continuon-model-artifacts/channels/{stable,beta,nightly,rollback}.json`

## Hardware Guidance
- **Seed/local iteration:** RTX 3050 laptop is acceptable for the first end-to-end seed (use tiny configs, grad accumulation, mixed precision); same codepath scales to TPU.
- **Edge baseline:** Raspberry Pi 5 **16GB** is the preferred drop-in upgrade to relieve memory pressure while keeping the AI HAT+ 26 TOPS accelerator.
- **Home edge brain upgrade:** Jetson Orin Nano Super Dev Kit as a companion to Pi (Pi handles sensing/recording; Jetson handles heavier local consolidation) if you want more on-prem capability.
- **Cloud forge:** Vertex AI TPU for global training; Colab TPU for quick JAX bring-up.

## Immediate Build Targets (v0)
1. Colab TPU seed notebook that trains an SSM/WaveCore seed, checkpoints to GCS, and exports an edge package.
2. Pi 5 runtime that loads the package, records RLDS, applies PII-safe transforms, and runs HOPE loops with bounded trainable surfaces.
3. Ingestion + curation: sidecar-driven BigQuery index, manifest-based dataset builder, sharded TFRecords.
4. Vertex TPU training job: reads manifests, trains, evaluates, blesses, and updates channel pointers.

## Operational Safeguards
- Enforce idle training budgets (time/thermal/battery) and maintain `last_good_local_delta` for rollback.
- Require hash verification + canary eval before activating OTA payloads.
- Keep model metadata with every artifact (dataset manifest versions, code commit hash, hardware, safety config version, export target).
