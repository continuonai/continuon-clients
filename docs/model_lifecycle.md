# Model Lifecycle: HOPE/CMS Governance

This note connects the over-the-air (OTA) delivery phases with the HOPE/CMS nested learning loops and clarifies how on-device memory persists between updates.

For the complete reconciled system architecture and training lifecycle, see [System Architecture and Training Lifecycle](./system-architecture.md).

## OTA phases mapped to HOPE/CMS loops
- **Fast & Mid loops stay resident on-device:** Reflexive controls and short-horizon skills ship inside the edge bundle and remain loaded across OTA installs so latency-sensitive behaviors do not depend on cloud reachability.
- **Slow loop checkpoints originate in cloud:** Corpus-scale Slow-loop training runs in Continuon Cloud, producing the baseline checkpoints that distill into the on-device Fast/Mid bundles.
- **Device traces replayed in cloud Slow loop:** Telemetry and RLDS logs captured by the device feed the next Slow-loop training round in cloud, ensuring on-device Fast/Mid adaptations are preserved when the global checkpoint is refreshed.

## Memory Plane persistence and merge behavior
The Continuum Memory System (CMS) keeps a **Memory Plane** on-device that is durable across OTA updates. During boot, the Memory Plane is merged with the latest global skills exported by the Slow loop so the device retains personalized habits while adopting new worldwide behaviors. HOPE/CMS responsibilities stay aligned: on-device Fast/Mid loops read from and write to the Memory Plane, while the cloud Slow loop reconciles device traces into the global model before packaging the next bundle.
