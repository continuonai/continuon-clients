# Specification: Memory Consolidation & Decay

## Overview
This track implements a long-term memory management system for the ContinuonBrain. As the robot accumulates thousands of experiences, this system ensures memory quality by merging redundant entries into synthesized "semantic anchors" and decaying the confidence of old, unused, or unvalidated data. This prevents memory bloat and prioritizes the most relevant and accurate information for retrieval.

## Functional Requirements
- **Threshold-based Consolidation:**
    - Automatically trigger a consolidation job when unvalidated memories exceed a configurable limit (e.g., 500).
    - Group memories using the existing semantic embedding (similarity > 0.90).
- **LLM-based Answer Synthesis:**
    - For every merged cluster, use the LLM (Slow Loop) to synthesize a single, high-quality answer that represents the group.
    - Store the result as a new "Semantic Anchor" with an initial high confidence score.
- **Dynamic Confidence Decay:**
    - **Model Evolution Trigger:** Automatically apply a 10% confidence penalty to all unvalidated memories when a new model checkpoint is promoted.
    - **Status-based Gating:** "Gold Data" (validated: true) is immune to decay.
    - **Recency Decay:** Apply a daily confidence decay (e.g., 0.95 factor) to unvalidated memories that haven't been accessed in 30 days.
- **Autonomy Orchestrator Integration:**
    - Register these background jobs with the `Autonomy Orchestrator` to ensure they only run during low-priority modes (Idle or Sleep Learning).

## Non-Functional Requirements
- **Resource Respect:** Consolidation must pause or slow down if `ResourceMonitor` flags a memory warning.
- **Atomicity:** Consolidation should be performant and not lock the primary `learned_conversations.jsonl` for more than 500ms at a time.

## Acceptance Criteria
- [ ] Memory consolidation reduces the total entry count by at least 20% in a simulated high-redundancy dataset.
- [ ] Synthesis correctly produces a single answer that incorporates the key points of the merged cluster.
- [ ] Unvalidated memories correctly show a confidence drop after a model promotion event.
- [ ] Validated memories maintain 1.0 confidence regardless of age or model changes.

## Out of Scope
- Automatic deletion of memories (decayed memories are kept but hidden from high-confidence retrieval).
- Real-time consolidation during active chat (it is always an offline/background task).
