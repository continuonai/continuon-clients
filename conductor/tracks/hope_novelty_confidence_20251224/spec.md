# Specification: HOPE Novelty-Based Confidence

## Overview
This track replaces hardcoded heuristics for agent confidence with a dynamic "Surprise" signal derived from the HOPE brain's prediction error (novelty). By using actual internal model state to determine confidence, the robot can more accurately identify novel situations and transparently escalate to high-level reasoning when its edge model is uncertain.

## Functional Requirements
- **HOPE Novelty Computation:**
    - Implement a `compute_novelty(prediction, actual)` method in the HOPE brain (JAX/PyTorch).
    - Use Mean Squared Error (MSE) or Cross-Entropy between the predicted and actual state vectors as the raw novelty metric.
- **Exponential Confidence Scaling:**
    - Implement the normalization formula: `confidence = exp(-k * novelty)`, where `k` is a configurable sensitivity constant.
    - Expose this as `hope_confidence` in the `HOPEAgent`.
- **Transparent Hierarchical Fallback:**
    - Update `BrainService.ChatWithGemma` to trigger an LLM fallback when `hope_confidence < threshold`.
    - If fallback occurs, prefix the response with a scientific uncertainty marker (e.g., "[Surprise: 0.85] I'm encountering a novel situation...").
- **Logging & Auditing:**
    - Log `novelty` and `confidence` into the `step_metadata` of every RLDS episode.
    - Stream these metrics to the Brain Studio SSE feed for real-time visualization.
- **Brain Studio Stability View:**
    - Add a "System Surprise" trend line to the stability dashboard to visualize environmental familiarity over time.

## Non-Functional Requirements
- **Latency:** Novelty computation must be performed in-loop with minimal overhead (<5ms).
- **Consistency:** Ensure the novelty signal is consistent across different operating modes (Manual vs. Autonomous).

## Acceptance Criteria
- [ ] Robot correctly identifies a never-before-seen command as "Novel" (low confidence).
- [ ] LLM responses triggered by low HOPE confidence are correctly prefixed with surprise data.
- [ ] RLDS logs contain valid `novelty` and `confidence` fields in `step_metadata`.
- [ ] Brain Studio displays real-time novelty scores in the "Thought Stream" and stability graphs.

## Out of Scope
- Automatic retraining triggered by novelty (Active Learning loops beyond logging).
- Multi-modal novelty (e.g., visual surprise vs. textual surprise).
