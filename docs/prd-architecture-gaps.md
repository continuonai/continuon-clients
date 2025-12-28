# PRD: Architecture Gap Closures

**Version:** 1.0  
**Date:** 2025-12-28  
**Owner:** Continuon Brain Team  
**Status:** Active Development

---

## Executive Summary

This PRD documents the architectural gaps between the current Continuon Brain implementation and the full HOPE/CMS vision described in the README and PRD. It defines requirements, acceptance criteria, and implementation priorities for closing these gaps.

---

## Gap 1: World Model Training

### Current State
- JAX CoreModel and Mamba World Model exist but are **untrained**
- WaveCore loops are implemented but haven't consumed real RLDS data
- World model predictions use stub/deterministic fallback

### Target State
- Trained world model that predicts physics from experience
- Robot can "imagine" action outcomes before execution
- Enables Chollet's "Symbolic Search" for invention

### Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| WM-1 | Train JAX CoreModel on ≥100 real RLDS episodes | P0 |
| WM-2 | Achieve prediction error <0.1 on held-out episodes | P0 |
| WM-3 | World model inference <50ms on Pi5 | P1 |
| WM-4 | Export trained checkpoint for OTA distribution | P1 |
| WM-5 | Integrate with HOPE Agent for planning queries | P0 |

### Acceptance Criteria
- [ ] WaveCore loops complete successfully with real data
- [ ] `predict_action_outcome()` returns trained predictions
- [ ] Beam search planning uses trained model
- [ ] Checkpoint saved to `/opt/continuonos/brain/model/world_model.pt`

### Implementation Path
```
1. Collect RLDS episodes from autonomous runs
2. Run WaveCore fast/mid/slow loops
3. Export checkpoint
4. Wire to HOPE Agent's world_model parameter
5. Validate prediction accuracy
```

---

## Gap 2: Unified VisionCore Backbone

### Current State
- SAM3 segmentation service exists (`sam3_vision.py`)
- Hailo object detection exists (`hailo_inference.py`)
- OAK-D depth capture exists (`oak_depth_capture.py`)
- **No unified perception backbone** combining these

### Target State
- Single VisionCore class that fuses:
  - RGB from OAK-D
  - Depth from OAK-D
  - Object detection from Hailo (26 TOPS)
  - Semantic segmentation from SAM3
  - VPU features from OAK-D Myriad X
- Outputs structured scene representation for HOPE

### Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| VC-1 | Unified `VisionCore` class wrapping all vision services | P0 |
| VC-2 | Scene graph output with detected objects + depth + masks | P0 |
| VC-3 | Real-time performance ≥10 FPS on Pi5 | P1 |
| VC-4 | Graceful degradation when components missing | P1 |
| VC-5 | Integration with HOPE Agent observation encoder | P0 |

### Acceptance Criteria
- [ ] `VisionCore.perceive(frame)` returns structured scene
- [ ] Scene includes: objects, depths, masks, confidence scores
- [ ] Works with partial hardware (e.g., no Hailo)
- [ ] HOPE Agent can query current scene understanding

### Architecture
```
┌──────────────────────────────────────────────────────┐
│                    VisionCore                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │   OAK-D    │  │   Hailo    │  │    SAM3    │     │
│  │ RGB+Depth  │  │ Detection  │  │ Segmentation│    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘     │
│        │               │               │            │
│        └───────────────┼───────────────┘            │
│                        │                            │
│              ┌─────────▼─────────┐                  │
│              │   Scene Graph     │                  │
│              │   Fusion Layer    │                  │
│              └─────────┬─────────┘                  │
│                        │                            │
│              ┌─────────▼─────────┐                  │
│              │ SceneRepresentation│                 │
│              │ - objects[]        │                 │
│              │ - depth_map        │                 │
│              │ - masks[]          │                 │
│              │ - confidence       │                 │
│              └───────────────────┘                  │
└──────────────────────────────────────────────────────┘
```

---

## Gap 3: SkillPolicies Integration

### Current State
- AINA policy head exists (`aina_impl/policy.py`)
- Not wired to HOPE's action output
- No skill selection mechanism

### Target State
- SkillPolicies module that:
  - Selects appropriate skill based on goal
  - Executes manipulation/locomotion primitives
  - Reports execution status back to HOPE

### Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| SP-1 | Skill registry with loadable policy heads | P1 |
| SP-2 | AINA policy integration for manipulation | P1 |
| SP-3 | Skill-specific LoRA adapters | P2 |
| SP-4 | Execution feedback loop to HOPE | P1 |

---

## Gap 4: LanguagePlanner Head

### Current State
- Using Gemma chat as general fallback
- No dedicated planning-focused LLM head

### Target State
- LanguagePlanner head specialized for:
  - Goal decomposition
  - Step-by-step planning
  - Constraint reasoning

### Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| LP-1 | Planning-tuned prompt template | P2 |
| LP-2 | Structured plan output (steps, dependencies) | P2 |
| LP-3 | Integration with world model for feasibility | P2 |

---

## Gap 5: VQ-VAE Latent Tokenization

### Current State
- VQ-GAN scaffold exists in `01_vision_dreamer/`
- Not connected to perception pipeline

### Target State
- Latent tokenization of visual inputs
- Compressed representation for world model

### Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| VQ-1 | Train VQ-VAE on robot camera data | P2 |
| VQ-2 | Export to Hailo HAT for acceleration | P3 |
| VQ-3 | Wire tokens to world model input | P2 |

---

## Implementation Priority Matrix

| Gap | Priority | Effort | Dependencies | Phase |
|-----|----------|--------|--------------|-------|
| World Model Training | P0 | Medium | RLDS data | 1 |
| VisionCore Backbone | P0 | Medium | Hardware present | 1 |
| SkillPolicies | P1 | High | VisionCore, World Model | 2 |
| LanguagePlanner | P2 | Medium | None | 2 |
| VQ-VAE Tokenization | P2 | High | VisionCore | 3 |

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| World Model Prediction Error | <0.1 | N/A (untrained) |
| VisionCore FPS | ≥10 | N/A |
| End-to-end Planning Latency | <200ms | N/A |
| Skill Execution Success Rate | >80% | N/A |

---

## Appendix: Current Alignment Score

```
HOPE/CMS Architecture:        ██████████ 100%
World Model Integration:      ████████░░  80% → Target: 100%
Semantic Search:              ██████████ 100%
SafetyHead:                   ██████████ 100%
Vision Integration:           ████████░░  80% → Target: 100%
Fast/Mid/Slow Loops:          ██████████ 100%

Current Overall:              ████████░░  87%
Target Overall:               ██████████ 95%
```

