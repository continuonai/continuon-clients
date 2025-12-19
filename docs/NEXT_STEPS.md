# ContinuonXR: Next Steps Roadmap

With the "Hope Independence" architecture validated (Curiosity Loop -> RLDS Data -> Mock/Local Inference), the system is ready for real-world growth.

## 1. Deploy & Test on Hardware 🤖
**Objective:** Validate the *real* `gemma-270m` (or larger 2B quantized) model on physical robot hardware (Raspberry Pi 5 + Hailo / Jetson Orin) where NPU/RAM availability avoids the OOM crashes seen in the dev environment.

*   **Actionable Items:**
    *   [ ] **Create Deployment Script:** Bundle `continuonbrain` into a deployable artifact (ZIP/Docker) excluding dev tools.
    *   [ ] **Hardware Config:** Ensure `settings.json` on the robot defines `agent_model: hope-v1` and points to the NPU delegate.
    *   [ ] **Test Run:** Execute `test_independence.py` on the robot to measure *actual* inference latency (target < 2s).

## 2. Visualize the "Brain" 🧠
**Objective:** Create a visual feedback loop for the "Compact Memory System" (CMS) to make learning observable and exciting.

*   **Actionable Items:**
    *   [ ] **Frontend Integration:** Connect the existing `static/brain-viz.js` (Three.js) to the `HOPEBrain` state in `BrainService`.
    *   [ ] **Visual Metaphors:**
        *   **New Memory:** A new particle/node appears when an RLDS episode is saved.
        *   **Consolidation:** Nodes merge or glow during the "Sleep/Training" cycle.
        *   **Query:** The active "path" lights up during inference.
    *   [ ] **Dashboard:** Embed this visualization in the "Research" tab of the Agency UI.

## 3. Teach a New Skill 🛠️
**Objective:** Prove the end-to-end pipeline by teaching HOPE a functional tool it doesn't currently know.

*   **Actionable Items:**
    *   [ ] **Define Tool:** Implement `SEARCH_WIKI` or `CALCULATOR` in `continuonbrain/ops/tool_definitions.py`.
    *   [ ] **Curiosity Session:** Run a *targeted* `RunChatLearn` session where the "Teacher" (Gemini) explicitly demonstrates *when* to use this new tool.
        *   *Prompt Key:* "Demonstrate using the Calculator for complex multiplication."
    *   [ ] **Train:** Trigger JAX training on this specific dataset.
    *   [ ] **Verify:** Ask HOPE "What is 123 * 456?" and verify it calls the tool instead of hallucinating.

---
*Created: 2025-12-19*
