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
    *   [ ] **Frontend Integration:** Connect `static/brain-viz.js` (Three.js) to the `HOPEBrain` state in `BrainService` and expose a JSON feed that the viz can poll/stream (episode nodes, consolidation cues, active path).
    *   [ ] **Visual Metaphors:**
        *   **New Memory:** A new particle/node appears when an RLDS episode is saved; tag nodes with episode metadata (task, tool used, timestamp).
        *   **Consolidation:** Nodes merge or glow during the "Sleep/Training" cycle; surface training metrics (loss deltas) as color/opacity.
        *   **Query:** The active "path" lights up during inference; show token/tool logits as edge weights to debug routing decisions.
    *   [ ] **Dashboard Embedding:** Embed this visualization in the "Research" tab of the Agency UI with start/stop controls and a "snapshot" button that exports the current graph view to RLDS for later replay.

## 3. Teach a New Skill (Calculator → Common Tools → Wikipedia) 🛠️
**Objective:** Prove the end-to-end pipeline by teaching HOPE a functional tool it doesn't currently know, then escalate difficulty toward research-grade retrieval.

*   **Phase A: Calculator competency**
    *   [ ] **Define Tool:** Implement `CALCULATOR` in `continuonbrain/ops/tool_definitions.py` with clear input/units handling.
    *   [ ] **Curiosity Session:** Run a *targeted* `RunChatLearn` session where the "Teacher" (Gemini) explicitly demonstrates *when* to use the calculator versus free-form reasoning.
        *   *Prompt Key:* "Demonstrate using the Calculator for complex multiplication."
    *   [ ] **Train & Verify:** Trigger JAX training on this dataset; ask HOPE "What is 123 * 456?" and verify it calls the tool instead of hallucinating.

*   **Phase B: Broader tool practice**
    *   [ ] **Extend Tooling:** Add at least one more common utility (e.g., unit conversion or date math) using the same tool definition path.
    *   [ ] **Curriculum:** Script `RunChatLearn` examples that contrast when to use each tool; log mistakes to RLDS and rerun a short fine-tune.
    *   [ ] **Evaluation:** Pose mixed queries ("Convert 72F to C and multiply by 3") and confirm correct multi-tool routing.

*   **Phase C: Wikipedia-style retrieval**
    *   [ ] **Offline Corpus Hook:** If a Wikipedia JSONL dump is available, wire `continuonbrain/eval/wiki_retriever.py` into HOPE tool definitions as `SEARCH_WIKI`, defaulting to no-op when the corpus is absent.
    *   [ ] **Teacher Sessions:** Capture examples where Gemini demonstrates citing retrieved snippets for factual answers.
    *   [ ] **Verification:** Ask factoid questions ("Who discovered penicillin?") and confirm the model prefers `SEARCH_WIKI` with citations before free-form completion.

---
*Created: 2025-12-19*
