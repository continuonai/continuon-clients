# Continuon AI: Repository TODO Backlog

This document provides a structured overview of all `TODO` and `FIXME` comments found in the ContinuonXR repository, categorized by module and prioritized by impact.

## Executive Summary
A repository-wide scan identified **21** actionable `TODO` items. 
- **High Priority:** 6
- **Medium Priority:** 13
- **Low Priority:** 2

The most critical gaps are in the **Brain Runtime integration** (Pi 5 sensor wiring) and **auth/ownership tokens** in the Companion App.

---

## 🧠 Brain Runtime (`continuonbrain`)

### High Priority 🔴
- **`trainer/examples/pi5_integration.py` (L187-190):** Wire `robot_idle`, `teleop_active`, `battery_level`, and `cpu_temp` to the actual Pi 5 sensors and runtime state.
- **`api/server.py` (L1450):** Implement actual task execution logic in the Robot API.

### Medium Priority 🟡
- **`trainer/CLOUD_EXPORT.md` (L42):** Replace placeholder model/tokenizer/LoRA config with real production values.
- **`services/brain_service.py` (L1451):** Add image support to multimodal payloads.
- **`reasoning/jax_adapter.py` (L23):** Support stateful rollout in the JAX reasoning engine.
- **`gemma_chat.py` (L538):** Handle images for API if supported (e.g., GPT-4o).
- **`behaviors/auto_charge.py` (L204):** Implement waypoint navigation for docking.
- **`behaviors/auto_charge.py` (L258):** Integrate vision system distance feedback for docking.

### Low Priority 🟢
- **`gemma_chat.py` (L629):** Switch to `apply_chat_template` if supported by the processor.

---

## 📱 Companion App (`continuonai`)

### High Priority 🔴
- **`lib/services/brain_client.dart` (L72):** Wire to persistent auth, subscription, and ownership tokens once the backend is ready.

### Medium Priority 🟡
- **`lib/services/brain_client.dart` (L716):** Implement a real reset endpoint (currently using a 'reflex' mode workaround).
- **`lib/screens/robot_list_screen.dart` (L1310):** Implement actual robot ownership transfer via the API.
- **`ios/Runner/Info.plist` (L68):** Replace placeholder `REVERSED_CLIENT_ID` with real Firebase config.
- **`firestore.rules` (L12):** Remove hardcoded developer email once RBAC is fully mature.
- **`android/app/build.gradle.kts` (L26):** Specify a unique Application ID for production.
- **`android/app/build.gradle.kts` (L38):** Configure signing for release builds.

---

## 👓 Android XR Shell (`apps/continuonxr`)

### Medium Priority 🟡
- **`src/main/java/com/continuonxr/app/config/AppConfig.kt` (L57):** Implement loading app configuration from disk or flags instead of defaults.

---

## 📄 Documentation & Misc

### Low Priority 🟢
- **`AGENTS.md` (L26):** Implement offline Wikipedia context for agents.

---

## Recommended Next Tracks
Based on this review, the following high-priority tracks are recommended:
1. **Track: Pi 5 Sensor Integration & Telemetry:** Address the missing sensor wiring in the Brain Runtime.
2. **Track: Robot API Task Execution:** Implement the core task execution logic in the Brain's gRPC/Web server.
3. **Track: Secure Ownership & Token Persistence:** Implement the persistent auth and ownership flow in the Companion App.
