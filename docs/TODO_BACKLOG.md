# ContinuonXR: Active TODO Backlog

**Last Updated:** 2026-01-25  
**Source:** Chief Scientist goals + Learning Partner blockers

---

## 🔴 Critical (Blocking Training Progress)

### Navigation Goal Blockers

| Task | Location | Status |
|------|----------|--------|
| Room Scanner needs real scan data | `trainer_ui/room_scanner.py` | 🔴 Blocked |
| HomeScan simulator empty scenes | `trainer_ui/house_scenes/` | 🔴 No data |
| OAK-D depth calibration | `trainer_ui/oakd_camera.py` | 🟡 Needs tuning |

**Action:** Run room scanner to capture home environment:
```bash
cd trainer_ui && python room_scanner.py --scan living_room
```

### Face Recognition Blockers

| Task | Location | Status |
|------|----------|--------|
| No family faces enrolled | `trainer_ui/face_db/` | 🔴 Empty |
| Recognition model not trained | `brain_b/memory/faces/` | 🔴 Missing |

**Action:** Enroll faces via trainer UI:
```bash
cd trainer_ui && python server.py
# Open http://localhost:8000 → Faces tab → Enroll
```

---

## 🟡 High Priority (This Week)

### Brain B Improvements

- [ ] **Add spatial memory** - Remember where objects were seen
  - Location: `brain_b/memory/spatial.py` (create)
  - Needed for: fetch tasks, navigation

- [ ] **Voice command parser** - Improve intent classification
  - Location: `brain_b/conversation/intents.py`
  - Current: ~60% accuracy, Target: 90%

- [ ] **Behavior composition** - Chain multiple behaviors
  - Location: `brain_b/actor_runtime/teaching.py`
  - Example: "patrol then return" 

### Hardware Integration

- [ ] **Arm pose library** - Common poses (home, reach, grasp)
  - Location: `trainer_ui/poses/`
  - Status: 3 poses defined, need 10+

- [ ] **Motor calibration** - Mecanum wheel speed tuning
  - Location: `brain_b/hardware/motors.py`
  - Issue: Drift during navigation

---

## 🟢 Medium Priority (This Month)

### Android XR Trainer (Qualcomm Bounty)

- [ ] NexaSDK voice pipeline integration
- [ ] Real-time RLDS recording from glasses
- [ ] Gesture recognition for arm control

Location: `apps/continuonxr/src/main/java/com/continuonxr/app/trainer/`

### Documentation

- [x] ~~Update NEXT_STEPS.md~~ ✅ 2026-01-25
- [x] ~~Update TODO_BACKLOG.md~~ ✅ 2026-01-25
- [ ] Update main README.md Quick Start
- [ ] Remove HOPE references from docs (105 instances)
- [ ] Add Chief Scientist documentation

### Testing

- [ ] Fix 10 failing tests from V1 checklist
  - Service registry: 3 tests
  - Task library fixtures: 3 tests
  - Wave particle tensors: 2 tests
  - Proto schema: 1 test
  - Safety clipping: 1 test

---

## 🔵 Low Priority (Backlog)

### Future Features

- [ ] Multi-robot coordination (L6 SWARM)
- [ ] Cloud TPU training export
- [ ] Public RLDS episode sharing
- [ ] Offline Wikipedia context

### Tech Debt

- [ ] Remove deprecated HOPE eval flows
- [ ] Consolidate `continuonbrain/` and `brain_b/`
- [ ] Clean up training log files (>500MB)

---

## Completed (Archive)

### 2026-01-25
- ✅ Chief Scientist daemon running
- ✅ Learning Partner 5-phase loop
- ✅ Seed Model v4.2.0 (0.84 benchmark)
- ✅ 124+ training cycles
- ✅ 4,218 RLDS episodes

### 2026-01-24
- ✅ Android XR Trainer app scaffold
- ✅ NexaSDK stubs for Qualcomm bounty
- ✅ Mechanical design: V-slot mast updates

### Earlier
- ✅ Brain B teachable behaviors
- ✅ Trainer UI with camera/voice
- ✅ RLDS pipeline and export
- ✅ OAK-D camera integration

---

## Quick Commands

```bash
# Check what Chief Scientist is working on
python scripts/compound/chief_scientist.py --goals

# Run a training cycle manually
python scripts/compound/learning_partner.py --train

# Check brain health
python scripts/compound/learning_partner.py --status

# Start trainer UI
cd trainer_ui && python server.py

# Run test suite
cd continuonbrain && python -m pytest tests/ -v
```

---

*Updated by documentation refresh - 2026-01-25*
