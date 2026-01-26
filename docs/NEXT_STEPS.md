# ContinuonXR: Next Steps Roadmap

**Last Updated:** 2026-01-25  
**Current State:** Learning Partner + Chief Scientist actively training

---

## Current System Status

### Active Training Systems

| System | Status | Cycle | Focus |
|--------|--------|-------|-------|
| **Chief Scientist** | 🟢 Running | Every 15 min | Family-oriented goals |
| **Learning Partner** | 🟢 Running | Every 5 min | TRAIN phase - simulator episodes |

### Seed Model v4.2.0

- **Parameters:** 12.8M
- **Benchmark Score:** 0.84 (17/23 tests)
- **Level Achieved:** ADVANCED (L3 of 6)
- **RLDS Episodes:** 4,218+
- **Training Cycles:** 124+

---

## Immediate Priorities (This Week)

### 1. 🎯 Navigation Goal (0% → 25%)

The Chief Scientist's top priority. Current blockers:

- [ ] **Room Scanner calibration** - HomeScan simulator needs real room data
- [ ] **Obstacle detection tuning** - OAK-D depth thresholds for furniture
- [ ] **Path planning integration** - Connect A* planner to Brain B

**Test:** Robot navigates living room without collision

### 2. 👤 Face Recognition (0% → 25%)

- [ ] **Capture family faces** - Use trainer_ui to enroll 3-5 family members
- [ ] **Train recognition model** - Fine-tune on captured faces
- [ ] **Greeting behavior** - Brain B says hello when recognizing someone

**Test:** Robot greets family member by name

### 3. 🔧 Hardware Integration

- [ ] **OAK-D streaming** - Verify RGB+Depth pipeline stable
- [ ] **Arm calibration** - 6-axis arm pose accuracy
- [ ] **Audio pipeline** - Voice command recognition

---

## Next Milestones (This Month)

### Phase 1: Basic Home Navigation ✅→🔄

```
Room Scanner → HomeScan Sim → Train Navigation → Deploy
     ↓              ↓              ↓              ↓
  3D Model    Synthetic Data   Brain B learns   Real robot
```

**Target:** Robot can navigate between rooms on command

### Phase 2: Family Recognition 🔄

- Enroll all family members (face + voice)
- Personalized greetings and interactions
- Remember preferences per person

**Target:** "Hey [Name], want me to get you a drink?"

### Phase 3: Fetch Tasks 📋

- Object detection (cups, remotes, phones)
- Grasp planning with arm
- Delivery to person

**Target:** "Bring me the TV remote"

---

## Architecture Overview (Current)

```
┌─────────────────────────────────────────────────────────┐
│                   Chief Scientist                        │
│            (Claude Code - Strategic Direction)           │
│         Goals: navigate, recognize, fetch, remind        │
└─────────────────────────┬───────────────────────────────┘
                          │ directs
┌─────────────────────────▼───────────────────────────────┐
│                   Learning Partner                       │
│                 (Autonomous Training)                    │
│      OBSERVE → SIMULATE → TRAIN → TRANSFER → DEPLOY     │
└─────────────────────────┬───────────────────────────────┘
                          │ trains
┌─────────────────────────▼───────────────────────────────┐
│                      Brain B                             │
│              (Teachable Robot Brain)                     │
│         Commands → Behaviors → Hardware Control          │
└─────────────────────────┬───────────────────────────────┘
                          │ controls
┌─────────────────────────▼───────────────────────────────┐
│                     Hardware                             │
│     OAK-D Camera │ 6-Axis Arm │ Mecanum Drive │ Audio   │
└─────────────────────────────────────────────────────────┘
```

---

## Deprecated/Legacy

The following are **no longer primary focus**:

- ~~HOPE Architecture~~ → Replaced by Learning Partner
- ~~Pi 5 + Hailo NPU~~ → OAK-D is primary perception
- ~~Gemma-370m on-device~~ → Using Seed Model v4.2.0
- ~~Manual curiosity sessions~~ → Chief Scientist automates this

---

## Success Metrics

| Goal | Current | Target | Timeline |
|------|---------|--------|----------|
| Navigation accuracy | 0% | 80% | 2 weeks |
| Face recognition | 0% | 90% | 2 weeks |
| Fetch success rate | 0% | 50% | 1 month |
| Voice command accuracy | ~60% | 90% | 2 weeks |

---

## How to Contribute

1. **Check Chief Scientist goals:** `python scripts/compound/chief_scientist.py --goals`
2. **Run a training cycle:** `python scripts/compound/learning_partner.py --train`
3. **Test on hardware:** `cd trainer_ui && python server.py`
4. **Review RLDS episodes:** Check `brain_b_data/` for training data quality

---

*This roadmap is auto-updated by Chief Scientist daemon*
