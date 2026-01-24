# V-Slot Mast Robot Design

**Target**: 6-foot human reach (~1.8-2.0m) with stable mobile base

This document specifies a tall robot configuration using V-slot aluminum extrusion for the mast structure, enabling human-height manipulation tasks.

---

## 1. Architecture Overview

```
                    ┌─────────┐
                    │ Camera  │ ← OAK-D Lite (pan servo optional)
                    └────┬────┘
                         │
    ═══════════════════════════════════  ← Top cross-brace (20×20, 280mm)
         │                         │
         │    ┌─────────────┐      │     ← Arm carriage (gantry plates)
         │    │  Shoulder   │      │       Height: 400-1500mm (motorized)
         │    │   Joints    │      │
         │    └──┬─────┬──┘        │
         │       │     │           │     ← Dual SO-ARM101 arms (~600mm reach)
         │      ╱       ╲          │
         │     ╱         ╲         │
         │                         │
    ─────┼─────────────────────────┼─────  ← Deck 3: Task deck (bins/payload)
         │                         │         Height: 1100mm
         │                         │
         │        ↑ ↑ ↑            │
         │       V-slot            │     ← 2× 20×40 uprights (1500mm each)
         │      20×40 mm           │
         │                         │
    ─────┼─────────────────────────┼─────  ← Deck 2: Tool/hand swap deck
         │                         │         Height: 600mm
         │                         │
         │                         │
    ─────┼─────────────────────────┼─────  ← Deck 1: Electronics deck
         │                         │         Height: 80mm (battery, Pi, power)
         │                         │
    ═════╧═════════════════════════╧═════  ← Base plate (ties mast to chassis)
         │                         │
    ┌────┴─────────────────────────┴────┐
    │    4WD Chassis (210×83×65mm)      │  ← Metal Robot Chassis Kit
    │    ○────○           ○────○        │     4× DC gear motors + wheels
    └───────────────────────────────────┘
                    ↓
              ══════════════
                 Ground
```

---

## 2. Exact Dimensions

### 2.1 Base & Footprint

| Parameter | Value | Notes |
|-----------|-------|-------|
| Chassis length | 210 mm | Smart-Prototyping 4WD kit |
| Chassis width | 83 mm | Between wheel centers |
| Chassis height | 65 mm | Ground to top of chassis plate |
| **Effective footprint** | **280 × 200 mm** | Including outriggers |
| Outrigger extension | 48.5 mm each side | For stability (optional) |
| Wheel diameter | 65 mm | Standard kit wheels |
| Ground clearance | ~15 mm | Under chassis plate |

### 2.2 Mast Structure

| Parameter | Value | Notes |
|-----------|-------|-------|
| Upright profile | 20×40 mm V-slot | 2× parallel rails |
| Upright length | **1500 mm** | Cut to size or order |
| Upright spacing | **200 mm** center-to-center | Matches deck width |
| Top cross-brace | 20×20 mm, 280 mm long | Ties uprights at top |
| Base plate | 280 × 200 × 3 mm aluminum | Bolts to chassis + uprights |

### 2.3 Deck Heights & Dimensions

| Deck | Height from Ground | Size | Function |
|------|-------------------|------|----------|
| **Deck 1** (Electronics) | 80 mm | 250 × 180 × 3 mm | Battery, Pi 5, buck converters |
| **Deck 2** (Tool swap) | 600 mm | 250 × 180 × 3 mm | End effector storage, quick-swap |
| **Deck 3** (Task/Payload) | 1100 mm | 250 × 180 × 3 mm | Cargo bins, items to carry |
| **Camera mount** | 1580 mm | N/A | OAK-D Lite on pan servo |

### 2.4 Arm Carriage (Moving Platform)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Carriage travel range | 400 - 1500 mm | Z-axis (height) |
| Carriage plate | 180 × 120 × 5 mm aluminum | Mounts shoulder joints |
| Gantry wheels | 4× per side (8 total) | V-slot mini wheels |
| Lift mechanism | GT2 belt + NEMA17 stepper | Or leadscrew for precision |

---

## 3. Reach Envelope Analysis

### 3.1 Height Calculation

```
Component                        Height Contribution
─────────────────────────────────────────────────────
Ground to chassis top:                    65 mm
Chassis to Deck 1:                        15 mm
Deck 1 to base plate:                      0 mm (coplanar)
Base plate to mast bottom:                 0 mm
Mast usable height:                     1500 mm
Arm carriage max position:              1500 mm (top of mast travel)
Shoulder height above carriage:           50 mm
Upper arm length (SO-ARM101):            150 mm
Forearm length (SO-ARM101):              150 mm
Gripper/hand length:                     100 mm
─────────────────────────────────────────────────────
MAXIMUM VERTICAL REACH:                2030 mm (6.66 ft)
```

### 3.2 Horizontal Reach

```
Arm fully extended horizontally:
  Upper arm:     150 mm
  Forearm:       150 mm
  Gripper:       100 mm
  ─────────────────────
  Total:         400 mm from shoulder

Shoulder offset from mast center:    100 mm (half of carriage width)
─────────────────────────────────────────────────────────────────────
MAXIMUM HORIZONTAL REACH:            500 mm from mast centerline
```

### 3.3 Work Envelope Summary

| Metric | Value | Human Equivalent |
|--------|-------|------------------|
| Max vertical reach | 2030 mm | 6'8" human reach |
| Min vertical reach | 515 mm | Knee height |
| Horizontal reach | 500 mm | Short arm span |
| Carriage Z travel | 1100 mm | Full torso range |

---

## 4. Stability Analysis

### 4.1 Mass Distribution

| Component | Mass (g) | Height of CG (mm) | Moment (g·mm) |
|-----------|----------|-------------------|---------------|
| 4WD Chassis | 760 | 35 | 26,600 |
| Battery (3S 8000mAh) | 520 | 80 | 41,600 |
| Deck 1 + Pi + electronics | 350 | 95 | 33,250 |
| Mast uprights (2× 20×40 Al) | 800 | 800 | 640,000 |
| Deck 2 (plate + tools) | 250 | 600 | 150,000 |
| Deck 3 (plate + payload) | 500 | 1100 | 550,000 |
| Arm carriage + arms (2×) | 900 | 900 (avg) | 810,000 |
| Top cross-brace + camera | 200 | 1550 | 310,000 |
| **TOTAL** | **4280 g** | | **2,561,450** |

**Overall Center of Gravity (CG):**
- Height: 2,561,450 / 4280 = **598 mm** above ground
- This is below Deck 2, which is good for stability

### 4.2 Tip-Over Analysis

For a robot not to tip, the CG must stay within the support polygon (wheel footprint).

**Support polygon (with outriggers):**
- Length: 210 mm (front-to-back wheel spacing)
- Width: 280 mm (with outriggers) or 83 mm (without)

**Critical tip angle (no outriggers):**
```
tan(θ_tip) = (half_width) / CG_height
           = 41.5 mm / 598 mm
           = 0.069
θ_tip      = 4.0°  ← VERY UNSTABLE!
```

**Critical tip angle (with outriggers):**
```
tan(θ_tip) = (half_width) / CG_height
           = 140 mm / 598 mm
           = 0.234
θ_tip      = 13.2°  ← Acceptable for indoor use
```

### 4.3 Stability Recommendations

1. **REQUIRED: Add outriggers** - Without them, robot tips at 4° (a small bump).
2. **Optional: Wider base chassis** - Consider 300mm+ width for outdoor use.
3. **Arm position matters** - Keep arms near mast when moving, extend only when stationary.
4. **Speed limits during turns:**

```
Maximum safe turn speed (no tip) with outriggers:
  v_max = sqrt(g × r × tan(θ_safe))

  Where:
    g = 9.81 m/s²
    r = turn radius = 0.5 m (typical indoor turn)
    θ_safe = 8° (half of tip angle, safety margin)

  v_max = sqrt(9.81 × 0.5 × tan(8°))
        = sqrt(9.81 × 0.5 × 0.14)
        = 0.83 m/s = 3 km/h
```

**Recommendation:** Limit speed to **0.5 m/s** (~1.8 km/h) during turns.

### 4.4 Dynamic Stability with Extended Arms

When arms are extended horizontally with a 500g payload each:

| Condition | Added Moment | New CG Height | Tip Angle |
|-----------|--------------|---------------|-----------|
| Arms retracted | 0 | 598 mm | 13.2° |
| Arms at 500mm, no load | +90,000 g·mm horiz | 598 mm | 11.8° |
| Arms at 500mm, 1kg load | +590,000 g·mm horiz | 598 mm | 8.4° |

**Conclusion:** With 1kg payload at full extension, tip angle reduces to 8.4°.
Still safe for careful operation, but avoid sudden movements.

---

## 5. Arm Placement Recommendations

### 5.1 Shoulder Joint Configuration

```
         Carriage Plate (180 × 120 mm)
    ┌────────────────────────────────────┐
    │                                    │
    │   ┌──────┐          ┌──────┐       │
    │   │Shoulder│        │Shoulder│     │  ← Left/Right arm bases
    │   │ Left  │          │ Right │     │     Spacing: 120mm center-to-center
    │   └───┬───┘          └───┬───┘     │
    │       │                  │         │
    └───────┼──────────────────┼─────────┘
            │                  │
           Arm               Arm
```

### 5.2 Recommended Shoulder Mount Heights

| Task Type | Carriage Position | Notes |
|-----------|-------------------|-------|
| Floor pickup | 400-500 mm | Arms reach down to ~100mm |
| Table work | 700-900 mm | Standard desk height |
| Counter/shelf | 1000-1200 mm | Kitchen counter height |
| High shelf | 1300-1500 mm | Top of mast travel |

### 5.3 Arm Specifications (SO-ARM101 Reference)

| Joint | Range | Default | Channel |
|-------|-------|---------|---------|
| Base (yaw) | 0-180° | 90° | 0 |
| Shoulder (pitch) | 0-180° | 90° | 1 |
| Elbow (pitch) | 0-180° | 90° | 2 |
| Wrist pitch | 0-180° | 90° | 3 |
| Wrist roll | 0-180° | 90° | 4 |
| Gripper | 30-90° | 30° (open) | 5 |

---

## 6. Lift Mechanism Options

### Option A: Belt Drive (Recommended for Speed)

```
    Motor (NEMA17)
         │
         ├─── 20T GT2 pulley
         │
    ═════╪═════  ← Belt path (closed loop)
         │
    ┌────┴────┐
    │ Carriage │
    └────┬────┘
         │
    ═════╪═════
         │
         └─── Idler pulley (20T)
```

| Parameter | Value |
|-----------|-------|
| Belt type | GT2, 6mm width |
| Pulley teeth | 20T (12.73mm pitch dia) |
| Travel speed | ~100 mm/s |
| Holding: | Stepper detent + brake |

### Option B: Leadscrew (Recommended for Precision/Load)

| Parameter | Value |
|-----------|-------|
| Screw type | T8 ACME, 8mm lead |
| Anti-backlash nut | POM or brass |
| Travel speed | ~20-40 mm/s |
| Holding: | Self-locking (inherent) |

**Recommendation:** Belt drive for most tasks. Add brake if power-off holding is needed.

---

## 7. Bill of Materials (BOM)

### 7.1 Mobility / Base

| Item | Qty | Source | Est. Price |
|------|-----|--------|------------|
| Metal Robot Chassis Kit (4WD) | 1 | Smart-Prototyping | $25 |
| 20×40 V-slot outrigger bars (150mm) | 2 | OpenBuilds | $8 |
| Outrigger feet (rubber caps) | 4 | Generic | $4 |

### 7.2 Mast Structure (V-Slot)

| Item | Qty | Source | Est. Price |
|------|-----|--------|------------|
| V-Slot 20×40 Linear Rail, 1500mm | 2 | OpenBuilds / Bulkman3D | $30 |
| V-Slot 20×20 Linear Rail, 300mm | 1 | OpenBuilds | $6 |
| V-Slot Gantry Set 20mm (4-wheel) | 2 | ValueHobby | $20 |
| Corner brackets (90°) | 12 | OpenBuilds | $15 |
| Joining plates | 4 | OpenBuilds | $10 |
| M5 T-nuts (drop-in) | 50 | Generic | $8 |
| M5×8 button head screws | 50 | Generic | $5 |
| M5×10 button head screws | 30 | Generic | $4 |

### 7.3 Deck Plates

| Item | Qty | Source | Est. Price |
|------|-----|--------|------------|
| Aluminum plate 280×200×3mm (base) | 1 | SendCutSend / local | $15 |
| Aluminum plate 250×180×3mm (decks) | 3 | SendCutSend / local | $36 |
| M3 standoffs assorted (10-40mm) | 20 | Generic | $8 |
| M3 screws + nuts | 50 | Generic | $5 |

### 7.4 Lift Mechanism

| Item | Qty | Source | Est. Price |
|------|-----|--------|------------|
| NEMA17 stepper motor (1.5A) | 1 | StepperOnline | $12 |
| GT2 belt, 6mm, 3m length | 1 | Generic | $6 |
| GT2 20T pulley, 5mm bore | 2 | Generic | $4 |
| Belt tensioner/idler | 1 | OpenBuilds | $5 |
| Stepper driver (TMC2209) | 1 | BTT / Generic | $8 |

### 7.5 Power System

| Item | Qty | Source | Est. Price |
|------|-----|--------|------------|
| 3S 8000mAh LiPo (Gens Ace/Ovonic) | 1 | Amazon / RC shop | $50 |
| Pololu D36V50F5 (5V 5.5A) | 1 | Pololu | $15 |
| Pololu D36V50F6 (6V 5.5A) | 1 | Pololu | $15 |
| XT60 connectors (pair) | 2 | Generic | $3 |
| Inline fuse holder + 15A fuse | 1 | Generic | $4 |
| Power switch (20A) | 1 | Generic | $5 |
| 14 AWG silicone wire (red/black) | 2m | Generic | $5 |
| 18 AWG silicone wire (red/black) | 3m | Generic | $5 |

### 7.6 Computing & Control

| Item | Qty | Source | Est. Price |
|------|-----|--------|------------|
| Raspberry Pi 5 (8GB) | 1 | Various | $80 |
| PCA9685 servo controller | 2 | Adafruit / Generic | $12 |
| OAK-D Lite camera | 1 | Luxonis | $150 |
| SG90 pan servo (camera) | 1 | Generic | $3 |

### 7.7 Arms

| Item | Qty | Source | Est. Price |
|------|-----|--------|------------|
| SO-ARM101 (or equivalent 6-DOF arm) | 2 | Various | $200 |

---

### Total Estimated Cost

| Category | Subtotal |
|----------|----------|
| Mobility/Base | $37 |
| Mast Structure | $98 |
| Deck Plates | $64 |
| Lift Mechanism | $35 |
| Power System | $102 |
| Computing & Control | $245 |
| Arms | $200 |
| **TOTAL** | **~$781** |

---

## 8. Assembly Notes

### 8.1 Build Order

1. **Assemble chassis** - Mount motors/wheels per kit instructions
2. **Attach base plate** - Drill/tap holes to match chassis mounting points
3. **Install mast uprights** - Bolt to base plate with corner brackets
4. **Add Deck 1** - Battery, Pi, power distribution
5. **Install lift mechanism** - Motor at bottom, belt/screw along uprights
6. **Mount carriage** - Gantry plates on uprights, belt attachment
7. **Add Deck 2 & 3** - Cross-bracing decks
8. **Install arms** - Mount shoulder joints to carriage plate
9. **Top cross-brace + camera** - Final structural element
10. **Wire everything** - Power, I2C, servo cables

### 8.2 Critical Alignments

- Mast uprights must be **parallel** (use spacer during assembly)
- Carriage must travel freely (no binding) - adjust V-wheels
- Decks should be **perpendicular** to mast (acts as bracing)

### 8.3 Cable Management

Route cables along mast using:
- Cable carriers (drag chain) for moving carriage
- Velcro straps for fixed cables
- Leave service loops at carriage for arm cables

---

## 9. Integration with ContinuonXR Software

### 9.1 Hardware Detection

Add lift motor to `hardware-detection.md`:
```python
# Z-axis lift stepper (NEMA17 via TMC2209)
LIFT_STEP_PIN = 18
LIFT_DIR_PIN = 19
LIFT_ENABLE_PIN = 20
```

### 9.2 Arm Manager Updates

Update `trainer_ui/hardware/arm_manager.py` to include Z-axis:
```python
class TallMastArmManager(ArmManager):
    def __init__(self):
        super().__init__()
        self.lift_position = 400  # mm from ground
        self.lift_min = 400
        self.lift_max = 1500

    def move_to_height(self, target_mm: int, speed: float = 0.5):
        """Move arm carriage to specified height."""
        # Clamp to safe range
        target_mm = max(self.lift_min, min(self.lift_max, target_mm))
        # ... stepper control logic
```

### 9.3 Safety Limits in Brain B

Add to `brain_b/sandbox/`:
```python
TALL_MAST_GUARDRAILS = {
    "max_speed_turning": 0.5,  # m/s
    "arm_retract_before_move": True,
    "lift_home_before_drive": True,
    "max_payload_extended": 1000,  # grams
}
```

---

## 10. Future Enhancements

1. **Wider base** - Upgrade to 400mm width for outdoor stability
2. **Active balancing** - IMU + reaction wheels (complex)
3. **Telescoping mast** - Extend beyond 1.5m when needed
4. **Quick-release arms** - Swap arm types without tools
5. **Counterweight system** - Slide weight opposite to arm extension

---

## References

- OpenBuilds V-Slot documentation: https://openbuilds.com
- Smart-Prototyping chassis: https://www.smart-prototyping.com
- Pololu regulators: https://www.pololu.com
- SO-ARM101 specs: (internal reference)
- Power-and-stacking.md: `/docs/power-and-stacking.md`
