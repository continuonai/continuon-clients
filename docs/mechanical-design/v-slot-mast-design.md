# V-Slot Mast Robot Design

**Target**: 6-foot human reach (~1.8-2.0m) with stable mobile base

**Power**: EcoFlow River Max Plus (720Wh portable power station)

This document specifies a tall robot configuration using V-slot aluminum extrusion for the mast structure, enabling human-height manipulation tasks. The design leverages the EcoFlow River Max Plus as both power source and stability counterweight.

---

## 1. Architecture Overview

```
                    ┌─────────┐
                    │ Camera  │ ← OAK-D Lite (pan servo optional)
                    └────┬────┘
                         │
    ═══════════════════════════════════  ← Top cross-brace (20×20, 320mm)
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
    ─────┼─────────────────────────┼─────  ← Deck 2: Electronics deck
         │                         │         Height: 550mm (Pi, power dist)
         │                         │
    ─────┼─────────────────────────┼─────  ← Deck 1: Tool/hand swap deck
         │                         │         Height: 300mm
         │                         │
    ═════╧═════════════════════════╧═════  ← Base plate (ties mast to platform)
         │       ┌───────────┐     │
         │       │ EcoFlow   │     │     ← EcoFlow River Max Plus
         │       │ River Max │     │       290×185×236mm, 8.0kg
         │       │   Plus    │     │       720Wh power station
         │       └───────────┘     │
    ┌────┴─────────────────────────┴────┐
    │    Mecanum Drive Platform         │  ← 4WD omnidirectional base
    │    (400×300mm)                    │     4× 100mm Mecanum + encoders
    │    ◎────◎           ◎────◎        │
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
| Platform length | 400 mm | Custom aluminum plate (SendCutSend) |
| Platform width | 300 mm | Accommodates EcoFlow footprint (290mm) |
| Platform height | 50 mm | Ground to top of platform |
| **Effective footprint** | **400 × 300 mm** | No outriggers needed |
| Wheel type | **100mm Mecanum** | 4× omnidirectional wheels |
| Wheel diameter | 100 mm | Aluminum frame, rubber rollers |
| Load capacity | 45 kg (wheels) | Robot total ~13kg - plenty of margin |
| Ground clearance | ~25 mm | Under platform |
| EcoFlow position | Centered on platform | 290×185mm battery footprint |
| Drive motors | JGB37-520 12V encoder | 178 RPM, 6mm D-shaft, hall encoder |

#### Mecanum Wheel Advantages

- **Omnidirectional movement**: Strafe sideways while keeping arms/camera oriented on task
- **Precise positioning**: No 3-point turns needed for manipulation work
- **Encoder feedback**: Hall encoders enable odometry for autonomous navigation
- **12V compatible**: Same voltage rail as servo power system

### 2.2 EcoFlow River Max Plus Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dimensions | 290 × 185 × 236 mm | L × W × H |
| Weight | 8.0 kg (17.6 lbs) | Including extra battery module |
| Capacity | 720 Wh | 360Wh base + 360Wh extra battery |
| AC Output | 600W (1800W X-Boost) | 3× AC outlets |
| USB-C Output | 100W | Powers Pi 5 directly |
| USB-A Outputs | 4× ports | For accessories |
| DC Output | 12.6V / 8A (100.8W) | Via car charger port |
| Charge Time | 1.6 hours (0-80%) | X-Stream fast charging |

### 2.3 Mast Structure

| Parameter | Value | Notes |
|-----------|-------|-------|
| Upright profile | 20×40 mm V-slot | 2× parallel rails |
| Upright length | **1500 mm** | Cut to size or order |
| Upright spacing | **240 mm** center-to-center | Wider for EcoFlow clearance |
| Top cross-brace | 20×20 mm, 320 mm long | Ties uprights at top |
| Base plate | 400 × 300 × 5 mm aluminum | Bolts to platform + uprights |

### 2.4 Deck Heights & Dimensions

| Deck | Height from Ground | Size | Function |
|------|-------------------|------|----------|
| **Platform** | 0-50 mm | 400 × 300 mm | Wheels, drive motors |
| **EcoFlow** | 50-286 mm | 290 × 185 × 236 mm | Power station (integral) |
| **Deck 1** (Tool swap) | 300 mm | 300 × 220 × 3 mm | End effector storage |
| **Deck 2** (Electronics) | 550 mm | 300 × 220 × 3 mm | Pi 5, power distribution |
| **Deck 3** (Task/Payload) | 1100 mm | 300 × 220 × 3 mm | Cargo bins, items to carry |
| **Camera mount** | 1580 mm | N/A | OAK-D Lite on pan servo |

### 2.5 Arm Carriage (Moving Platform)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Carriage travel range | 400 - 1500 mm | Z-axis (height) |
| Carriage plate | 200 × 140 × 5 mm aluminum | Mounts shoulder joints |
| Gantry wheels | 4× per side (8 total) | V-slot mini wheels |
| Lift mechanism | GT2 belt + NEMA17 stepper | Or leadscrew for precision |

---

## 3. Reach Envelope Analysis

### 3.1 Height Calculation

```
Component                        Height Contribution
─────────────────────────────────────────────────────
Ground to platform top:                   50 mm
EcoFlow height:                          236 mm
Clearance above EcoFlow:                  14 mm
Base plate to mast bottom:                 0 mm
Mast usable height:                     1500 mm
Arm carriage max position:              1500 mm (top of mast travel)
Shoulder height above carriage:           50 mm
Upper arm length (SO-ARM101):            150 mm
Forearm length (SO-ARM101):              150 mm
Gripper/hand length:                     100 mm
─────────────────────────────────────────────────────
MAXIMUM VERTICAL REACH:                2050 mm (6.73 ft)
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

_Definitions_: "Carriage height" refers to the shoulder/carriage position relative to ground. "End-effector height" refers to the gripper tip at full arm extension.

| Metric | Value | Human Equivalent |
|--------|-------|------------------|
| Max vertical reach (end-effector up) | 2050 mm | 6'9" human reach |
| Max carriage height | 1550 mm | Over-shoulder height |
| Min carriage height | 450 mm | Low waist height |
| Min end-effector height (reaching down) | ~100 mm | Near-floor pickup |
| Horizontal reach | 500 mm | Short arm span |
| Carriage Z travel | 1100 mm | Full torso range |

---

## 4. Stability Analysis

### 4.1 Mass Distribution

The EcoFlow River Max Plus (8.0 kg) at the base provides exceptional stability as a low center-of-gravity counterweight.

| Component | Mass (g) | Height of CG (mm) | Moment (g·mm) |
|-----------|----------|-------------------|---------------|
| Platform + wheels + motors | 1,500 | 25 | 37,500 |
| **EcoFlow River Max Plus** | **8,000** | **168** | **1,344,000** |
| Deck 1 (plate + tools) | 300 | 300 | 90,000 |
| Deck 2 + Pi + electronics | 450 | 565 | 254,250 |
| Mast uprights (2× 20×40 Al) | 800 | 800 | 640,000 |
| Deck 3 (plate + payload) | 500 | 1100 | 550,000 |
| Arm carriage + arms (2×) | 900 | 900 (avg) | 810,000 |
| Top cross-brace + camera | 200 | 1550 | 310,000 |
| **TOTAL** | **12,650 g** | | **4,035,750** |

**Overall Center of Gravity (CG):**
- Height: 4,035,750 / 12,650 = **319 mm** above ground
- This is well below Deck 2 and only slightly above Deck 1
- The heavy EcoFlow battery lowers CG by 279 mm compared to LiPo design (598 mm → 319 mm)

### 4.2 Tip-Over Analysis

For a robot not to tip, the CG must stay within the support polygon (wheel footprint).

**Support polygon:**
- Length: 350 mm (front-to-back wheel spacing)
- Width: 250 mm (left-to-right wheel spacing)

**Critical tip angle (lateral):**
```
tan(θ_tip) = (half_width) / CG_height
           = 125 mm / 319 mm
           = 0.392
θ_tip      = 21.4°  ← EXCELLENT stability!
```

**Critical tip angle (fore-aft):**
```
tan(θ_tip) = (half_length) / CG_height
           = 175 mm / 319 mm
           = 0.549
θ_tip      = 28.8°  ← Very stable
```

**Comparison with LiPo design:**
| Configuration | CG Height | Tip Angle (lateral) | Rating |
|---------------|-----------|---------------------|--------|
| LiPo (no outriggers) | 598 mm | 4.0° | Unstable |
| LiPo (with outriggers) | 598 mm | 13.2° | Marginal |
| **EcoFlow design** | **319 mm** | **21.4°** | **Excellent** |

### 4.3 Stability Recommendations

1. **No outriggers required** - The heavy EcoFlow provides natural ballast
2. **Safe for indoor and outdoor use** - 21° tip margin handles rough surfaces
3. **Arm position less critical** - Large stability margin allows extended operation
4. **Higher speed limits available** - See calculations below

### 4.4 Maximum Safe Turn Speed

```
Maximum safe turn speed (no tip):
  v_max = sqrt(g × r × tan(θ_safe))

  Where:
    g = 9.81 m/s²
    r = turn radius = 0.5 m (typical indoor turn)
    θ_safe = 15° (half of tip angle, safety margin)

  v_max = sqrt(9.81 × 0.5 × tan(15°))
        = sqrt(9.81 × 0.5 × 0.268)
        = 1.15 m/s = 4.1 km/h
```

**Recommendation:** Safe speed up to **1.0 m/s** (~3.6 km/h) during turns.

### 4.5 Dynamic Stability with Extended Arms

When arms are extended horizontally with payload, we compute the lateral CG shift and remaining tip margin.

**Method:** The lateral CG shift is calculated as:
```
Δx = M_horizontal / m_total
```
The remaining safe tip angle is:
```
θ_safe = arctan((support_half_width - Δx) / CG_height)
```

Assuming support half-width of 125 mm and CG height of 319 mm:

| Condition | Horiz. Moment | Lateral CG Shift | Margin to Edge | Remaining Tip Angle |
|-----------|---------------|------------------|----------------|---------------------|
| Arms retracted | 0 | 0 mm | 125 mm | 21.4° |
| Arms at 500mm, no load | 90,000 g·mm | 7 mm | 118 mm | 20.3° |
| Arms at 500mm, 0.5kg/arm (1kg total) | 590,000 g·mm | 47 mm | 78 mm | 13.7° |
| Arms at 500mm, 1kg/arm (2kg total) | 1,090,000 g·mm | 86 mm | 39 mm | 7.0° |

**Conclusion:** With 1 kg total payload (500g per arm) at full 500mm extension, the robot maintains a 13.7° tip margin - still safe for normal operation. With 2 kg total payload, tip margin drops to 7° which requires careful, slow movements.

**Key Insight:** The EcoFlow-based design handles extended arm payloads far better than the LiPo design due to the 279 mm lower CG.

---

## 5. Arm Placement Recommendations

### 5.1 Shoulder Joint Configuration

```
         Carriage Plate (200 × 140 mm)
    ┌────────────────────────────────────┐
    │                                    │
    │   ┌──────┐          ┌──────┐       │
    │   │Shoulder│        │Shoulder│     │  ← Left/Right arm bases
    │   │ Left  │          │ Right │     │     Spacing: 140mm center-to-center
    │   └───┬───┘          └───┬───┘     │
    │       │                  │         │
    └───────┼──────────────────┼─────────┘
            │                  │
           Arm               Arm
```

### 5.2 Recommended Shoulder Mount Heights

| Task Type | Carriage Position | End-Effector Range | Notes |
|-----------|-------------------|-------------------|-------|
| Floor pickup | 450-550 mm | ~100-250 mm | Arms reach down |
| Table work | 700-900 mm | 400-600 mm | Standard desk height |
| Counter/shelf | 1000-1200 mm | 700-900 mm | Kitchen counter height |
| High shelf | 1300-1500 mm | 1000-1200 mm | Top of mast travel |

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

**Pricing convention:** All prices shown are **estimated line totals** (Qty × Unit Price) in USD.

### 7.1 Mobility / Base Platform (Mecanum Drive)

| Item | Qty | Source | Est. Line Total (USD) |
|------|-----|--------|----------------------|
| Platform plate (400×300×5mm Al) | 1 | SendCutSend | $35 |
| 100mm Mecanum wheels (set of 4) | 1 | Amazon (B0DL3JT2KR) | $55 |
| JGB37-520 encoder motors 12V 178RPM | 4 | Amazon (B0CCN584JW) | $50 |
| Motor brackets (if not included) | 4 | Amazon | $10 |
| BTS7960 43A H-Bridge motor driver | 1 | Amazon (B07TFB22H5) | $15 |
| 6mm shaft couplers (motor to wheel) | 4 | Amazon | $8 |
| M5 hardware, standoffs | 1 set | Generic | $12 |

**Subtotal: $185**

#### Mecanum Base Part Links

- **Wheels**: amazon.com/dp/B0DL3JT2KR (100mm, 45kg load, 4pcs)
- **Motors**: amazon.com/dp/B0CCN584JW (JGB37-520 12V 178RPM w/ encoder)
- **Driver**: amazon.com/dp/B07TFB22H5 (BTS7960 43A dual H-bridge)
- **Alt motors**: amazon.com/dp/B0DJ2WWH59 (includes wheel kit)

### 7.2 Power System (EcoFlow)

| Item | Qty | Source | Est. Line Total (USD) |
|------|-----|--------|----------------------|
| **EcoFlow River Max Plus (720Wh)** | 1 | EcoFlow / Amazon | $649 |
| USB-C PD trigger board (20V→5V for Pi) | 1 | Adafruit | $8 |
| 12V→6V buck converter (servo power) | 1 | Pololu | $15 |
| DC barrel cable for EcoFlow car port | 1 | Amazon | $8 |
| Power distribution board | 1 | Generic | $15 |
| Cable management (velcro, clips) | 1 set | Generic | $10 |

**Subtotal: $705**

### 7.3 Mast Structure (V-Slot)

| Item | Qty | Source | Est. Line Total (USD) |
|------|-----|--------|----------------------|
| V-Slot 20×40 Linear Rail, 1500mm | 2 | OpenBuilds / Bulkman3D | $30 |
| V-Slot 20×20 Linear Rail, 350mm | 1 | OpenBuilds | $7 |
| V-Slot Gantry Set 20mm (4-wheel) | 2 | ValueHobby | $20 |
| Corner brackets (90°) | 12 | OpenBuilds | $15 |
| Joining plates | 4 | OpenBuilds | $10 |
| M5 T-nuts (drop-in) | 50 | Generic | $8 |
| M5×8 button head screws | 50 | Generic | $5 |
| M5×10 button head screws | 30 | Generic | $4 |

**Subtotal: $99**

### 7.4 Deck Plates

| Item | Qty | Source | Est. Line Total (USD) |
|------|-----|--------|----------------------|
| Aluminum plate 300×220×3mm (decks) | 3 | SendCutSend / local | $45 |
| M3 standoffs assorted (10-40mm) | 20 | Generic | $8 |
| M3 screws + nuts | 50 | Generic | $5 |

**Subtotal: $58**

### 7.5 Lift Mechanism

| Item | Qty | Source | Est. Line Total (USD) |
|------|-----|--------|----------------------|
| NEMA17 stepper motor (1.5A) | 1 | StepperOnline | $12 |
| GT2 belt, 6mm, 3.5m length | 1 | Generic | $7 |
| GT2 20T pulley, 5mm bore | 2 | Generic | $4 |
| Belt tensioner/idler | 1 | OpenBuilds | $5 |
| Stepper driver (TMC2209) | 1 | BTT / Generic | $8 |

**Subtotal: $36**

### 7.6 Computing & Control

| Item | Qty | Source | Est. Line Total (USD) |
|------|-----|--------|----------------------|
| Raspberry Pi 5 (8GB) | 1 | Various | $80 |
| PCA9685 servo controller | 2 | Adafruit / Generic | $12 |
| OAK-D Lite camera | 1 | Luxonis | $150 |
| SG90 pan servo (camera) | 1 | Generic | $3 |

**Subtotal: $245**

### 7.7 Arms

| Item | Qty | Source | Est. Line Total (USD) |
|------|-----|--------|----------------------|
| SO-ARM101 (or equivalent 6-DOF arm) | 2 | Various | $200 |

**Subtotal: $200**

---

### Total Estimated Cost

| Category | Subtotal |
|----------|----------|
| Mobility/Base Platform (Mecanum) | $185 |
| Power System (EcoFlow) | $705 |
| Mast Structure | $99 |
| Deck Plates | $58 |
| Lift Mechanism | $36 |
| Computing & Control | $245 |
| Arms | $200 |
| **TOTAL** | **~$1,528** |

**Note:** The EcoFlow River Max Plus represents ~46% of the total cost but provides:
- 720Wh capacity (8× more than 89Wh LiPo)
- 8-14+ hours of operation
- Safe LFP chemistry (no fire risk)
- Built-in AC outlets for tools
- USB-C 100W for Pi 5
- Excellent stability as ballast (8kg low CG)

**Mecanum Base Upgrade:** The omnidirectional Mecanum drive adds $75 over caster/motor setup but enables:
- Strafe sideways while keeping camera/arms on target
- Precise positioning for manipulation tasks
- No 3-point turns in tight spaces
- Encoder feedback for odometry/mapping

---

## 8. Power System Details

### 8.1 Runtime Estimates

Based on 50W typical power consumption (Pi 5 + servos active):

| Mode | Power Draw | Runtime (720Wh) |
|------|------------|-----------------|
| Idle | 20W | 36 hours |
| Active Vision | 47W | 15 hours |
| Peak Motion | 62W | 11 hours |
| Typical Mixed Use | 50W | 14 hours |

### 8.2 Power Distribution

```
EcoFlow River Max Plus
├── USB-C (100W) ────────→ Raspberry Pi 5 (via USB-C PD)
├── Car Port (12V/8A) ───→ Buck Conv (12V→6V) ───→ Servo Power Rail
├── USB-A #1 ────────────→ OAK-D Lite Camera
├── USB-A #2 ────────────→ Optional peripherals
└── AC Outlet ───────────→ Power tools, charging, etc.
```

### 8.3 Charging

- **Wall charging:** 0-80% in 1.6 hours via X-Stream
- **Solar charging:** 4-8 hours with 200W panels
- **Car charging:** 7.5 hours via 12V adapter

---

## 9. Assembly Notes

### 9.1 Build Order

1. **Assemble platform** - Mount casters and drive motors
2. **Secure EcoFlow mount** - Velcro straps or retention bracket
3. **Attach base plate** - Bolt to platform above EcoFlow
4. **Install mast uprights** - Bolt to base plate with corner brackets
5. **Add Deck 1** - Tool/hand swap deck
6. **Install lift mechanism** - Motor at bottom, belt along uprights
7. **Mount carriage** - Gantry plates on uprights, belt attachment
8. **Add Deck 2** - Electronics (Pi 5, PCA9685s)
9. **Add Deck 3** - Task/payload deck
10. **Install arms** - Mount shoulder joints to carriage plate
11. **Top cross-brace + camera** - Final structural element
12. **Wire everything** - Power cables, I2C, servo cables

### 9.2 Critical Alignments

- Mast uprights must be **parallel** (use spacer during assembly)
- Carriage must travel freely (no binding) - adjust V-wheels
- Decks should be **perpendicular** to mast (acts as bracing)
- EcoFlow must be **securely retained** (vibration + tip resistance)

### 9.3 Cable Management

Route cables along mast using:
- Cable carriers (drag chain) for moving carriage
- Velcro straps for fixed cables
- Leave service loops at carriage for arm cables

---

## 10. Integration with ContinuonXR Software

### 10.1 Hardware Detection

Add lift motor to `hardware-detection.md`:
```python
# Z-axis lift stepper (NEMA17 via TMC2209)
LIFT_STEP_PIN = 18
LIFT_DIR_PIN = 19
LIFT_ENABLE_PIN = 20
```

### 10.2 Arm Manager Updates

The existing `trainer_ui/hardware/arm_manager.py` defines `ArmController` and `DualArmManager`. For the tall mast design, create a subclass that adds Z-axis control:

```python
from trainer_ui.hardware.arm_manager import DualArmManager

class TallMastDualArmManager(DualArmManager):
    """Extends DualArmManager with Z-axis lift control."""

    def __init__(self, hw_config=None):
        super().__init__(hw_config)
        self.lift_position = 450  # mm from ground (carriage height)
        self.lift_min = 450
        self.lift_max = 1500

    def move_to_height(self, target_mm: int, speed: float = 0.5):
        """Move arm carriage to specified height."""
        # Clamp to safe range
        target_mm = max(self.lift_min, min(self.lift_max, target_mm))
        # Stepper control logic via TMC2209
        # ...implementation...
        self.lift_position = target_mm
```

### 10.3 Safety Limits in Brain B

Add to `brain_b/sandbox/`:
```python
TALL_MAST_GUARDRAILS = {
    "max_speed_turning": 1.0,  # m/s (higher limit with EcoFlow stability)
    "arm_retract_before_move": False,  # Not required with stable base
    "lift_home_before_drive": False,   # Optional with good stability
    "max_payload_per_arm": 1000,  # grams (500g safe, 1kg marginal)
    "max_total_payload": 2000,  # grams at full extension
}
```

---

## 11. Future Enhancements

1. **Solar panel integration** - Mount 100W panel for outdoor extended operation
2. **Active load monitoring** - Track EcoFlow remaining capacity via USB
3. **Automatic docking** - Return to charging station when low
4. **Telescoping mast** - Extend beyond 1.5m when needed
5. **Quick-release arms** - Swap arm types without tools
6. **Payload scale** - Weigh items before lifting for safety

---

## References

- EcoFlow River Max Plus: [EcoFlow US](https://us.ecoflow.com/products/river-max-plus-portable-power-station)
- OpenBuilds V-Slot documentation: https://openbuilds.com
- Pololu regulators: https://www.pololu.com
- SO-ARM101 specs: (internal reference)
- Power-and-stacking.md: `/docs/power-and-stacking.md`
