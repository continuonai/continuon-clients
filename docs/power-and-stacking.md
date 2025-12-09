# Power Requirements & Hardware Stacking Guide

This document details the power requirements for the ContinuonXR robot stack and provides practical stacking configurations for different deployment scenarios.

## Power Budget Analysis

### Component Power Consumption

```
┌──────────────────────────────────────────────────────┐
│ Component               │ Voltage │ Current │ Power  │
├─────────────────────────┼─────────┼─────────┼────────┤
│ Raspberry Pi 5          │ 5.1V    │ 5.0A    │ 25.5W  │
│ AI HAT+ (Hailo-8L)      │ 5.0V    │ 0.3A    │  1.5W  │
│ OAK-D Lite Camera       │ 5.0V    │ 1.0A    │  5.0W  │
│ PCA9685 #1 (idle)       │ 5-6V    │ 0.01A   │  0.1W  │
│ PCA9685 #2 (idle)       │ 5-6V    │ 0.01A   │  0.1W  │
│ SO-ARM101 Servos (12×)  │ 6.0V    │ 5.0A    │ 30.0W  │
│ INA219 Battery Mon.     │ 3.3V    │ 0.001A  │  0.0W  │
├─────────────────────────┼─────────┼─────────┼────────┤
│ TOTAL (Peak)            │ -       │ -       │ 62.2W  │
│ TOTAL (Typical)         │ -       │ -       │ 50.0W  │
│ TOTAL (Idle)            │ -       │ -       │ 20.0W  │
└──────────────────────────────────────────────────────┘
```

### Power State Breakdown

**Idle Mode** (~20W):
- Pi 5 idle: 8W
- AI HAT+ idle: 0.5W
- OAK-D standby: 1.5W
- Servos unpowered: 0W
- Support circuitry: 0.5W

**Active Vision** (~47W):
- Pi 5 active: 15W
- AI HAT+ inference: 1.5W
- OAK-D streaming: 5W
- Servos holding (12×): 20W
- Support circuitry: 0.5W

**Peak Motion** (~62W):
- Pi 5 peak: 20W
- AI HAT+ peak: 1.5W
- OAK-D peak: 5W
- Servos moving (all 12): 30W
- Support circuitry: 5W

**Sleep Learning** (~25W):
- Pi 5 training: 18W
- AI HAT+ training: 2W
- OAK-D off: 0W
- Servos off: 0W
- Storage I/O: 5W

## Power Supply Options

### Option 1: Tethered Operation (Recommended for Development)

```
Wall Outlet
    ↓
┌─────────────────────────┐
│ 27W USB-C PD PSU        │ → Raspberry Pi 5 (official)
└─────────────────────────┘

┌─────────────────────────┐
│ 6V 8A Switching PSU     │ → Servo Power Rail (2× arms)
└─────────────────────────┘
```

**Pros:**
- Unlimited runtime
- No battery management needed
- Consistent voltage
- Lower cost

**Cons:**
- Tethered to wall
- Limited mobility
- Cable management

**Cost:** ~$55
- Pi 5 27W PSU: $12
- 6V 8A PSU: $25
- Cables/connectors: $18

### Option 2: Battery Powered (Mobile Robot)

```
┌─────────────────────────┐
│ 3S LiPo Battery         │
│ 11.1V nominal, 8000mAh  │
│ ~89Wh capacity          │
└─────────────────────────┘
    ↓           ↓
Buck Conv.  Buck Conv.
5.1V/5A     6.0V/8A
    ↓           ↓
  Pi 5      Servos (2× arms)
```

**Runtime Estimates:**
- Idle: ~4.5 hours (89Wh / 20W)
- Active: ~1.9 hours (89Wh / 47W)
- Peak: ~1.4 hours (89Wh / 62W)
- Mixed use: ~1.5-2 hours realistic

**Pros:**
- Fully mobile
- Self-charging capable
- Professional robot deployment

**Cons:**
- Battery maintenance required
- Higher complexity
- Added weight (~300g)
- Safety considerations (LiPo)

**Cost:** ~$185
- 3S 8000mAh LiPo: $65
- 2× Buck converters (higher rated): $40
- Battery monitor (INA219): $10
- BMS/charging circuit: $25
- Pogo pins/dock hardware: $25
- Connectors/wiring: $20

### Option 3: Hybrid (Battery + Tethered Backup)

```
┌─────────────────────────┐
│ Charging Dock (12V 5A)  │
└─────────────────────────┘
         ↓ (when docked)
┌─────────────────────────┐
│ 3S LiPo + BMS           │ ← Auto-switch to dock power
└─────────────────────────┘
    ↓           ↓
Buck Conv.  Buck Conv.
```

**Best of both worlds:**
- Mobile when needed
- Auto-charges when docked
- Continuous operation during charging
- Graceful power transitions

**Cost:** ~$235 (Option 2 + dock)

### Option 4: Budget Mobile (Power Bank)

```
┌─────────────────────────┐
│ USB-C PD Power Bank     │
│ 20000mAh, 65W output    │ → Pi 5 via USB-C PD
└─────────────────────────┘
         +
┌─────────────────────────┐
│ 6V USB Power Supply     │ → Servos via USB→6V adapter
└─────────────────────────┘
```

**Runtime:** ~2-3 hours typical use

**Pros:**
- Off-the-shelf components
- No LiPo charging/safety concerns
- Easy to swap batteries
- Can charge via any USB-C

**Cons:**
- Bulkier than LiPo
- Two separate power sources
- Less elegant solution
- No integrated monitoring

**Cost:** ~$80
- 65W USB-C power bank: $50
- USB to 6V converter: $15
- Cables: $15

## Physical Stacking Configurations

### Configuration A: Desktop Development Stack

```
┌─────────────────────────┐
│  Breadboard             │
│  - PCA9685 #1 (0x40)    │ ← Jumper wires to Pi GPIO
│  - PCA9685 #2 (0x41)    │    (shared I2C bus)
│  - INA219 (optional)    │
│  - Test circuits        │
└─────────────────────────┘

┌─────────────────────────┐
│  SO-ARM101 Left Arm     │ ← PWM from PCA9685 #1
│  (on desk/mount)        │    Power: External 6V PSU
└─────────────────────────┘
┌─────────────────────────┐
│  SO-ARM101 Right Arm    │ ← PWM from PCA9685 #2
│  (on desk/mount)        │    Power: Shared 6V PSU
└─────────────────────────┘

┌─────────────────────────┐
│  OAK-D Lite Camera      │ ← USB 3.0 cable
│  (on tripod/mount)      │
└─────────────────────────┘

┌─────────────────────────┐
│  AI HAT+                │ ← M.2 Key M
└─────────────────────────┘
┌─────────────────────────┐
│  Raspberry Pi 5         │ ← 27W USB-C PSU
│  (on desk)              │
└─────────────────────────┘
```

**Use Case:** Development, testing, algorithm work  
**Mobility:** None (desk-bound)  
**Power:** Tethered (Option 1)  
**Cost:** ~$55 (PSUs only)

### Configuration B: Compact HAT Stack (Tethered)

```
┌─────────────────────────┐
│  Proto HAT              │
│  - PCA9685 #1 (0x40)    │ ← Sits on GPIO pins
│  - PCA9685 #2 (0x41)    │    (chained I2C)
│  - INA219 sensor        │
│  - Power distribution   │
└─────────────────────────┘
┌─────────────────────────┐
│  AI HAT+                │ ← M.2 + HAT standoffs
└─────────────────────────┘
┌─────────────────────────┐
│  Raspberry Pi 5         │ ← 27W PSU
│  + heatsink/fan         │
└─────────────────────────┘
        │
        └─→ Servo cable harness
        └─→ OAK-D USB 3.0

```

**Use Case:** Lab bench, semi-permanent installation  
**Mobility:** Portable but needs power  
**Power:** Tethered (Option 1)  
**Cost:** ~$75 (PSUs + proto HAT)

**Advantages:**
- Compact vertical stack
- Clean wiring
- Easy to move as unit
- Professional appearance

### Configuration C: Mobile Robot Platform

```
┌─────────────────────────┐
│  Camera Mast            │
│  - OAK-D Lite           │ ← USB 3.0 to Pi
│  - Optional: pan servo  │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Top Deck               │
│  - SO-ARM101 Left Arm   │
│  - SO-ARM101 Right Arm  │
│  - Servo wiring         │
└─────────────────────────┘
┌─────────────────────────┐
│  Middle Deck            │
│  - AI HAT+ stack        │ ← M.2 + GPIO
│  - Raspberry Pi 5       │
│  - PCA9685 #1 (0x40)    │
│  - PCA9685 #2 (0x41)    │
└─────────────────────────┘
┌─────────────────────────┐
│  Bottom Deck            │
│  - 3S LiPo (8000mAh)    │
│  - Buck converters      │
│  - BMS/power dist.      │
│  - Pogo pins (charging) │
└─────────────────────────┘
┌─────────────────────────┐
│  Wheel Base             │
│  - Drive motors         │
│  - Motor controller     │
│  - Odometry encoders    │
└─────────────────────────┘
```

**Use Case:** Dual-arm mobile manipulation  
**Mobility:** Full (self-charging capable)  
**Power:** Battery (Option 2/3)  
**Cost:** ~$185-235 (power system) + chassis

**Dimensions (typical):**
- Footprint: 250mm × 200mm (wider for dual arms)
- Height: 350mm (with camera mast)
- Weight: ~2.5kg fully loaded

### Configuration D: Arm-Only Testbed

```
┌─────────────────────────┐
│  OAK-D Lite             │ ← Fixed mount, looking at workspace
│  (overhead/side mount)  │
└─────────────────────────┘

┌─────────────────────────┐
│  SO-ARM101 Left + Right │ ← Dual workspace manipulators
│  (fixed to table)       │
└─────────────────────────┘
         ↑
    PWM signals
         ↓
┌─────────────────────────┐
│  PCA9685 #1 & #2        │ ← On breadboard or PCB
└─────────────────────────┘
┌─────────────────────────┐
│  AI HAT+                │
└─────────────────────────┘
┌─────────────────────────┐
│  Raspberry Pi 5         │ ← Fixed position, desk/shelf
└─────────────────────────┘
```

**Use Case:** Dual-arm VLA training, bimanual tasks  
**Mobility:** Arms only (fixed base)  
**Power:** Tethered (Option 1)  
**Cost:** ~$55 (minimal)

**Advantages:**
- Repeatable workspace
- Maximum stability
- Easy cable management
- Best for data collection

## Recommended Component Specs

### Buck Converters

**For Pi 5 (5.1V rail):**
- Input: 7-24V DC
- Output: 5.0-5.2V adjustable
- Current: 6A continuous minimum
- Efficiency: >90%
- Features: Overcurrent protection, thermal shutdown
- Example: DROK LM2596 with voltage display

**For Servos (6.0V rail):**
- Input: 7-24V DC
- Output: 5.5-6.5V adjustable
- Current: 8A continuous minimum (for dual arms)
- Features: Low ripple (<50mV), stable under load
- Example: Pololu D24V90F6 or dual D24V50F6

### Battery Specifications

**3S LiPo (Recommended for Dual Arms):**
- Voltage: 11.1V nominal (9.9-12.6V range)
- Capacity: 8000-10000mAh (larger for dual arms)
- Discharge: 30C minimum (50C better)
- Connector: XT60 or Deans
- Protection: Built-in or external BMS required
- Example: Zeee 3S 8000mAh 50C or Turnigy 10000mAh

**Safety Requirements:**
- LiPo-safe charging bag
- Temperature monitoring
- Voltage cutoff: 9.0V absolute minimum
- Balance charging required
- Fire extinguisher nearby

**Alternative (Safer):** 3S Li-ion
- Same voltage but lower energy density
- Much safer chemistry
- Longer cycle life
- Example: 3× 18650 cells in holder

### Power Bank (Budget Option)

**Specs:**
- Capacity: 20000mAh minimum
- Output: 65W USB-C PD minimum
- Ports: 1× USB-C PD + 1× USB-A QC
- Features: Passthrough charging, PPS support
- Example: Anker 737 PowerCore 24000mAh

## Wiring Best Practices

### Wire Gauge Selection

```
Current Rating → Wire Gauge (AWG)
─────────────────────────────────
< 1A           → 24 AWG
1-3A           → 22 AWG
3-5A           → 20 AWG
5-10A          → 18 AWG
10-15A         → 16 AWG
15-20A         → 14 AWG
```

**Recommended for this stack:**
- Main power (battery to converters): 18 AWG
- Pi 5 power: 20 AWG (short runs) or 18 AWG
- Servo power rail: 18 AWG
- Individual servo leads: 22 AWG (factory)
- Signal wires (I2C, PWM): 24-26 AWG

### Connector Standards

**Power Distribution:**
- Battery: XT60 (main), JST-XH (balance)
- Buck converter input: Screw terminals or XT60
- Pi 5: Official USB-C cable or GPIO pins 2/4 (5V)
- Servos: JST or Dupont (factory standard)

**Signal/Data:**
- I2C: Dupont female or JST-PH 4-pin
- PWM: Dupont female 3-pin
- USB: Standard Type-C and Type-A

### Grounding Strategy

```
Single Point Ground (Star Configuration)
─────────────────────────────────────────

                Battery Negative
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
    Buck Conv 1   Buck Conv 2    INA219
      GND           GND           GND
          ↓             ↓             ↓
        Pi 5        Servos      Monitoring
       (digital)   (motor)      (analog)
```

**Critical Rules:**
1. **Single ground reference** at battery negative
2. **No ground loops** between power supplies
3. **Star topology** not daisy-chain
4. **Separate digital/motor grounds** until star point
5. **Twisted pairs** for I2C signals with ground

## Thermal Management

### Heat Generation by Component

**Raspberry Pi 5:** 5-15W dissipated
- Official active cooler: **Required**
- Passive heatsink: Insufficient under load
- Target: <70°C under continuous load

**AI HAT+ (Hailo-8L):** 1-2W dissipated
- Included heatsink: Usually sufficient
- Add small fan if stacking closely
- Target: <60°C during inference

**Buck Converters:** 1-3W each
- Heatsink included on most modules
- Airflow beneficial
- Target: <80°C

**Servos (SO-ARM101):** 1-2W each when active
- Natural convection adequate
- Duty cycle <50% recommended
- Target: <60°C case temperature

### Cooling Solutions

**Minimal (Desktop):**
- Pi 5 official active cooler
- Open air circulation
- Cost: $5

**Standard (Mobile Robot):**
- Pi 5 active cooler
- HAT+ heatsink
- Perforated chassis panels
- Cost: $10

**Advanced (Enclosed Robot):**
- Pi 5 active cooler
- 40mm fan over HAT stack
- Buck converter heatsinks
- Vented enclosure design
- Cost: $20

## Power Monitoring & Safety

### Essential Monitoring

**Battery Voltage:**
- Monitor via INA219 sensor
- Alert at <10.5V (3.5V/cell)
- Emergency shutdown at <9.9V (3.3V/cell)
- Log voltage every 10 seconds

**Current Draw:**
- Monitor total system current
- Alert if >8A continuous
- Track power state vs. current
- Useful for detecting mechanical issues

**Temperature:**
- Pi 5 CPU temp (vcgencmd)
- Battery temperature (if sensor available)
- Servo temperature (thermal camera optional)

### Safety Features (Recommended)

1. **Software:**
   - Low voltage cutoff (LVC) in startup_manager.py
   - Overcurrent detection
   - Thermal throttling awareness
   - Graceful shutdown on battery critical

2. **Hardware:**
   - Inline fuse: 10A on battery output
   - Reverse polarity protection diodes
   - Buck converter overcurrent limits
   - BMS with short circuit protection

3. **Mechanical:**
   - LiPo charging bag/box
   - Battery mounting: velcro + retention strap
   - Servo wire strain relief
   - Prevent short circuits (insulated standoffs)

## Stacking Hardware & Mounting

### Standoff Specifications

**Pi 5 to AI HAT+:**
- M2.5 × 11mm standoffs (GPIO HAT height)
- Metal (brass) preferred for grounding
- Nylon acceptable for insulation

**Multi-deck Robot:**
- M3 × 50mm standoffs between decks
- Aluminum for strength + light weight
- Use locknuts to prevent vibration loosening

**Camera Mount:**
- 1/4"-20 thread (standard tripod)
- Ball joint for adjustability
- Or fixed bracket for repeatability

### Mounting Plates

**Material Options:**
- **Acrylic (3mm):** Cheap, easy to laser cut, insulating
- **Aluminum (2mm):** Strong, conductive (grounding plane)
- **3D Printed (PLA/PETG):** Custom shapes, moderate strength
- **Carbon Fiber:** Premium, light, strong, expensive

**Recommended Deck Size:** 250mm × 200mm for dual SO-ARM101 robot (wider for side-by-side arms)

### Assembly Tips

1. **Build bottom-up:** Start with battery deck, add converters, then compute
2. **Wire as you stack:** Don't stack fully then try to wire
3. **Test each layer:** Verify power before adding next component
4. **Label everything:** Wire colors, voltage rails, I2C addresses
5. **Cable management:** Use zip ties, spiral wrap, or cable channels
6. **Allow disassembly:** Don't make servicing impossible
7. **Document:** Take photos at each stage

## Cost Summary by Configuration

| Configuration | Power System | Hardware | Total | Runtime |
|--------------|--------------|----------|-------|---------|
| Desktop Dev  | $55          | $15      | $70   | ∞       |
| Compact HAT  | $55          | $45      | $100  | ∞       |
| Mobile Robot | $185         | $95      | $280  | 1.5-2hr |
| Power Bank   | $90          | $35      | $125  | 2-3hr   |
| Hybrid Dock  | $235         | $120     | $355  | ∞*      |

*With auto-charging system (see docs/self-charging-system.md)

## Power Optimization Tips

### Software Optimizations

1. **Idle Components When Unused:**
   ```python
   # Turn off camera when not needed
   camera.stop_capture()
   
   # Release servo power
   arm.disable_torque()
   
   # Reduce CPU frequency
   subprocess.run(['sudo', 'cpufreq-set', '-g', 'powersave'])
   ```

2. **Batch Processing:**
   - Process frames at 10Hz not 30Hz when possible
   - Run inference on keyframes only
   - Use lower resolution for non-critical vision

3. **Sleep Learning Optimization:**
   - Disable camera completely
   - Dim HDMI output (or disable)
   - Lower CPU clock during training

### Hardware Optimizations

1. **Use Efficient Buck Converters:** 95% efficiency vs 85% saves ~2W
2. **Disable Unused Pi Peripherals:** HDMI, WiFi (if using Ethernet)
3. **Quality Servos:** Metal gear servos more efficient than plastic
4. **Proper Voltage:** 6.0V for servos (not 7.4V) reduces wasted heat

## Future Power Enhancements

### Upcoming Considerations

1. **Solar Panel Integration:**
   - 10W solar panel: ~$25
   - MPPT charge controller: ~$15
   - Trickle charge during outdoor use
   - Extends runtime by 15-30%

2. **Hot-Swap Battery:**
   - Dual battery system with isolation diodes
   - Swap one battery while other powers system
   - Zero downtime operation

3. **Wireless Charging:**
   - Qi coils (50W): ~$40
   - No mechanical wear (vs pogo pins)
   - Slower charging but more reliable

4. **Supercapacitor Backup:**
   - 50F supercap: ~$20
   - Handles peak current spikes
   - Reduces battery stress
   - Allows graceful shutdown on power loss

5. **Energy Harvesting:**
   - Regenerative braking on motors
   - Kinetic energy recovery
   - Experimental but interesting

## Troubleshooting Power Issues

### Pi 5 Won't Boot

**Symptom:** Rainbow square, no HDMI output  
**Cause:** Insufficient power (voltage drop or current limit)  
**Solution:**
- Check voltage at GPIO pins (measure pin 2/4 to pin 6)
- Should be 5.0-5.2V under load
- Use thicker wires or shorter runs
- Verify PSU can deliver 5A

### Servos Jittering

**Symptom:** Servos shake or oscillate when holding position  
**Cause:** Voltage sag, noise on power rail, or insufficient current  
**Solution:**
- Add 1000-2200µF capacitor across servo power rail
- Use separate buck converter for servos
- Ensure ground is shared between Pi and PCA9685
- Check PWM signal quality with scope

### Random Reboots

**Symptom:** Pi restarts unexpectedly during servo motion  
**Cause:** Voltage dip when motors draw current  
**Solution:**
- Separate power rails (Pi vs servos)
- Increase battery capacity
- Add bulk capacitance (2200µF+)
- Reduce simultaneous servo movements

### Battery Drains Too Fast

**Symptom:** <1 hour runtime on 5000mAh battery  
**Cause:** Excessive current draw or bad battery  
**Solution:**
- Use INA219 to measure actual current
- Check for short circuits or stuck servos
- Test battery capacity with hobby charger
- Optimize software (disable unused features)

### Buck Converter Overheating

**Symptom:** Converter very hot (>80°C), possibly shutting down  
**Cause:** Overloaded or inefficient  
**Solution:**
- Verify current draw is within spec
- Add heatsink or airflow
- Use higher current-rated converter
- Check input voltage (higher input = more heat)

## References & Resources

### Datasheets
- Raspberry Pi 5 Power Guide: https://www.raspberrypi.com/documentation/computers/raspberry-pi-5.html#power-supply
- Hailo-8L AI HAT+: https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator/
- PCA9685 Servo Driver: https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf
- INA219 Current Sensor: https://www.ti.com/lit/ds/symlink/ina219.pdf

### Calculators
- Wire gauge calculator: https://www.calculator.net/wire-size-calculator.html
- Battery runtime calculator: https://www.batterystuff.com/kb/tools/calculator-sizing-a-12-volt-battery-to-a-load.html
- Buck converter efficiency: https://www.ti.com/tool/WEBENCH-POWER-DESIGNER

### Safety Guides
- LiPo battery safety: https://rogershobbycenter.com/lipoguide
- Electrical safety basics: https://learn.sparkfun.com/tutorials/how-to-use-a-multimeter

### Suppliers
- Adafruit (sensors, breakouts): https://www.adafruit.com
- Pololu (buck converters, motors): https://www.pololu.com
- HobbyKing (LiPo batteries): https://hobbyking.com
- Amazon/AliExpress (bulk components)

## Appendix: Power Distribution PCB Design

For production robots, consider designing a custom power distribution board:

**Features:**
- Single PCB combines buck converters, monitoring, BMS
- Screw terminals for battery input
- Barrel jacks or USB-C for outputs
- Onboard INA219 + ESP32 for wireless monitoring
- Protection: Fuses, TVS diodes, reverse polarity
- LED indicators for each rail

**Cost:** ~$50 for 5 boards from JLCPCB (fully assembled)

**Reference Design:** Available in `/hardware/power-distribution-pcb/` (future)
