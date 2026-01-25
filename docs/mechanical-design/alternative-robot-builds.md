# Alternative Robot Builds

This document provides alternative robot configurations without the EcoFlow battery, exploring different battery technologies and home-purpose robot chassis options.

**Reference:** For the EcoFlow-based tall mast design, see [v-slot-mast-design.md](./v-slot-mast-design.md)

---

## Battery Alternatives Overview

The EcoFlow River Max Plus ($649, 720Wh, 8.0kg) provides excellent power and stability-through-mass, but alternatives exist for different budgets and use cases.

| Battery Type | Capacity | Weight | Cost | Runtime* | Stability Contribution |
|--------------|----------|--------|------|----------|------------------------|
| EcoFlow River Max Plus | 720Wh | 8.0kg | $649 | 14+ hrs | Excellent (ballast) |
| **LiFePO4 12V 20Ah** | 256Wh | 2.8kg | $120 | 5 hrs | Good |
| **Milwaukee M18 HD12.0** | 216Wh | 1.3kg | $200 | 4 hrs | Moderate |
| **E-bike Battery 48V 15Ah** | 720Wh | 4.5kg | $180 | 14 hrs | Good |
| **18650 Custom Pack** | 150-300Wh | 1-2kg | $80-150 | 3-6 hrs | Low |
| **Jackery 300 Plus** | 288Wh | 3.8kg | $200 | 6 hrs | Good |
| **Lead Acid (SLA) 12V 18Ah** | 216Wh | 5.5kg | $45 | 4 hrs | Good (ballast) |

*Runtime based on 50W typical draw

---

## Battery Option 1: LiFePO4 12V 20Ah

**Best for:** Budget builds, safety-conscious, long cycle life

```
┌─────────────────────────────────────┐
│  LiFePO4 12V 20Ah Battery           │
│  Dimensions: 181 × 77 × 168 mm      │
│  Weight: 2.8 kg                     │
│  Capacity: 256Wh                    │
│  Cycles: 2000+                      │
└─────────────────────────────────────┘
         │
    ┌────┴────┐
    │ 12V Bus │
    └────┬────┘
         ├──→ Buck 12V→5V (5A) ──→ Pi 5
         ├──→ Buck 12V→6V (8A) ──→ Servos
         └──→ Direct 12V ──→ Drive motors
```

### Specifications

| Parameter | Value |
|-----------|-------|
| Chemistry | LiFePO4 (LFP) |
| Nominal Voltage | 12.8V (4S) |
| Capacity | 20Ah / 256Wh |
| Max Discharge | 20A continuous |
| BMS | Built-in (overcurrent, undervoltage, temp) |
| Charging | 14.6V CC/CV, 5A typical |
| Cycle Life | 2000+ cycles to 80% |
| Operating Temp | -20°C to 60°C |

### Advantages

- **Safe chemistry** - No thermal runaway risk, no fire hazard
- **Long lifespan** - 2000+ cycles vs 300-500 for LiPo
- **Built-in BMS** - No external protection needed
- **Flat discharge curve** - Consistent voltage throughout use
- **Wide temp range** - Works in cold environments

### Disadvantages

- **Lower energy density** - Larger than equivalent LiPo
- **Lower stability contribution** - 2.8kg vs 8kg EcoFlow
- **Requires counterweight** - Add ballast for tall mast stability

### Recommended Products

| Product | Capacity | Weight | Price | Source |
|---------|----------|--------|-------|--------|
| Ampere Time 12V 20Ah | 256Wh | 2.8kg | $110 | Amazon |
| CHINS 12V 20Ah Mini | 256Wh | 2.5kg | $100 | Amazon |
| Renogy 12V 20Ah | 256Wh | 2.9kg | $130 | Renogy |
| Weize 12V 20Ah | 256Wh | 2.6kg | $90 | Amazon |

### Power Distribution

```
┌────────────────────────────────────────────────────────┐
│ LiFePO4 12V 20Ah                                       │
│ (+) ───┬─── Fuse 15A ───┬─── Buck 5V/6A ──→ Pi 5      │
│        │                │                              │
│        │                ├─── Buck 6V/10A ──→ Servos   │
│        │                │                              │
│        │                └─── Direct ──→ Motors/Lift   │
│ (-) ───┴─── Common GND ─────────────────────────────── │
└────────────────────────────────────────────────────────┘
```

### BOM for LiFePO4 Build

| Item | Cost |
|------|------|
| LiFePO4 12V 20Ah battery | $100 |
| Buck converter 5V 6A | $12 |
| Buck converter 6V 10A | $15 |
| Fuse holder + 15A fuse | $5 |
| Wiring/connectors | $15 |
| **Total Power System** | **$147** |

**Savings vs EcoFlow:** $558 (79% less)

---

## Battery Option 2: Milwaukee M18 Tool Batteries

**Best for:** Makers with existing tool batteries, hot-swap capability

```
┌─────────────────────────────┐
│  Milwaukee M18 HD12.0       │
│  (or M18 CP3.0 for lighter) │
│  Weight: 1.3kg (HD12)       │
│  Capacity: 216Wh            │
└─────────────────────────────┘
         │
    ┌────┴────┐
    │ M18 Dock│ ← Custom mount or commercial adapter
    └────┬────┘
         │ 18V nominal (20V max)
         │
    ┌────┴────┐
    │ DC-DC   │
    └────┬────┘
```

### Why Tool Batteries?

1. **Hot-swap capability** - Swap batteries in seconds
2. **Existing ecosystem** - Many makers already own M18 batteries
3. **Quality cells** - Premium cells with excellent BMS
4. **Standard form factor** - Multiple capacity options
5. **Easy charging** - Use existing chargers

### Milwaukee M18 Battery Options

| Model | Capacity | Weight | Cost | Runtime |
|-------|----------|--------|------|---------|
| M18 CP2.0 | 36Wh | 0.35kg | $40 | 45 min |
| M18 CP3.0 | 54Wh | 0.45kg | $50 | 1 hr |
| M18 XC5.0 | 90Wh | 0.68kg | $80 | 1.8 hrs |
| M18 XC8.0 | 144Wh | 0.95kg | $130 | 2.9 hrs |
| **M18 HD12.0** | **216Wh** | **1.3kg** | **$200** | **4.3 hrs** |
| M18 FORGE 6.0 | 108Wh | 0.68kg | $90 | 2.2 hrs |

### Adapter Options

**Option A: DIY Dock**
```
M18 Battery
    ↓
3D Printed Dock with contacts
    ↓
XT60 connector → 18V bus
```
- Cost: $5-10 (print + contacts)
- Requires careful contact alignment

**Option B: Commercial Adapter**
- M18 to USB-C PD adapter: ~$30
- M18 to DC barrel adapter: ~$25
- Powers Pi directly, needs separate servo power

**Option C: Dual Battery Setup**
```
M18 HD12.0 #1 ──→ Pi 5 + Electronics
M18 XC5.0 #2 ──→ Servos + Motors
```
- Separate power domains
- Hot-swap either independently

### BOM for M18 Build (assuming you own batteries)

| Item | Cost |
|------|------|
| M18 3D printed dock | $10 |
| M18 to DC adapter board | $25 |
| Buck converter 5V 6A | $12 |
| Buck converter 6V 10A | $15 |
| Wiring/connectors | $15 |
| **Total Power System** | **$77** (batteries owned) |

| Item | Cost |
|------|------|
| M18 HD12.0 battery | $200 |
| M18 3D printed dock | $10 |
| DC-DC converters + wiring | $40 |
| **Total with battery** | **$250** |

### DeWalt/Makita Alternatives

The same approach works with other tool battery ecosystems:

| Brand | Battery | Capacity | Weight | Cost |
|-------|---------|----------|--------|------|
| DeWalt | DCB612 (12Ah) | 216Wh | 1.4kg | $180 |
| Makita | BL1860B (6Ah) | 108Wh | 0.68kg | $90 |
| Ryobi | PBP007 (6Ah) | 108Wh | 0.95kg | $100 |

---

## Battery Option 3: E-bike Battery Pack

**Best for:** Maximum capacity on budget, existing e-bike owners

```
┌─────────────────────────────────────────┐
│  E-bike Battery 48V 15Ah                │
│  Dimensions: ~365 × 90 × 90 mm (tube)   │
│  Weight: 4.5 kg                         │
│  Capacity: 720Wh                        │
│  Form: Downtube, Hailong, Shark         │
└─────────────────────────────────────────┘
         │
    ┌────┴────┐
    │ 48V Bus │ (42-54.6V range)
    └────┬────┘
         │
    ┌────┴─────────────────────────┐
    │ DC-DC Step-down 48V→12V 10A │
    └────┬─────────────────────────┘
         │ 12V intermediate
         ├──→ Buck 5V ──→ Pi 5
         ├──→ Buck 6V ──→ Servos
         └──→ Direct ──→ Motors
```

### Why E-bike Batteries?

1. **Best Wh/$ ratio** - 720Wh for ~$180
2. **High quality cells** - Samsung, LG, Panasonic 18650/21700
3. **Built-in BMS** - Full protection circuits
4. **Standard connectors** - XT60, Anderson, DC barrel
5. **Available everywhere** - Amazon, AliExpress, local shops

### Form Factors

| Style | Dimensions | Weight | Mounting |
|-------|------------|--------|----------|
| Hailong | 365×90×90mm | 3-5kg | Bolt-on frame |
| Downtube | 400×85×85mm | 3-5kg | Water bottle mount |
| Rear Rack | 370×145×85mm | 4-6kg | Flat top surface |
| Triangle | Custom | 4-8kg | Frame triangle |

**Recommended: Hailong or Rear Rack** for robot mounting

### Specifications (48V 15Ah Example)

| Parameter | Value |
|-----------|-------|
| Chemistry | Li-ion (NMC) 18650 or 21700 |
| Configuration | 13S 5P (typical) |
| Nominal Voltage | 48V (46.8V actual) |
| Full Charge | 54.6V |
| Cutoff | 39V |
| Capacity | 15Ah / 720Wh |
| Max Discharge | 30A continuous |
| BMS | Built-in (balance, overcurrent, temp) |
| Charger | 54.6V 2-3A included |

### Voltage Conversion

E-bike batteries run at 48V nominal, requiring step-down conversion:

```
48V ──→ [48V→12V Converter] ──→ 12V ──→ [12V→5V] ──→ Pi
                                    └──→ [12V→6V] ──→ Servos
```

**OR** direct high-voltage converters:

```
48V ──→ [48V→5V 10A Converter] ──→ Pi + Electronics
    └──→ [48V→6V 15A Converter] ──→ Servos
```

### Recommended Converters

| Converter | Input | Output | Current | Cost |
|-----------|-------|--------|---------|------|
| DROK 48V→12V | 36-72V | 12V | 10A | $18 |
| SMAKN 48V→5V | 36-75V | 5V | 10A | $15 |
| Kohree 48V→12V | 30-60V | 12V | 20A | $25 |

### BOM for E-bike Battery Build

| Item | Cost |
|------|------|
| 48V 15Ah Hailong battery | $180 |
| 48V→12V converter 10A | $20 |
| Buck 12V→5V 6A | $12 |
| Buck 12V→6V 10A | $15 |
| Mount bracket | $15 |
| Wiring/connectors | $15 |
| **Total Power System** | **$257** |

**Same capacity as EcoFlow (720Wh) for 63% less cost**

---

## Battery Option 4: Custom 18650/21700 Pack

**Best for:** Maximum customization, weight optimization

```
┌─────────────────────────────────────┐
│  Custom 4S6P 18650 Pack             │
│  Cells: 24× Samsung 30Q (3000mAh)   │
│  Dimensions: ~150 × 100 × 70 mm     │
│  Weight: 1.2 kg                     │
│  Capacity: 266Wh                    │
└─────────────────────────────────────┘
```

### Cell Options

| Cell | Capacity | Discharge | Cost/cell | Pack 4S6P |
|------|----------|-----------|-----------|-----------|
| Samsung 30Q | 3000mAh | 15A | $4 | $96 |
| Samsung 35E | 3500mAh | 8A | $4 | $96 |
| LG MJ1 | 3500mAh | 10A | $4 | $96 |
| Molicel P28A | 2800mAh | 35A | $5 | $120 |
| Samsung 50E (21700) | 5000mAh | 10A | $6 | $144 |

### Configuration Options

**4S6P (14.8V, 18Ah):**
- Voltage: 14.8V nominal (16.8V full)
- Capacity: 18Ah / 266Wh
- Weight: ~1.2kg
- Perfect for direct 12V compatibility

**6S4P (22.2V, 12Ah):**
- Voltage: 22.2V nominal
- Capacity: 12Ah / 266Wh
- Weight: ~1.2kg
- Good for motor efficiency

### Required Components

| Component | Purpose | Cost |
|-----------|---------|------|
| 24× 18650 cells | Energy storage | $96 |
| 4S BMS 30A | Protection | $8 |
| 4S battery holder/case | Physical structure | $15 |
| Nickel strip 0.15mm | Cell connections | $10 |
| XT60 connectors | Power output | $5 |
| Heat shrink/fish paper | Insulation | $5 |
| Balance lead connector | Charging | $3 |
| **Total** | | **$142** |

### Build Notes

**Tools Required:**
- Spot welder (DIY or commercial)
- Multimeter
- Heat gun
- Soldering iron (for BMS connections only)

**Safety:**
- Never solder directly to cells
- Use appropriate BMS for cell chemistry
- Include balance charging capability
- Test pack capacity after assembly

---

## Battery Option 5: Compact Power Stations

**Best for:** Convenience, safety, integrated charging

Smaller alternatives to EcoFlow that still provide the "power station" convenience:

| Model | Capacity | Weight | Cost | Outputs |
|-------|----------|--------|------|---------|
| **Jackery 300 Plus** | 288Wh | 3.8kg | $200 | USB-C 100W, 12V |
| Anker 521 | 256Wh | 4.5kg | $200 | USB-C 65W, AC |
| Bluetti EB3A | 268Wh | 4.6kg | $200 | USB-C 100W, 12V |
| Goal Zero Yeti 200X | 187Wh | 2.3kg | $300 | USB-C 60W, 12V |
| EcoFlow River 2 | 256Wh | 3.5kg | $200 | USB-C 100W, AC |

### Jackery 300 Plus Build

```
┌──────────────────────────────────────┐
│  Jackery 300 Plus                    │
│  Dimensions: 230 × 155 × 167 mm      │
│  Weight: 3.8 kg                      │
│  LiFePO4 chemistry (3000+ cycles)    │
└──────────────────────────────────────┘
         │
    ┌────┼───────────────────────────┐
    │    │                           │
    ↓    ↓                           ↓
USB-C   12V Car Port              AC 300W
100W    (for servos)              (tools)
  ↓         ↓
Pi 5    Buck 6V→Servos
```

### BOM for Jackery Build

| Item | Cost |
|------|------|
| Jackery 300 Plus | $200 |
| USB-C PD trigger (optional) | $8 |
| 12V→6V buck converter | $15 |
| DC barrel adapter cable | $8 |
| **Total Power System** | **$231** |

**Benefits over EcoFlow:**
- 64% lower cost ($231 vs $649)
- LiFePO4 chemistry (safer, longer life)
- Still provides stability (3.8kg ballast)
- USB-C 100W direct to Pi 5

---

## Battery Option 6: Lead Acid (Budget Ballast)

**Best for:** Maximum stability on minimum budget, stationary robots

```
┌─────────────────────────────────────┐
│  SLA 12V 18Ah                       │
│  Dimensions: 181 × 76 × 167 mm      │
│  Weight: 5.5 kg                     │
│  Capacity: 216Wh                    │
│  Cost: $40-50                       │
└─────────────────────────────────────┘
```

### Why Lead Acid?

1. **Cheapest $/Wh** - $0.20/Wh vs $0.50-1.00 for Li-ion
2. **Heavy = stable** - 5.5kg provides excellent ballast
3. **Safe chemistry** - No fire risk, handles abuse
4. **Easy charging** - Simple float chargers work
5. **Available everywhere** - Auto parts stores, Walmart

### Disadvantages

- Heavy (though this helps stability)
- Limited cycles (200-300)
- Must stay upright (unless AGM)
- Larger footprint per Wh
- Not suitable for high discharge rates

### Recommended for

- Stationary arm robots (no driving)
- Development/testing platforms
- Budget builds where weight is acceptable
- Situations where ballast is needed anyway

### BOM for SLA Build

| Item | Cost |
|------|------|
| SLA 12V 18Ah battery | $45 |
| Float charger | $20 |
| Buck 12V→5V 6A | $12 |
| Buck 12V→6V 10A | $15 |
| Battery box/mount | $10 |
| Wiring | $10 |
| **Total Power System** | **$112** |

**Savings vs EcoFlow:** $593 (84% less)

---

## Home Robot Chassis Options

### Chassis Comparison

| Chassis | Footprint | Payload | Drive | Cost | Best For |
|---------|-----------|---------|-------|------|----------|
| **Waveshare WAVE ROVER** | 270×240mm | 5kg | 4WD mecanum | $180 | Indoor/outdoor |
| **Yahboom Transbot** | 200×170mm | 2kg | Tracked | $120 | Carpets, learning |
| **Roomba Create 3** | 340mm dia | 9kg | Diff drive | $300 | Vacuuming base |
| **Turtlebot4 Lite** | 340mm dia | 9kg | Diff drive | $1,100 | ROS2 development |
| **Elegoo Tumbller** | 80×75mm | 0.5kg | 2WD balance | $60 | Small demos |
| **Custom Aluminum** | Variable | 10kg+ | Configurable | $150-300 | Heavy duty |
| **3D Printed** | Variable | 2-5kg | Configurable | $50-100 | Full custom |

---

## Chassis Option 1: Waveshare WAVE ROVER

**Best for:** Versatile indoor/outdoor mobile manipulation

```
┌─────────────────────────────────────────────────────┐
│                    WAVE ROVER                        │
│   ┌─────┐                             ┌─────┐       │
│   │Mecan│                             │Mecan│       │
│   │ um  │     ┌───────────────┐       │ um  │       │
│   └─────┘     │   Deck Area   │       └─────┘       │
│               │  (ESP32 slot) │                      │
│   ┌─────┐     │               │       ┌─────┐       │
│   │Mecan│     └───────────────┘       │Mecan│       │
│   │ um  │                             │ um  │       │
│   └─────┘                             └─────┘       │
│                                                      │
│   Dimensions: 270 × 240 × 105 mm                    │
│   Weight: 1.4 kg (base only)                        │
└─────────────────────────────────────────────────────┘
```

### Specifications

| Parameter | Value |
|-----------|-------|
| Dimensions | 270 × 240 × 105 mm |
| Base Weight | 1.4 kg |
| Max Payload | 5 kg |
| Drive | 4× Mecanum wheels (N20 motors) |
| Speed | 0.6 m/s max |
| Control | ESP32-S3 (WiFi/BLE) |
| Power | 7.4V (2S LiPo) or 12V input |
| I/O | UART, I2C, GPIO expansion |

### Why WAVE ROVER?

1. **Omnidirectional** - Mecanum wheels allow sideways movement
2. **Good payload** - 5kg supports tall mast with arms
3. **Expansion ready** - Multiple mounting holes, ESP32 programmable
4. **Reasonable price** - $180 complete with electronics
5. **Indoor/outdoor** - Works on carpet, tile, concrete

### Integration

```
┌─────────────────────────────────────────────────────┐
│                    MODIFIED WAVE ROVER              │
│                                                     │
│   ┌───────────────────────────────────────────┐     │
│   │           Upper Deck (custom)             │     │
│   │   Pi 5 + Arms + Camera                    │     │
│   └───────────────────────────────────────────┘     │
│                      │                              │
│   ┌───────────────────────────────────────────┐     │
│   │          Battery Bay (LiFePO4)            │     │
│   └───────────────────────────────────────────┘     │
│                      │                              │
│   ┌──────────────────┴──────────────────────┐       │
│   │           WAVE ROVER Base               │       │
│   │      ESP32 + Motor Drivers              │       │
│   └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

**Communication:**
- ESP32 ↔ Pi 5 via UART or WiFi
- Pi 5 sends velocity commands
- ESP32 handles low-level motor control

### BOM

| Item | Cost |
|------|------|
| Waveshare WAVE ROVER | $180 |
| Custom upper deck plate | $30 |
| Standoffs/mounting hardware | $15 |
| **Total Chassis** | **$225** |

---

## Chassis Option 2: Yahboom Transbot

**Best for:** Budget tracked robot, carpet navigation

```
┌─────────────────────────────────────────┐
│            TRANSBOT                      │
│                                          │
│   ╔════════════════════════════════╗     │
│   ║                                ║     │
│   ║  ┌──────────────────────────┐  ║     │
│   ║  │    Deck (RasPi slot)     │  ║     │
│   ║  └──────────────────────────┘  ║     │
│   ║                                ║     │
│   ╚════════════════════════════════╝     │
│       │                        │         │
│   [Track]                  [Track]       │
│                                          │
│   Dimensions: 200 × 170 × 120 mm         │
└─────────────────────────────────────────┘
```

### Specifications

| Parameter | Value |
|-----------|-------|
| Dimensions | 200 × 170 × 120 mm |
| Weight | 0.8 kg |
| Max Payload | 2 kg |
| Drive | Tracked (tank style) |
| Speed | 0.4 m/s |
| Control | Expansion board for Pi |
| Power | 2S LiPo included |

### Why Transbot?

1. **Tracks handle carpet well** - No wheel slip
2. **Compact** - Fits in small spaces
3. **Budget friendly** - $120 complete
4. **Pi-ready** - Designed for Raspberry Pi mount
5. **Camera included** - Pan-tilt camera mount

### Limitations

- Lower payload (2kg) limits arm weight
- Not suitable for tall mast (stability)
- Better for single arm or tabletop tasks

---

## Chassis Option 3: iRobot Create 3 (Roomba Base)

**Best for:** Professional development, ROS2 integration

```
┌─────────────────────────────────────────────────────┐
│                   CREATE 3                           │
│                                                     │
│              ┌─────────────────┐                    │
│        ╭─────│   Cargo Bay     │─────╮              │
│       ╱      │   (9kg payload) │      ╲             │
│      │       └─────────────────┘       │            │
│      │                                 │            │
│      │    ○                     ○      │            │
│      │  (wheel)             (wheel)   │            │
│       ╲                               ╱             │
│        ╰─────────────────────────────╯              │
│                     ○                               │
│                  (caster)                           │
│                                                     │
│   Diameter: 340 mm                                  │
│   Payload: 9 kg                                     │
└─────────────────────────────────────────────────────┘
```

### Specifications

| Parameter | Value |
|-----------|-------|
| Diameter | 340 mm |
| Height | 92 mm |
| Weight | 3.2 kg |
| Max Payload | 9 kg |
| Drive | Differential (2 wheels + caster) |
| Speed | 0.3 m/s |
| Runtime | 2-3 hours (built-in battery) |
| Charging | Auto-dock included |
| Interface | ROS2 native, USB-C |

### Why Create 3?

1. **High payload** - 9kg supports heavy mast + EcoFlow alternative
2. **Auto-docking** - Built-in charging system
3. **ROS2 native** - Professional robotics stack
4. **Proven reliability** - Roomba heritage
5. **Cliff/bump sensors** - Built-in safety

### Integration

```
┌─────────────────────────────────────────────────────┐
│                 CREATE 3 + MAST                     │
│                                                     │
│                    [Camera]                         │
│                       │                             │
│               ════════╪════════  Top brace          │
│                   │   │   │                         │
│                  [Arms + Carriage]                  │
│                   │   │   │                         │
│                   │   │   │    V-slot mast          │
│                   │   │   │                         │
│               ════╪═══╪═══╪════  Deck               │
│                   │   │   │                         │
│              ┌────┴───┴───┴────┐                    │
│              │    Battery Bay  │  LiFePO4 or        │
│              │                 │  E-bike battery    │
│              └─────────────────┘                    │
│         ╭─────────────────────────────╮             │
│        │         CREATE 3 BASE         │            │
│         ╰─────────────────────────────╯             │
└─────────────────────────────────────────────────────┘
```

### Cost

| Item | Cost |
|------|------|
| iRobot Create 3 | $300 |
| Mounting adapter plate | $40 |
| **Total Chassis** | **$340** |

---

## Chassis Option 4: Custom Aluminum Frame

**Best for:** Heavy duty, maximum customization

```
┌─────────────────────────────────────────────────────┐
│            CUSTOM ALUMINUM CHASSIS                   │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │  Top Plate (6mm Al)                         │   │
│   │  400 × 300 mm                               │   │
│   │    ○     ○     ○     ○    Mounting holes   │   │
│   └─────────────────────────────────────────────┘   │
│         │         │         │         │             │
│     [Standoff] [Standoff] [Standoff] [Standoff]     │
│         │         │         │         │             │
│   ┌─────┴─────────┴─────────┴─────────┴─────┐       │
│   │  Bottom Plate (6mm Al)                  │       │
│   │    ○         ○         ○         ○      │       │
│   └──┬──────────┬──────────┬──────────┬─────┘       │
│      │          │          │          │             │
│   [Motor]   [Caster]    [Caster]   [Motor]         │
│   +Wheel                            +Wheel          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Components

| Component | Specification | Source | Cost |
|-----------|---------------|--------|------|
| Top plate | 400×300×6mm Al | SendCutSend | $35 |
| Bottom plate | 400×300×6mm Al | SendCutSend | $35 |
| Standoffs | 50mm M6 Al (4×) | McMaster | $15 |
| Drive motors | 12V DC w/ encoder (2×) | Pololu/Amazon | $50 |
| Wheels | 100mm rubber (2×) | Amazon | $20 |
| Casters | 50mm swivel (2×) | Amazon | $15 |
| Motor driver | L298N or TB6612 | Amazon | $10 |
| Hardware | Screws, nuts, etc. | Various | $20 |
| **Total** | | | **$200** |

### Advantages

- **Full customization** - Exact dimensions you need
- **Heavy duty** - Supports 15kg+ payload
- **Modular** - Easy to add/remove components
- **Repairable** - Standard parts, no proprietary pieces

### Design Files

Available at `/hardware/chassis/custom-aluminum/` (future):
- DXF files for laser/waterjet cutting
- Fusion 360 assembly
- STEP export

---

## Chassis Option 5: 3D Printed Platform

**Best for:** Rapid prototyping, one-off builds

```
┌─────────────────────────────────────────────────────┐
│          3D PRINTED MODULAR CHASSIS                  │
│                                                     │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │
│   │Mod 1│ │Mod 2│ │Mod 3│ │Mod 4│  Interlocking     │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  modules          │
│      └──────┴──────┴──────┘                         │
│                  │                                   │
│              [Base Rail]                             │
│      ○──────────┴──────────○                        │
│   [Wheel]              [Wheel]                      │
│                                                     │
│   Material: PETG or ABS                             │
│   Infill: 40-60%                                    │
│   Print time: 20-40 hours                           │
└─────────────────────────────────────────────────────┘
```

### Open Source Designs

| Design | Size | Payload | Link |
|--------|------|---------|------|
| OpenBot | 200×150mm | 1kg | github.com/isl-org/OpenBot |
| SCUTTLE | 300×250mm | 5kg | scuttlerobot.org |
| Linorobot2 | 300×200mm | 3kg | github.com/linorobot |
| QPR | 250×200mm | 2kg | Printables |

### Material Recommendations

| Material | Strength | Heat Resist | Cost | Use Case |
|----------|----------|-------------|------|----------|
| PLA | Low | 60°C | $20/kg | Prototypes only |
| PETG | Medium | 80°C | $25/kg | Indoor robots |
| ABS | Medium | 100°C | $25/kg | Outdoor/enclosed |
| ASA | Medium | 100°C | $30/kg | Outdoor UV stable |
| Nylon | High | 120°C | $50/kg | Load-bearing parts |

### Estimated Print Costs

| Component | Weight | Time | Filament Cost |
|-----------|--------|------|---------------|
| Base plate | 300g | 12h | $8 |
| Motor mounts (2×) | 100g | 4h | $3 |
| Battery holder | 150g | 6h | $4 |
| Deck mounts | 200g | 8h | $5 |
| **Total** | **750g** | **30h** | **$20** |

Add motors, wheels, electronics: ~$80-120

---

## Complete Build Configurations

### Build A: Budget Home Assistant (Under $400)

**Target:** Simple home robot for light tasks

```
Configuration:
├── Chassis: Yahboom Transbot ($120)
├── Battery: Built-in 2S LiPo + power bank ($50)
├── Compute: Raspberry Pi 5 ($80)
├── Arms: Single SO-ARM101 ($100)
└── Camera: USB webcam ($30)

Total: ~$380
```

**Capabilities:**
- Fetch small items
- Video presence/telepresence
- Voice assistant platform
- Educational platform

---

### Build B: Mid-Range Manipulator (Under $700)

**Target:** Dual-arm mobile manipulation

```
Configuration:
├── Chassis: Waveshare WAVE ROVER ($180)
├── Battery: LiFePO4 12V 20Ah ($120)
├── Compute: Raspberry Pi 5 + AI HAT+ ($130)
├── Arms: Dual SO-ARM101 ($200)
├── Camera: OAK-D Lite ($150)
├── Power distribution ($30)
└── Mounting hardware ($30)

Total: ~$840 (but close to $700 with alternatives)
```

**Optimized version:**
```
├── Chassis: Custom aluminum (DIY) ($150)
├── Battery: LiFePO4 12V 20Ah ($100)
├── Compute: Pi 5 + USB Coral ($110)
├── Arms: Dual SO-ARM101 ($200)
├── Camera: OAK-D Lite ($150)

Total: ~$710
```

---

### Build C: Tall Reach Robot (Under $1000)

**Target:** Human-height reach without EcoFlow

```
Configuration:
├── Chassis: Create 3 or custom ($300)
├── Battery: E-bike 48V 15Ah ($180)
├── Mast: V-slot 20×40 1500mm ($99)
├── Compute: Pi 5 + AI HAT+ ($130)
├── Arms: Dual SO-ARM101 ($200)
├── Camera: OAK-D Lite ($150)
├── Lift mechanism ($36)
├── Power conversion ($50)
└── Counterweight (steel plate) ($30)

Total: ~$1,175
```

**Note:** Without EcoFlow's 8kg ballast, add 4-6kg steel counterweight at base for stability.

---

### Build D: Professional Research Platform (Under $1500)

**Target:** Full capability, matches EcoFlow build performance

```
Configuration:
├── Chassis: iRobot Create 3 ($300)
├── Battery: Jackery 300 Plus ($200)
├── Mast: V-slot 20×40 full ($99)
├── Compute: Pi 5 8GB + AI HAT+ ($180)
├── Arms: Dual SO-ARM101 ($200)
├── Camera: OAK-D Lite ($150)
├── Lift mechanism ($36)
├── Power distribution ($40)
├── Sensors (IMU, force/torque) ($100)
├── Enclosure/shell ($100)
└── Mounting/hardware ($50)

Total: ~$1,455
```

**Comparison to EcoFlow build ($1,453):**
- Same total cost
- Less battery capacity (288Wh vs 720Wh)
- Safer chemistry (LiFePO4)
- Auto-docking from Create 3
- Slightly less stable (need counterweight)

---

## Stability Comparison Without EcoFlow

Since EcoFlow provides 8kg of low ballast, alternatives need compensation:

| Configuration | Battery Weight | Additional Ballast | Total Low Mass | Tip Angle |
|---------------|----------------|-------------------|----------------|-----------|
| EcoFlow build | 8.0kg | 0kg | 8.0kg | 21.4° |
| LiFePO4 + 5kg plate | 2.8kg | 5kg steel | 7.8kg | 20.8° |
| E-bike + 3kg plate | 4.5kg | 3kg steel | 7.5kg | 20.2° |
| Jackery + 4kg plate | 3.8kg | 4kg steel | 7.8kg | 20.8° |
| Lead acid (SLA) | 5.5kg | 2kg plate | 7.5kg | 20.2° |

**Steel ballast plate:**
- 300×200×10mm steel plate = 4.7kg
- Cost: ~$20-30 from metal supplier
- Mount on bottom deck alongside battery

---

## Decision Matrix

| Priority | Recommended Battery | Recommended Chassis |
|----------|--------------------|--------------------|
| **Lowest cost** | Lead Acid SLA | 3D Printed |
| **Best value** | LiFePO4 12V 20Ah | Waveshare WAVE ROVER |
| **Longest runtime** | E-bike 48V 15Ah | Custom Aluminum |
| **Easiest swap** | Milwaukee M18 | Create 3 |
| **Safest** | LiFePO4 or Jackery | Create 3 (auto-dock) |
| **Lightest** | Custom 18650 pack | 3D Printed |
| **Professional** | Jackery 300 Plus | Create 3 |

---

## Summary

Removing the EcoFlow River Max Plus from the build opens up significant cost savings while requiring attention to stability through alternative ballast. The best alternatives are:

1. **LiFePO4 12V 20Ah** - Best overall value ($120, safe, long-life)
2. **E-bike 48V 15Ah** - Best capacity/dollar ($180, 720Wh)
3. **Jackery 300 Plus** - Best convenience ($200, all-in-one)

For chassis, the **Waveshare WAVE ROVER** ($180) and **iRobot Create 3** ($300) offer the best balance of capability, payload, and ease of integration.

**Total savings vs EcoFlow build:**
- Budget build: Save $600+ (60%)
- Mid-range build: Save $400+ (40%)
- Professional build: Similar cost, different tradeoffs

---

## References

- Waveshare WAVE ROVER: https://www.waveshare.com/wave-rover.htm
- iRobot Create 3: https://edu.irobot.com/create3
- LiFePO4 batteries: https://www.amperetime.com
- E-bike batteries: alibaba.com, amazon.com
- Jackery: https://www.jackery.com
- OpenBuilds V-Slot: https://openbuilds.com
