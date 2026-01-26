# ContinuonXR Robot Build Progress

**Started:** 2026-01-26
**Target:** V-Slot Mast Robot with dual SO-ARM101 arms

---

## ✅ Acquired Components

### Phase 1: Chassis
- [x] **Professional 97mm Mecanum Chassis** - $99
  - 4× 97mm Mecanum wheels
  - 4× 12V 360RPM encoder motors
  - Aluminum extrusion platform (330×290mm)
  - Amazon: B09VZV35RF

### Phase 2: Power
- [x] **EcoFlow River Max Plus** - $649
  - 720Wh capacity
  - USB-C 100W (for Pi 5)
  - Car port 12.6V/8A (for motors/servos)
  - 8.0kg (provides stability ballast)

---

## 🔄 In Progress / Next to Buy

### Phase 3: Mast Structure (~$141)
- [ ] V-Slot 2040 Rails 1500mm (×2) - FAHKNS 5-pack
  - Amazon: B0CF9TXKJ3 (~$70)
- [ ] V-Slot 2020 Rail 500mm (for cross-brace)
  - Search: "2020 v-slot 500mm" (~$12)
- [ ] Gantry Plate Kit (×2)
  - Amazon: B08L39S8QB (~$24)
- [ ] 90° Corner Brackets (10pk)
  - Search: "2040 corner bracket" (~$15)
- [ ] M5 T-Nuts (50pk)
  - Search: "M5 t-nut v-slot" (~$8)
- [ ] M5 Screws Assortment
  - Amazon: B08P4PK77V (~$12)

### Phase 4: Power Distribution (~$56)
- [ ] Buck Converter 12V→6V 10A (servo power)
  - Amazon: B07JZ2B9TK (~$15)
- [ ] DC Barrel Cable for EcoFlow car port
  - Amazon: B07C61434H (~$8)
- [ ] Power Distribution Board
  - Amazon: B08P1JZSLV (~$10)
- [ ] Fuse Holder + 15A Fuse
  - Amazon: B07PNLQ2Y9 (~$6)
- [ ] XT60 Connectors (5 pairs)
  - Amazon: B07VRZR5TL (~$9)
- [ ] USB-C PD Trigger Board (optional)
  - Amazon: B0B6FYJM1N (~$8)

### Phase 5: Lift Mechanism (~$50)
- [ ] NEMA17 High-Torque Stepper (2.5A)
  - Amazon: B00PNEQI7W (~$18) ★ Recommended
- [ ] GT2 Timing Belt 6mm × 5m
  - Amazon: B07GFYLJP8 (~$8)
- [ ] GT2 20T Pulleys (5pk)
  - Amazon: B07VPB54VN (~$8)
- [ ] GT2 Idler Pulleys (2pk)
  - Amazon: B07GCX7T5B (~$6)
- [ ] TMC2209 Stepper Driver
  - Amazon: B0D2J73TM8 (~$10)
- [ ] Limit Switches (5pk)
  - Amazon: B07QQ2RBL5 (~$7)

### Phase 6: Deck Plates (~$88)
- [ ] Deck plates 300×220×3mm ×3 (SendCutSend)
  - Material: 5052 Aluminum, 3mm (~$45)
- [ ] Carriage plate 200×140×5mm (SendCutSend)
  - Material: 5052 Aluminum, 5mm (~$12)
- [ ] M5 Standoffs assorted
  - Amazon (~$10)
- [ ] M5 screws + nuts
  - Amazon (~$6)
- [ ] Cargo bins/containers
  - Amazon: B07DFDS56F (~$18)

### Phase 7: Computing (~$245)
- [ ] Raspberry Pi 5 (8GB)
  - PiShop/Adafruit (~$80)
- [ ] PCA9685 Servo Controller (×2)
  - Amazon/Adafruit (~$12)
- [ ] OAK-D Lite Camera
  - Luxonis (~$150)
- [ ] SG90 Pan Servo (camera)
  - Amazon: B07Q6JGWNV (~$3)

### Phase 8: Arms (~$200)
- [ ] SO-ARM101 6-DOF Arm (×2)
  - Various suppliers (~$100 each)

---

## 💰 Budget Summary

| Phase | Description | Est. Cost | Status |
|-------|-------------|-----------|--------|
| 1 | Chassis | $99 | ✅ Done |
| 2 | Power (EcoFlow) | $649 | ✅ Done |
| 3 | Mast Structure | $141 | 🔄 Next |
| 4 | Power Distribution | $56 | Pending |
| 5 | Lift Mechanism | $50 | Pending |
| 6 | Deck Plates | $88 | Pending |
| 7 | Computing | $245 | Pending |
| 8 | Arms | $200 | Pending |
| **TOTAL** | | **$1,528** | |
| **Remaining** | | **$780** | |

---

## 📋 Build Order (Recommended)

1. **Mast Structure** - Can mount to chassis immediately
2. **Deck Plates** - Order from SendCutSend (1-2 week lead time)
3. **Power Distribution** - Wire up EcoFlow to robot
4. **Lift Mechanism** - Install on mast
5. **Computing** - Pi 5 + controllers
6. **Arms** - Final integration

---

## 🔗 Key Documentation

- [V-Slot Mast Design](./v-slot-mast-design.md) - Full specifications
- [Alternative Robot Builds](./alternative-robot-builds.md) - Budget options
- [Power & Stacking Guide](../power-and-stacking.md) - Wiring details

---

## 📝 Notes

- OpenBuilds is out of business (Jan 2026) - use FAHKNS or Iverntech for V-slot
- EcoFlow provides 8kg ballast for excellent stability (21° tip angle)
- Arm carriage is light (precision) / Deck 3 is heavy (cargo) - separate concerns
