# ContinuonXR Plate Designs

Laser-cut plate designs for SendCutSend or similar fabrication services.

## Files

| File | Size | Thickness | Material | Purpose |
|------|------|-----------|----------|---------|
| `deck-plate-300x220.svg` | 300×220mm | 3mm | 5052 Aluminum | Decks 1, 2, 3 (order 3×) |
| `carriage-plate-200x140.svg` | 200×140mm | 5mm | 5052 Aluminum | Arm carriage (order 1×) |

## Hole Legend

- **Black outline**: Cut boundary
- **Red circles**: M5 mounting holes (5.5mm clearance)
- **Green circles**: Arm mounting holes (5.5mm)
- **Blue shapes**: Pass-through slots/holes for wiring or V-slot

## Ordering from SendCutSend

1. Go to [sendcutsend.com](https://sendcutsend.com)
2. Upload SVG file
3. Select material: **5052 Aluminum**
4. Select thickness: **3mm** (decks) or **5mm** (carriage)
5. Finish: Mill finish or clear anodized

## Deck Plate Layout (300×220mm)

```
    ┌─────────────────────────────────────────────┐
    │ o                                         o │  ← Corner mount holes
    │ o    o          o          o              o │
    │                                             │
    │      o          o          o                │  ← Accessory holes
    │ ┌──┐                                  ┌──┐  │
    │ │  │ (V-slot pass-through)            │  │  │  ← Upright slots
    │ └──┘                                  └──┘  │
    │      o          o          o                │
    │                                             │
    │ o                                         o │
    │ o    o          o          o              o │
    └─────────────────────────────────────────────┘
```

## Carriage Plate Layout (200×140mm)

```
    ┌───────────────────────────────────────┐
    │ o              o              o       │  ← Gantry mount
    │ o                              o      │
    │    o    o          o    o            │  ← Left arm    Right arm
    │    │  ( )  │      │  ( )  │          │  ← Wire holes
    │    o    o          o    o            │
    │ o                              o      │
    │ o              o              o       │
    └───────────────────────────────────────┘
```

## V-Slot Upright Spacing

- Uprights spaced **240mm** center-to-center
- Pass-through slots sized for 20×40mm V-slot with clearance
- Decks mount using 90° corner brackets to uprights

## Notes

- All dimensions in millimeters
- M5 holes are 5.5mm for clearance fit
- Round corners on slots (2mm radius) for stress relief
- If you need DXF format, use an online SVG→DXF converter or open in Inkscape and export
