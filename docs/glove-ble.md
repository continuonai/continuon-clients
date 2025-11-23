# Continuon Glove BLE Notes

## Connectivity
- Device: Continuon Glove v0 (e.g., XIAO nRF52840 Sense).
- Transport: BLE, request MTU >= 64 bytes for frame payloads.
- Service UUID: `7d0e1000-5c86-4c84-9c72-6fa4cbb8a9c5`.
- Characteristics:
  - Data notify: `7d0e1001-5c86-4c84-9c72-6fa4cbb8a9c5` (streamed frames, notifications enabled).
  - Control/config write: `7d0e1002-5c86-4c84-9c72-6fa4cbb8a9c5` (optional future use for calibration or rate changes).

## Frame format (v1, 56 bytes)
All multi-byte fields are little-endian.

```
byte 0      : version (0x01)
byte 1      : status flags (bit0 on-body, bit1 IMU calibrated, bit2 battery low, others reserved)
bytes 2-3   : sequence uint16 (wraps at 65535)
bytes 4-7   : sample timestamp micros since boot uint32
bytes 8-17  : flex[5] uint16 (0..1023)
bytes 18-33 : fsr[8] uint16 (0..1023)
bytes 34-41 : orientation_quat[4] int16 (scaled by 1e4)
bytes 42-47 : accel[3] int16 (mg)
bytes 48-49 : temperature centi-deg C uint16 (optional; 2500 = 25.00 C)
bytes 50-51 : battery millivolts uint16
bytes 52-55 : crc32/le
```

## Ingestion requirements
- Negotiate MTU and subscribe to notifications; reject or warn if MTU < 64.
- Decode frames as above, normalizing:
  - `glove.flex` and `glove.fsr`: divide by 1023 to obtain [0,1].
  - `glove.orientation_quat`: divide by 1e4 and renormalize to a unit quaternion.
  - `glove.accel`: convert mg to m/s^2 using 9.80665 m/s^2 per g.
- Timestamp frames upon receipt; measure inter-arrival jitter and track dropped sequence numbers in diagnostics.
- Surface connection health (MTU, RSSI, drop counts, computed sample rate) into RLDS diagnostics alongside glove fields.
