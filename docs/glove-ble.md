# Continuon Glove BLE Notes (Draft)

## Connectivity
- Device: Continuon Glove v0 (e.g., XIAO nRF52840 Sense).
- Transport: BLE, request MTU >= 64 bytes for frame payloads.
- Service/characteristic UUIDs: to be assigned; target single notify characteristic for data frames plus one control characteristic for config.

## Frame format (example placeholder, 45 bytes)
```
byte 0      : flags/version
bytes 1-10  : flex[5] uint16 (0..1023)
bytes 11-26 : fsr[8] uint16 (0..1023)  (fsr7 shares last byte for now)
bytes 27-34 : orientation_quat[4] int16 (scaled by 1e4)
bytes 35-40 : accel[3] int16 (mg)
bytes 41-44 : crc32/le
```

Actual encoding will be finalized with firmware; parser should allow pluggable mapping and scaling.

## Ingestion requirements
- Negotiate MTU and subscribe to notifications; reject connection if MTU < 64.
- Convert raw integers to normalized floats in range [0,1] for flex/fsr; convert orientation to unit quaternion; accel to m/s^2.
- Timestamp frames upon receipt; measure inter-arrival jitter and record in diagnostics.
- Surface connection health (RSSI, drop counts, reconnect attempts) into RLDS diagnostics.

