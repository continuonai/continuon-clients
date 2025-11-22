package com.continuonxr.app.glove

/**
 * Parses raw BLE payloads into `GloveFrame`.
 * The format is documented in docs/glove-ble.md and will be finalized with firmware.
 */
object GloveFrameParser {
    fun parse(raw: ByteArray, timestampNanos: Long): GloveFrame? {
        // Expect at least 45 bytes based on draft format; reject if undersized.
        if (raw.size < 45) return null

        // TODO: Parse real values once firmware format is finalized.
        val flex = FloatArray(5) { 0f }
        val fsr = FloatArray(8) { 0f }
        val orientationQuat = floatArrayOf(0f, 0f, 0f, 1f)
        val accel = floatArrayOf(0f, 0f, 0f)

        return GloveFrame(
            timestampNanos = timestampNanos,
            flex = flex,
            fsr = fsr,
            orientationQuat = orientationQuat,
            accel = accel,
        )
    }
}

