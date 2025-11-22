package com.continuonxr.app.glove

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Parses raw BLE payloads into `GloveFrame`.
 * The format is documented in docs/glove-ble.md and will be finalized with firmware.
 */
object GloveFrameParser {
    private const val EXPECTED_MIN_BYTES = 45
    private const val MAX_SENSOR_VALUE = 1023f
    private const val ORIENTATION_SCALE = 1e4f
    private const val GRAVITY = 9.80665f

    fun parse(raw: ByteArray, timestampNanos: Long): GloveFrame? {
        if (raw.size < EXPECTED_MIN_BYTES) return null
        val buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)

        buffer.get() // flags/version, currently unused

        val flex = FloatArray(5) { normalizeUInt16(buffer.short) }
        val fsr = FloatArray(8) { normalizeUInt16(buffer.short) }

        val orientationQuat = FloatArray(4) {
            buffer.short.toFloat() / ORIENTATION_SCALE
        }.normalizeQuat()

        val accel = FloatArray(3) {
            val mg = buffer.short.toFloat() / 1000f
            mg * GRAVITY
        }

        // Remaining bytes (CRC) are ignored for now.

        return GloveFrame(
            timestampNanos = timestampNanos,
            flex = flex,
            fsr = fsr,
            orientationQuat = orientationQuat,
            accel = accel,
        )
    }

    private fun normalizeUInt16(value: Short): Float {
        val unsigned = value.toInt() and 0xFFFF
        return (unsigned / MAX_SENSOR_VALUE).coerceIn(0f, 1f)
    }

    private fun FloatArray.normalizeQuat(): FloatArray {
        val norm = sqrt(this.fold(0.0) { acc, f -> acc + (f * f) })
        return if (norm > 0.0) this.map { (it / norm).toFloat() }.toFloatArray() else this
    }
}
