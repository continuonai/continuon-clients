package com.continuonxr.app.glove

import com.continuonxr.app.config.GloveConfig

/**
 * BLE ingest stub for Continuon Glove v0.
 * Handles MTU negotiation, frame parsing, and diagnostics.
 */
class GloveBleClient(private val config: GloveConfig) {
    fun connect(onFrame: (GloveFrame) -> Unit, onDiagnostics: (GloveDiagnostics) -> Unit) {
        // TODO: Request MTU, subscribe to notifications, parse payloads via GloveFrameParser, emit frames.
    }

    fun disconnect() {
        // TODO: Close BLE connection and clean up resources.
    }
}

data class GloveFrame(
    val timestampNanos: Long,
    val flex: FloatArray,          // size 5, normalized 0..1
    val fsr: FloatArray,           // size 8, normalized 0..1
    val orientationQuat: FloatArray, // size 4
    val accel: FloatArray,         // size 3, m/s^2
)

data class GloveDiagnostics(
    val mtu: Int,
    val sampleRateHz: Float,
    val dropCount: Int,
    val rssi: Int?,
)
