package com.continuonxr.app.glove

/**
 * Lightweight tracker for MTU negotiation, sequence gaps, and sample rate estimation.
 * Keeps us aligned with RLDS diagnostics fields without coupling to Android APIs.
 */
class GloveDiagnosticsTracker(
    private val minMtu: Int,
    private val targetSampleRateHz: Int,
) {
    private var lastSequence: Int? = null
    private var lastArrivalNanos: Long? = null
    private var negotiatedMtu: Int = minMtu
    private var dropCount: Int = 0
    private var sampleRateHz: Float = 0f

    fun onMtuNegotiated(mtu: Int): GloveDiagnosticsTracker {
        negotiatedMtu = mtu
        if (mtu < minMtu) dropCount++
        return this
    }

    fun onFrame(frame: GloveFrame, arrivalTimestampNanos: Long, rssi: Int? = null): GloveDiagnostics {
        frame.sequence?.let { seq ->
            lastSequence?.let { prev ->
                val delta = ((seq - prev + 65536) % 65536)
                if (delta > 1) {
                    dropCount += delta - 1
                }
            }
            lastSequence = seq
        }

        lastArrivalNanos?.let { previous ->
            val deltaNanos = arrivalTimestampNanos - previous
            if (deltaNanos > 0) {
                sampleRateHz = 1_000_000_000f / deltaNanos.toFloat()
            }
        } ?: run {
            sampleRateHz = targetSampleRateHz.toFloat()
        }
        lastArrivalNanos = arrivalTimestampNanos

        return snapshot(rssi)
    }

    fun snapshot(rssi: Int? = null): GloveDiagnostics = GloveDiagnostics(
        mtu = negotiatedMtu,
        sampleRateHz = sampleRateHz,
        dropCount = dropCount,
        rssi = rssi,
    )
}
