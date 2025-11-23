package com.continuonxr.app.glove

import org.junit.Assert.assertEquals
import org.junit.Test

class GloveDiagnosticsTrackerTest {
    @Test
    fun tracksMtuAndDrops() {
        val tracker = GloveDiagnosticsTracker(minMtu = 64, targetSampleRateHz = 100)

        tracker.onMtuNegotiated(60)
        val snapshot = tracker.snapshot()

        assertEquals(60, snapshot.mtu)
        assertEquals(1, snapshot.dropCount)
    }

    @Test
    fun estimatesSampleRateAndSequenceGaps() {
        val tracker = GloveDiagnosticsTracker(minMtu = 64, targetSampleRateHz = 100)
        val frame1 = GloveFrame(
            timestampNanos = 0,
            flex = emptyList(),
            fsr = emptyList(),
            orientationQuat = emptyList(),
            accel = emptyList(),
            sequence = 1,
        )
        val frame2 = frame1.copy(sequence = 3)

        tracker.onFrame(frame1, arrivalTimestampNanos = 0)
        val diag = tracker.onFrame(frame2, arrivalTimestampNanos = 10_000_000)

        assertEquals(1, diag.dropCount) // missed sequence 2
        assertEquals(100f, diag.sampleRateHz, 1e-2f) // 10 ms spacing -> 100 Hz
    }
}
