package com.continuonxr.app.logging

import com.continuonxr.app.glove.GloveDiagnostics

/**
 * Helper to merge glove diagnostics into RLDS diagnostics blocks.
 */
fun Diagnostics.withGloveDiagnostics(glove: GloveDiagnostics): Diagnostics =
    copy(
        gloveDrops = glove.dropCount,
        gloveSampleRateHz = glove.sampleRateHz,
        bleRssi = glove.rssi ?: bleRssi,
    )

