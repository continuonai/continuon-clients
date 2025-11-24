package com.continuonxr.app.input

/**
 * Extremely lightweight input fusion placeholder that normalizes a handful of
 * synthetic sensor samples. The goal is to give Mode shells something to wire
 * into while keeping all computations cheap enough for stubbed builds.
 */
data class SensorSample(
    val name: String,
    val value: Float,
    val timestampNanos: Long,
)

data class FusedInput(
    val combinedValue: Float,
    val confidence: Float,
    val description: String,
)

class InputFusionEngine {
    fun fuse(samples: List<SensorSample>): FusedInput {
        if (samples.isEmpty()) {
            return FusedInput(
                combinedValue = 0f,
                confidence = 0f,
                description = "No samples provided",
            )
        }

        val combined = samples.map { it.value }.average().toFloat()
        val confidence = (samples.size / 5f).coerceAtMost(1f)
        val summaryNames = samples.joinToString { it.name }

        return FusedInput(
            combinedValue = combined,
            confidence = confidence,
            description = "Fused ${samples.size} inputs: $summaryNames",
        )
    }
}
