package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.glove.GloveDiagnostics

/**
 * Convenience wrapper to start RLDS episodes with repo defaults and enrich steps with
 * timestamps/diagnostics.
 */
class RldsRecorder(
    private val writer: RldsEpisodeWriter,
    private val defaults: EpisodeDefaults?,
) {
    fun start(metadata: EpisodeMetadata) {
        writer.startEpisodeWithDefaults(metadata, defaults)
    }

    fun recordStep(
        observation: Observation,
        action: Action,
        isTerminal: Boolean = false,
        gloveDiagnostics: GloveDiagnostics? = null,
        videoTimestampNanos: Long? = null,
        depthTimestampNanos: Long? = null,
        stepMetadata: Map<String, String> = emptyMap(),
    ) {
        val enrichedObservation = observation
            .withVideoDepthTimestamps(videoTimestampNanos, depthTimestampNanos)
            .withGloveDiagnostics(gloveDiagnostics)
        writer.recordStep(
            observation = enrichedObservation,
            action = action,
            isTerminal = isTerminal,
            stepMetadata = stepMetadata,
        )
    }

    fun completeEpisode() = writer.completeEpisode()
    fun recordedCount(): Int = writer.recordedCount()

    companion object {
        fun fromConfig(config: LoggingConfig): RldsRecorder {
            val writer = RldsEpisodeWriter(config)
            val defaults = EpisodeDefaults.fromLoggingConfig(config)
            return RldsRecorder(writer, defaults)
        }
    }
}

