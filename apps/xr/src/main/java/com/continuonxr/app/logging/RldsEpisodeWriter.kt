package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.glove.GloveFrame
import com.continuonxr.app.connectivity.RobotState

/**
 * Minimal RLDS writer stub. Enforces required fields from docs/rlds-schema.md.
 * Replace with a streaming writer that persists steps to disk and batches uploads.
 */
class RldsEpisodeWriter(private val config: LoggingConfig) {
    private val steps = mutableListOf<Step>()

    fun startEpisode(metadata: EpisodeMetadata) {
        steps.clear()
        // TODO: Persist metadata.json and open step writer.
    }

    fun recordStep(
        observation: Observation,
        action: Action,
        isTerminal: Boolean = false,
    ) {
        steps.add(Step(observation, action, isTerminal))
        // TODO: Stream to storage and enforce schema validation.
    }

    fun completeEpisode() {
        // TODO: Flush and optionally upload.
    }

    fun recordedCount(): Int = steps.size
}

data class EpisodeMetadata(
    val xrMode: String,
    val controlRole: String,
    val environmentId: String,
    val tags: List<String> = emptyList(),
)

data class Observation(
    val headsetPose: Pose,
    val rightHandPose: Pose,
    val leftHandPose: Pose?,
    val gloveFrame: GloveFrame?,
    val robotState: RobotState?,
    val videoFrameId: String? = null,
    val depthFrameId: String? = null,
    val diagnostics: Diagnostics = Diagnostics(),
)

data class Action(
    val command: FloatArray,
    val source: String,
    val annotation: Annotation? = null,
    val uiAction: UiAction? = null,
)

data class Pose(
    val position: FloatArray = floatArrayOf(0f, 0f, 0f),
    val orientationQuat: FloatArray = floatArrayOf(0f, 0f, 0f, 1f),
    val valid: Boolean = true,
)

data class Diagnostics(
    val latencyMs: Float = 0f,
    val gloveDrops: Int = 0,
    val bleRssi: Int? = null,
)

data class Annotation(
    val kind: String,
    val payload: Map<String, Any>,
)

data class UiAction(
    val actionType: String,
    val context: Map<String, String>,
)

data class Step(
    val observation: Observation,
    val action: Action,
    val isTerminal: Boolean = false,
)
