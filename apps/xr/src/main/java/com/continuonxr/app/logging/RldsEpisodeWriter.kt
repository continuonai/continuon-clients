package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.glove.GloveFrame
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.BufferedWriter
import java.io.File

/**
 * RLDS writer with a pluggable sink. Default sink writes metadata.json and steps.jsonl.
 */
class RldsEpisodeWriter(
    private val config: LoggingConfig,
    private val sink: EpisodeSink = FileEpisodeSink(config),
) {
    private val steps = mutableListOf<Step>()
    private var started = false

    fun startEpisode(metadata: EpisodeMetadata) {
        steps.clear()
        sink.onStart(metadata)
        started = true
    }

    fun recordStep(
        observation: Observation,
        action: Action,
        isTerminal: Boolean = false,
    ) {
        check(started) { "Episode not started" }
        val step = Step(observation, action, isTerminal)
        steps.add(step)
        sink.onStep(step)
    }

    fun completeEpisode() {
        if (!started) return
        sink.onComplete()
        started = false
    }

    fun recordedCount(): Int = steps.size
}

interface EpisodeSink {
    fun onStart(metadata: EpisodeMetadata)
    fun onStep(step: Step)
    fun onComplete()
}

private class FileEpisodeSink(private val config: LoggingConfig) : EpisodeSink {
    private val json = Json { encodeDefaults = true }
    private var stepsWriter: BufferedWriter? = null

    override fun onStart(metadata: EpisodeMetadata) {
        val dir = File(config.episodeOutputDir)
        dir.mkdirs()
        File(dir, "metadata.json").writeText(json.encodeToString(metadata))
        stepsWriter = File(dir, "steps.jsonl").bufferedWriter()
    }

    override fun onStep(step: Step) {
        stepsWriter?.apply {
            write(json.encodeToString(step))
            newLine()
        }
    }

    override fun onComplete() {
        stepsWriter?.flush()
        stepsWriter?.close()
        stepsWriter = null
    }
}

@Serializable
data class EpisodeMetadata(
    val xrMode: String,
    val controlRole: String,
    val environmentId: String,
    val tags: List<String> = emptyList(),
)

@Serializable
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

@Serializable
data class Action(
    val command: FloatArray,
    val source: String,
    val annotation: Annotation? = null,
    val uiAction: UiAction? = null,
)

@Serializable
data class Pose(
    val position: FloatArray = floatArrayOf(0f, 0f, 0f),
    val orientationQuat: FloatArray = floatArrayOf(0f, 0f, 0f, 1f),
    val valid: Boolean = true,
)

@Serializable
data class Diagnostics(
    val latencyMs: Float = 0f,
    val gloveDrops: Int = 0,
    val bleRssi: Int? = null,
)

@Serializable
data class Annotation(
    val kind: String,
    val payload: Map<String, String>,
)

@Serializable
data class UiAction(
    val actionType: String,
    val context: Map<String, String>,
)

@Serializable
data class Step(
    val observation: Observation,
    val action: Action,
    val isTerminal: Boolean = false,
)
