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
    private val validator: RldsValidator = RldsValidator(),
    private val uploader: RldsUploader = defaultUploader(config),
) {
    private val steps = mutableListOf<Step>()
    private var started = false

    fun startEpisode(metadata: EpisodeMetadata) {
        steps.clear()
        if (config.validateRlds) {
            val issues = validator.validateEpisodeMetadata(metadata)
            handleIssues(issues)
        }
        sink.onStart(metadata)
        started = true
    }

    fun recordStep(
        observation: Observation,
        action: Action,
        isTerminal: Boolean = false,
        stepMetadata: Map<String, String> = emptyMap(),
    ) {
        check(started) { "Episode not started" }
        val step = Step(observation, action, isTerminal, stepMetadata)
        if (config.validateRlds) {
            val issues = validator.validateStep(step)
            handleIssues(issues)
        }
        steps.add(step)
        sink.onStep(step)
    }

    fun completeEpisode() {
        if (!started) return
        sink.onComplete()
        if (config.uploadOnComplete) {
            sink.episodeDir()?.let { uploader.upload(it) }
        }
        started = false
    }

    fun recordedCount(): Int = steps.size

    private fun handleIssues(issues: List<ValidationIssue>) {
        val errors = issues.filter { it.severity == ValidationIssue.Severity.ERROR }
        if (errors.isNotEmpty() && config.failOnValidationError) {
            val message = errors.joinToString("; ") { it.message }
            throw IllegalArgumentException("RLDS validation failed: $message")
        }
    }
}

interface EpisodeSink {
    fun onStart(metadata: EpisodeMetadata)
    fun onStep(step: Step)
    fun onComplete()
    fun episodeDir(): File?
}

private class FileEpisodeSink(private val config: LoggingConfig) : EpisodeSink {
    private val json = Json { encodeDefaults = true }
    private var stepsWriter: BufferedWriter? = null
    private var dir: File? = null

    override fun onStart(metadata: EpisodeMetadata) {
        dir = File(config.episodeOutputDir)
        dir?.mkdirs()
        dir?.let {
            File(it, "metadata.json").writeText(json.encodeToString(metadata))
            stepsWriter = File(it, "steps.jsonl").bufferedWriter()
        }
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

    override fun episodeDir(): File? = dir
}

private fun defaultUploader(config: LoggingConfig): RldsUploader {
    return if (config.uploadOnComplete && config.uploadEndpoint != null) {
        HttpRldsUploader(config)
    } else {
        NoopRldsUploader()
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
    val gaze: Gaze? = null,
    val gloveFrame: GloveFrame?,
    val robotState: RobotState?,
    val videoFrameId: String? = null,
    val depthFrameId: String? = null,
    val audio: Audio? = null,
    val uiContext: UiContext? = null,
    val diagnostics: Diagnostics = Diagnostics(),
)

@Serializable
data class Action(
    val command: List<Float>,
    val source: String,
    val annotation: Annotation? = null,
    val uiAction: UiAction? = null,
)

@Serializable
data class Pose(
    val position: List<Float> = listOf(0f, 0f, 0f),
    val orientationQuat: List<Float> = listOf(0f, 0f, 0f, 1f),
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
data class Gaze(
    val origin: List<Float>,
    val direction: List<Float>,
    val confidence: Float = 0f,
    val targetId: String? = null,
)

@Serializable
data class Step(
    val observation: Observation,
    val action: Action,
    val isTerminal: Boolean = false,
    val stepMetadata: Map<String, String> = emptyMap(),
)

@Serializable
data class Audio(
    val uri: String,
    val sampleRateHz: Int,
    val numChannels: Int,
    val format: String? = null,
    val frameId: String? = null,
)

@Serializable
data class UiContext(
    val activePanel: String? = null,
    val layout: Map<String, String> = emptyMap(),
    val focusContext: Map<String, String> = emptyMap(),
)
