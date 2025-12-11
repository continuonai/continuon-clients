package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.glove.GloveFrame
import kotlinx.serialization.SerialName
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
    private val sink: EpisodeSink = defaultSink(config),
    private val validator: RldsValidator = RldsValidator(),
    private val uploader: RldsUploader = defaultUploader(config),
) {
    private val steps = mutableListOf<Step>()
    private var started = false
    private var currentMetadata: EpisodeMetadata? = null

    fun startEpisode(metadata: EpisodeMetadata) {
        steps.clear()
        if (config.validateRlds) {
            val issues = validator.validateEpisodeMetadata(metadata)
            handleIssues(issues)
        }
        sink.onStart(metadata)
        started = true
        currentMetadata = metadata
    }

    fun startEpisodeWithDefaults(metadata: EpisodeMetadata, defaults: EpisodeDefaults?) {
        startEpisode(metadata.withDefaults(defaults))
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
        val metadata = currentMetadata
        if (config.uploadOnComplete && metadata != null) {
            sink.episodeDir()?.let { uploader.enqueueUpload(it, metadata) }
        }
        started = false
        currentMetadata = null
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
        if (!config.writeEpisodesToDisk) return
        val root = File(config.episodeOutputDir)
        root.mkdirs()
        dir = File(root, "episode-${System.currentTimeMillis()}")
        dir?.mkdirs()
        dir?.let {
            File(it, "metadata.json").writeText(json.encodeToString(metadata))
            val stepsDir = File(it, "steps")
            stepsDir.mkdirs()
            val stepsFile = File(stepsDir, stepsFileName())
            stepsWriter = stepsFile.bufferedWriter()
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

    private fun stepsFileName(): String {
        return "000000.jsonl"
    }
}

private fun defaultUploader(config: LoggingConfig): RldsUploader {
    return if (config.uploadOnComplete && config.uploadEndpoint != null) {
        QueueingRldsUploader(
            transport = HttpRldsUploader(config),
            maxRetries = config.maxUploadRetries,
            retryBackoffMs = config.uploadRetryBackoffMs,
        )
    } else {
        NoopRldsUploader()
    }
}

private fun defaultSink(config: LoggingConfig): EpisodeSink {
    return if (config.writeEpisodesToDisk) FileEpisodeSink(config) else NoopEpisodeSink()
}

private class NoopEpisodeSink : EpisodeSink {
    override fun onStart(metadata: EpisodeMetadata) {}
    override fun onStep(step: Step) {}
    override fun onComplete() {}
    override fun episodeDir(): File? = null
}

@Serializable
data class EpisodeMetadata(
    @SerialName("continuon.xr_mode") val xrMode: String,
    @SerialName("continuon.control_role") val controlRole: String,
    @SerialName("environment_id") val environmentId: String,
    val software: SoftwareInfo? = null,
    val tags: List<String> = emptyList(),
    @SerialName("robot_id") val robotId: String? = null,
    @SerialName("robot_model") val robotModel: String? = null,
    @SerialName("frame_convention") val frameConvention: String? = null,
    @SerialName("start_time_unix_ms") val startTimeUnixMs: Long? = null,
    @SerialName("duration_ms") val durationMs: Long? = null,
)

@Serializable
data class SoftwareInfo(
    val xrAppVersion: String? = null,
    val brainVersion: String? = null,
    val gloveFirmwareVersion: String? = null,
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
    val videoTimestampNanos: Long? = null,
    val depthTimestampNanos: Long? = null,
    val audio: Audio? = null,
    val uiContext: UiContext? = null,
    val diagnostics: Diagnostics = Diagnostics(),
)

@Serializable
data class Action(
    val command: ControlCommand?,
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
    val gloveSampleRateHz: Float? = null,
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
