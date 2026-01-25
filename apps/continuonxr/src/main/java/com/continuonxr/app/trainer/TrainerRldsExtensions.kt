package com.continuonxr.app.trainer

import com.continuonxr.app.logging.*
import com.continuonxr.app.nexa.CameraFrame
import com.continuonxr.app.nexa.Detection
import kotlinx.serialization.Serializable
import java.io.File
import java.util.UUID

/**
 * Extensions for RLDS recording in Trainer mode.
 *
 * Adds support for:
 * - Camera frames with video frame IDs
 * - Voice transcripts
 * - VLM scene descriptions
 * - Object detections
 * - Teaching mode context
 */

/**
 * Trainer-specific observation data.
 */
@Serializable
data class TrainerObservation(
    // Voice data
    val voiceTranscript: String? = null,
    val voiceConfidence: Float? = null,

    // Vision data
    val sceneDescription: String? = null,
    val detections: List<DetectionRecord>? = null,

    // Teaching mode
    val teachingState: String? = null,  // "idle", "recording:[name]", "playing:[name]"
    val teachingStepIndex: Int? = null,

    // Device info
    val deviceOrientation: String? = null,  // "portrait", "landscape"
    val inputSource: String? = null  // "voice", "joystick", "slider", "button"
)

/**
 * Serializable detection record.
 */
@Serializable
data class DetectionRecord(
    val label: String,
    val boundingBox: List<Float>,  // [x1, y1, x2, y2] normalized 0-1
    val confidence: Float
)

/**
 * Extension to create trainer observation from current state.
 */
fun createTrainerObservation(
    baseObservation: Observation,
    voiceTranscript: String? = null,
    sceneDescription: String? = null,
    detections: List<Detection>? = null,
    teachingState: TeachingState? = null,
    inputSource: String? = null
): Observation {
    // Encode trainer data in stepMetadata since Observation is fixed
    return baseObservation.copy(
        uiContext = baseObservation.uiContext?.copy(
            focusContext = baseObservation.uiContext.focusContext + buildMap {
                voiceTranscript?.let { put("voice_transcript", it) }
                sceneDescription?.let { put("scene_description", it) }
                inputSource?.let { put("input_source", it) }
                teachingState?.let { put("teaching_state", it.toStateString()) }
            }
        ) ?: UiContext(
            activePanel = "trainer",
            focusContext = buildMap {
                voiceTranscript?.let { put("voice_transcript", it) }
                sceneDescription?.let { put("scene_description", it) }
                inputSource?.let { put("input_source", it) }
                teachingState?.let { put("teaching_state", it.toStateString()) }
            }
        )
    )
}

/**
 * Extension to convert TeachingState to string.
 */
fun TeachingState.toStateString(): String = when (this) {
    is TeachingState.Idle -> "idle"
    is TeachingState.Recording -> "recording:${this.behaviorName}"
    is TeachingState.Playing -> "playing:${this.behaviorName}"
}

/**
 * Extension to convert Detection to DetectionRecord.
 */
fun Detection.toRecord(imageWidth: Int, imageHeight: Int): DetectionRecord {
    return DetectionRecord(
        label = label,
        boundingBox = listOf(
            boundingBox.left / imageWidth,
            boundingBox.top / imageHeight,
            boundingBox.right / imageWidth,
            boundingBox.bottom / imageHeight
        ),
        confidence = confidence
    )
}

/**
 * TrainerRldsRecorder wraps RldsEpisodeWriter with trainer-specific functionality.
 */
class TrainerRldsRecorder(
    private val writer: RldsEpisodeWriter,
    private val videoDir: File
) {
    private var stepIndex = 0
    private var currentEpisodeId: String? = null

    /**
     * Start a new training episode.
     */
    fun startEpisode(
        robotId: String? = null,
        environmentId: String = "trainer-session"
    ) {
        currentEpisodeId = UUID.randomUUID().toString()
        stepIndex = 0

        val metadata = EpisodeMetadata(
            xrMode = "trainer",
            controlRole = "human_teleop",
            environmentId = environmentId,
            robotId = robotId,
            tags = listOf("trainer", "nexasdk", "android"),
            startTimeUnixMs = System.currentTimeMillis(),
            software = SoftwareInfo(
                xrAppVersion = "1.0.0-trainer"
            )
        )

        writer.startEpisode(metadata)
    }

    /**
     * Record a step with camera frame and trainer context.
     */
    fun recordStep(
        observation: Observation,
        action: Action,
        frame: CameraFrame? = null,
        voiceTranscript: String? = null,
        sceneDescription: String? = null,
        detections: List<Detection>? = null,
        teachingState: TeachingState = TeachingState.Idle,
        inputSource: String? = null,
        isTerminal: Boolean = false
    ) {
        // Save camera frame if provided
        val videoFrameId = frame?.let { saveFrame(it) }

        // Create enhanced observation
        val enhancedObservation = observation.copy(
            videoFrameId = videoFrameId,
            videoTimestampNanos = frame?.timestampNanos,
            uiContext = UiContext(
                activePanel = "trainer",
                layout = mapOf("mode" to "portrait"),
                focusContext = buildMap {
                    voiceTranscript?.let { put("voice_transcript", it) }
                    sceneDescription?.let { put("scene_description", it) }
                    inputSource?.let { put("input_source", it) }
                    put("teaching_state", teachingState.toStateString())
                    detections?.let {
                        put("detections_count", it.size.toString())
                        put("detections_labels", it.map { d -> d.label }.joinToString(","))
                    }
                }
            )
        )

        // Record step
        writer.recordStep(
            observation = enhancedObservation,
            action = action,
            isTerminal = isTerminal,
            stepMetadata = mapOf(
                "step_index" to stepIndex.toString(),
                "input_source" to (inputSource ?: "unknown")
            )
        )

        stepIndex++
    }

    /**
     * Save a camera frame and return its ID.
     */
    private fun saveFrame(frame: CameraFrame): String {
        val frameId = "frame_${System.currentTimeMillis()}_$stepIndex"
        val frameFile = File(videoDir, "$frameId.jpg")

        try {
            frameFile.outputStream().use { out ->
                frame.bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 80, out)
            }
        } catch (e: Exception) {
            // Log but don't fail recording
            android.util.Log.w("TrainerRldsRecorder", "Failed to save frame: $e")
        }

        return frameId
    }

    /**
     * Complete the current episode.
     */
    fun completeEpisode() {
        writer.completeEpisode()
        currentEpisodeId = null
        stepIndex = 0
    }

    /**
     * Get the number of recorded steps.
     */
    fun recordedCount(): Int = writer.recordedCount()

    /**
     * Check if recording is active.
     */
    fun isRecording(): Boolean = currentEpisodeId != null
}

/**
 * Create a default trainer RLDS recorder.
 */
fun createTrainerRldsRecorder(
    outputDir: File,
    videoDir: File = File(outputDir, "video_frames")
): TrainerRldsRecorder {
    videoDir.mkdirs()

    val config = com.continuonxr.app.config.LoggingConfig(
        episodeOutputDir = outputDir.absolutePath,
        writeEpisodesToDisk = true,
        validateRlds = true,
        failOnValidationError = false
    )

    val writer = RldsEpisodeWriter(config)
    return TrainerRldsRecorder(writer, videoDir)
}
