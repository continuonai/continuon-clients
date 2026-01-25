package com.continuonxr.app.nexa

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.*

/**
 * VisionPipeline connects camera frames to NexaSDK VLM for scene understanding.
 *
 * Features:
 * - On-demand scene description
 * - Object detection with bounding boxes
 * - Backpressure handling (drops oldest frames)
 * - Configurable inference rate
 */
class VisionPipeline(
    private val nexaManager: NexaManager,
    private val config: VisionConfig = VisionConfig()
) {
    companion object {
        private const val TAG = "VisionPipeline"
    }

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // Frame buffer with backpressure (drop oldest)
    private val frameBuffer = MutableSharedFlow<CameraFrame>(
        replay = 0,
        extraBufferCapacity = config.maxBufferedFrames,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    // Latest detection results
    private val _detections = MutableStateFlow<List<Detection>>(emptyList())
    val detections: StateFlow<List<Detection>> = _detections.asStateFlow()

    // Scene description (updated on demand)
    private val _sceneDescription = MutableStateFlow<String>("")
    val sceneDescription: StateFlow<String> = _sceneDescription.asStateFlow()

    // Pipeline state
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    private var continuousDetectionJob: Job? = null

    /**
     * Submit a camera frame for processing.
     * If the buffer is full, the oldest frame is dropped.
     */
    fun submitFrame(frame: CameraFrame) {
        frameBuffer.tryEmit(frame)
    }

    /**
     * Describe the current scene (on-demand, single inference).
     */
    suspend fun describeScene(frame: CameraFrame): Result<String> {
        _isProcessing.value = true
        return try {
            val result = nexaManager.runVlm(
                prompt = DESCRIBE_SCENE_PROMPT,
                image = frame.bitmap
            )

            result.onSuccess { description ->
                _sceneDescription.value = description
            }

            result
        } finally {
            _isProcessing.value = false
        }
    }

    /**
     * Stream scene description tokens for real-time display.
     */
    fun describeSceneStream(frame: CameraFrame): Flow<String> = flow {
        _isProcessing.value = true
        try {
            nexaManager.runVlmStream(DESCRIBE_SCENE_PROMPT, frame.bitmap)
                .collect { token ->
                    emit(token)
                }
        } finally {
            _isProcessing.value = false
        }
    }

    /**
     * Detect objects in a frame.
     */
    suspend fun detectObjects(frame: CameraFrame): Result<List<Detection>> {
        _isProcessing.value = true
        return try {
            val result = nexaManager.runVlm(
                prompt = DETECT_OBJECTS_PROMPT,
                image = frame.bitmap
            )

            result.map { response ->
                parseDetections(response, frame.width, frame.height).also { detections ->
                    _detections.value = detections
                }
            }
        } finally {
            _isProcessing.value = false
        }
    }

    /**
     * Start continuous object detection on incoming frames.
     * Respects the configured inference rate.
     */
    fun startContinuousDetection() {
        if (continuousDetectionJob?.isActive == true) return

        Log.d(TAG, "Starting continuous detection at ${config.maxInferenceRate} fps")

        continuousDetectionJob = scope.launch {
            val intervalMs = (1000.0 / config.maxInferenceRate).toLong()

            frameBuffer.collect { frame ->
                val startTime = System.currentTimeMillis()

                detectObjects(frame)

                // Rate limiting
                val elapsed = System.currentTimeMillis() - startTime
                val delay = intervalMs - elapsed
                if (delay > 0) {
                    delay(delay)
                }
            }
        }
    }

    /**
     * Stop continuous detection.
     */
    fun stopContinuousDetection() {
        continuousDetectionJob?.cancel()
        continuousDetectionJob = null
        Log.d(TAG, "Stopped continuous detection")
    }

    /**
     * Find a specific object by name.
     * Returns bounding box if found.
     */
    suspend fun findObject(objectName: String, frame: CameraFrame): Result<Detection?> {
        val prompt = "Find the $objectName in this image. " +
                "If found, respond with: FOUND [x1,y1,x2,y2] where coordinates are 0-1 normalized. " +
                "If not found, respond with: NOT_FOUND"

        return nexaManager.runVlm(prompt, frame.bitmap).map { response ->
            if (response.contains("FOUND")) {
                parseSingleDetection(response, objectName, frame.width, frame.height)
            } else {
                null
            }
        }
    }

    /**
     * Answer a question about the scene.
     */
    suspend fun askAboutScene(question: String, frame: CameraFrame): Result<String> {
        return nexaManager.runVlm(question, frame.bitmap)
    }

    /**
     * Release resources.
     */
    fun release() {
        scope.cancel()
        _detections.value = emptyList()
        _sceneDescription.value = ""
    }

    // Parse VLM detection response into structured data
    private fun parseDetections(response: String, imageWidth: Int, imageHeight: Int): List<Detection> {
        val detections = mutableListOf<Detection>()

        // Expected format: "object_name [x1,y1,x2,y2] confidence"
        // or "1. object_name at [x1,y1,x2,y2]"
        val pattern = Regex("""(\w+(?:\s+\w+)?)\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*]?\s*(?:(\d+(?:\.\d+)?)%?)?""")

        pattern.findAll(response).forEach { match ->
            try {
                val name = match.groupValues[1].trim()
                val x1 = match.groupValues[2].toFloat()
                val y1 = match.groupValues[3].toFloat()
                val x2 = match.groupValues[4].toFloat()
                val y2 = match.groupValues[5].toFloat()
                val confidence = match.groupValues.getOrNull(6)?.toFloatOrNull() ?: 0.8f

                // Convert normalized to pixel coordinates
                val box = RectF(
                    x1 * imageWidth,
                    y1 * imageHeight,
                    x2 * imageWidth,
                    y2 * imageHeight
                )

                detections.add(Detection(
                    label = name,
                    boundingBox = box,
                    confidence = confidence.coerceIn(0f, 1f)
                ))
            } catch (e: Exception) {
                Log.w(TAG, "Failed to parse detection: ${match.value}", e)
            }
        }

        return detections
    }

    private fun parseSingleDetection(
        response: String,
        objectName: String,
        imageWidth: Int,
        imageHeight: Int
    ): Detection? {
        val pattern = Regex("""\[([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)]""")
        val match = pattern.find(response) ?: return null

        return try {
            val x1 = match.groupValues[1].toFloat()
            val y1 = match.groupValues[2].toFloat()
            val x2 = match.groupValues[3].toFloat()
            val y2 = match.groupValues[4].toFloat()

            Detection(
                label = objectName,
                boundingBox = RectF(
                    x1 * imageWidth,
                    y1 * imageHeight,
                    x2 * imageWidth,
                    y2 * imageHeight
                ),
                confidence = 0.9f
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse single detection", e)
            null
        }
    }

    companion object Prompts {
        const val DESCRIBE_SCENE_PROMPT = """Describe what you see in this image in 2-3 sentences.
Focus on objects a robot could interact with (items on tables, tools, containers).
Be concise and specific."""

        const val DETECT_OBJECTS_PROMPT = """List all objects you can see in this image that a robot could interact with.
For each object, provide:
- object_name [x1,y1,x2,y2] where coordinates are 0-1 normalized (top-left is 0,0)

Example format:
cup [0.3,0.4,0.5,0.7]
book [0.1,0.2,0.4,0.5]

Only list objects, no other text."""
    }
}

/**
 * Configuration for VisionPipeline.
 */
data class VisionConfig(
    val maxBufferedFrames: Int = 2,
    val maxInferenceRate: Float = 2f,  // FPS for continuous detection
    val resizeWidth: Int = 640,        // Resize frames before inference
    val resizeHeight: Int = 480
)

/**
 * A camera frame ready for processing.
 */
data class CameraFrame(
    val bitmap: Bitmap,
    val timestampNanos: Long,
    val width: Int = bitmap.width,
    val height: Int = bitmap.height
)

/**
 * An object detection result.
 */
data class Detection(
    val label: String,
    val boundingBox: RectF,
    val confidence: Float
)
