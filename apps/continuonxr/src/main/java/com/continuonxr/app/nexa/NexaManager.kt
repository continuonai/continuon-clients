package com.continuonxr.app.nexa

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * NexaManager handles the lifecycle and inference of NexaSDK models.
 *
 * Supports:
 * - VLM (Vision-Language Model) for scene understanding
 * - ASR (Automatic Speech Recognition) for voice commands
 *
 * All inference runs on Qualcomm Hexagon NPU when available, with fallback to GPU/CPU.
 */
class NexaManager(private val context: Context) {

    companion object {
        private const val TAG = "NexaManager"

        // Model names for NexaSDK
        const val VLM_MODEL = "omni-neural-4b"
        const val ASR_MODEL = "whisper-small"

        // Plugin IDs for hardware acceleration
        const val PLUGIN_NPU = "npu"
        const val PLUGIN_GPU = "gpu"
        const val PLUGIN_CPU = "cpu"
    }

    // Internal state
    private var isInitialized = false
    private var vlmWrapper: NexaVlmWrapper? = null
    private var asrWrapper: NexaAsrWrapper? = null

    private val initMutex = Mutex()
    private val vlmMutex = Mutex()
    private val asrMutex = Mutex()

    /**
     * Initialize NexaSDK and download models if needed.
     * Call this early (e.g., in Application.onCreate or MainActivity.onCreate).
     */
    suspend fun initialize(): Result<Unit> = withContext(Dispatchers.IO) {
        initMutex.withLock {
            if (isInitialized) {
                return@withContext Result.success(Unit)
            }

            try {
                Log.d(TAG, "Initializing NexaSDK...")

                // Initialize NexaSDK
                NexaSdkWrapper.initialize(context)

                // Check NPU availability
                val npuAvailable = NexaSdkWrapper.isNpuAvailable()
                Log.d(TAG, "NPU available: $npuAvailable")

                isInitialized = true
                Log.d(TAG, "NexaSDK initialized successfully")
                Result.success(Unit)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize NexaSDK", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Load the VLM model for vision tasks.
     */
    suspend fun loadVlm(): Result<Unit> = withContext(Dispatchers.IO) {
        vlmMutex.withLock {
            if (vlmWrapper != null) {
                return@withContext Result.success(Unit)
            }

            try {
                Log.d(TAG, "Loading VLM model: $VLM_MODEL")

                val pluginId = if (NexaSdkWrapper.isNpuAvailable()) PLUGIN_NPU else PLUGIN_GPU
                val modelPath = getModelPath(VLM_MODEL)

                vlmWrapper = NexaVlmWrapper.create(
                    modelName = VLM_MODEL,
                    modelPath = modelPath,
                    pluginId = pluginId
                )

                Log.d(TAG, "VLM loaded on $pluginId")
                Result.success(Unit)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load VLM", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Load the ASR model for speech recognition.
     */
    suspend fun loadAsr(): Result<Unit> = withContext(Dispatchers.IO) {
        asrMutex.withLock {
            if (asrWrapper != null) {
                return@withContext Result.success(Unit)
            }

            try {
                Log.d(TAG, "Loading ASR model: $ASR_MODEL")

                val pluginId = if (NexaSdkWrapper.isNpuAvailable()) PLUGIN_NPU else PLUGIN_GPU
                val modelPath = getModelPath(ASR_MODEL)

                asrWrapper = NexaAsrWrapper.create(
                    modelName = ASR_MODEL,
                    modelPath = modelPath,
                    pluginId = pluginId
                )

                Log.d(TAG, "ASR loaded on $pluginId")
                Result.success(Unit)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load ASR", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Run VLM inference on an image.
     * Returns the model's text response.
     */
    suspend fun runVlm(prompt: String, image: Bitmap): Result<String> = withContext(Dispatchers.IO) {
        val wrapper = vlmWrapper ?: return@withContext Result.failure(
            IllegalStateException("VLM not loaded. Call loadVlm() first.")
        )

        try {
            Log.d(TAG, "Running VLM with prompt: ${prompt.take(50)}...")
            val startTime = System.currentTimeMillis()

            val response = wrapper.generate(prompt, image)

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "VLM completed in ${elapsed}ms")

            Result.success(response)
        } catch (e: Exception) {
            Log.e(TAG, "VLM inference failed", e)
            Result.failure(e)
        }
    }

    /**
     * Stream VLM inference tokens for real-time display.
     */
    fun runVlmStream(prompt: String, image: Bitmap): Flow<String> = flow {
        val wrapper = vlmWrapper ?: throw IllegalStateException("VLM not loaded")

        wrapper.generateStream(prompt, image).collect { token ->
            emit(token)
        }
    }

    /**
     * Run ASR on audio samples.
     * @param audio PCM audio samples (16kHz, mono, 16-bit)
     * @return Transcribed text
     */
    suspend fun runAsr(audio: ShortArray): Result<String> = withContext(Dispatchers.IO) {
        val wrapper = asrWrapper ?: return@withContext Result.failure(
            IllegalStateException("ASR not loaded. Call loadAsr() first.")
        )

        try {
            Log.d(TAG, "Running ASR on ${audio.size} samples")
            val startTime = System.currentTimeMillis()

            val transcript = wrapper.transcribe(audio)

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "ASR completed in ${elapsed}ms: ${transcript.take(50)}")

            Result.success(transcript)
        } catch (e: Exception) {
            Log.e(TAG, "ASR inference failed", e)
            Result.failure(e)
        }
    }

    /**
     * Stream ASR for real-time transcription.
     */
    fun runAsrStream(audioFlow: Flow<ShortArray>): Flow<String> = flow {
        val wrapper = asrWrapper ?: throw IllegalStateException("ASR not loaded")

        audioFlow.collect { chunk ->
            val transcript = wrapper.transcribe(chunk)
            if (transcript.isNotBlank()) {
                emit(transcript)
            }
        }
    }

    /**
     * Check if models are loaded and ready.
     */
    fun isVlmReady(): Boolean = vlmWrapper != null
    fun isAsrReady(): Boolean = asrWrapper != null
    fun isReady(): Boolean = isInitialized && isVlmReady() && isAsrReady()

    /**
     * Release all models and free resources.
     * Call this in Activity.onDestroy or when no longer needed.
     */
    fun release() {
        Log.d(TAG, "Releasing NexaManager resources")

        vlmWrapper?.release()
        vlmWrapper = null

        asrWrapper?.release()
        asrWrapper = null
    }

    /**
     * Unload models but keep SDK initialized (for memory pressure).
     */
    suspend fun unloadModels() {
        vlmMutex.withLock {
            vlmWrapper?.release()
            vlmWrapper = null
        }
        asrMutex.withLock {
            asrWrapper?.release()
            asrWrapper = null
        }
        Log.d(TAG, "Models unloaded")
    }

    private fun getModelPath(modelName: String): String {
        val modelsDir = File(context.filesDir, "nexa_models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }
        return File(modelsDir, modelName).absolutePath
    }
}
