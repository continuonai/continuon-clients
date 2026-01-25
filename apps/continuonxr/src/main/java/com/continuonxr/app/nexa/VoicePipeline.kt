package com.continuonxr.app.nexa

import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.*

/**
 * VoicePipeline connects audio capture to NexaSDK ASR for voice commands.
 *
 * Features:
 * - Real-time transcription
 * - Voice activity detection (VAD)
 * - Command extraction from transcripts
 * - Continuous listening mode
 */
class VoicePipeline(
    private val nexaManager: NexaManager,
    private val config: VoiceConfig = VoiceConfig()
) {
    companion object {
        private const val TAG = "VoicePipeline"
    }

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // Audio buffer for accumulating samples
    private val audioBuffer = MutableSharedFlow<ShortArray>(
        replay = 0,
        extraBufferCapacity = config.maxBufferedChunks,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    // Accumulated audio for current utterance
    private val currentUtteranceBuffer = mutableListOf<Short>()

    // Latest transcript
    private val _transcript = MutableStateFlow("")
    val transcript: StateFlow<String> = _transcript.asStateFlow()

    // Partial (in-progress) transcript
    private val _partialTranscript = MutableStateFlow("")
    val partialTranscript: StateFlow<String> = _partialTranscript.asStateFlow()

    // Pipeline state
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    // Voice activity state
    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()

    private var listeningJob: Job? = null
    private var silenceCounter = 0

    /**
     * Submit audio samples for processing.
     * @param samples PCM audio samples (16kHz, mono, 16-bit signed)
     */
    fun submitAudio(samples: ShortArray) {
        if (_isListening.value) {
            audioBuffer.tryEmit(samples)
        }
    }

    /**
     * Start listening for voice commands.
     */
    fun startListening() {
        if (listeningJob?.isActive == true) return

        Log.d(TAG, "Starting voice pipeline")
        _isListening.value = true
        _transcript.value = ""
        _partialTranscript.value = ""
        currentUtteranceBuffer.clear()
        silenceCounter = 0

        listeningJob = scope.launch {
            audioBuffer.collect { chunk ->
                processAudioChunk(chunk)
            }
        }
    }

    /**
     * Stop listening.
     */
    fun stopListening() {
        Log.d(TAG, "Stopping voice pipeline")
        _isListening.value = false
        listeningJob?.cancel()
        listeningJob = null

        // Process any remaining audio
        if (currentUtteranceBuffer.isNotEmpty()) {
            scope.launch {
                finalizeUtterance()
            }
        }
    }

    /**
     * Process a single audio chunk.
     */
    private suspend fun processAudioChunk(chunk: ShortArray) {
        // Voice activity detection
        val energy = calculateEnergy(chunk)
        val isSpeech = energy > config.vadThreshold

        if (isSpeech) {
            _isSpeaking.value = true
            silenceCounter = 0
            currentUtteranceBuffer.addAll(chunk.toList())

            // Real-time transcription if enabled
            if (config.enablePartialResults && currentUtteranceBuffer.size >= config.minSamplesForPartial) {
                updatePartialTranscript()
            }
        } else {
            silenceCounter++

            if (_isSpeaking.value && silenceCounter >= config.silenceChunksToEnd) {
                // End of utterance detected
                _isSpeaking.value = false
                finalizeUtterance()
            }
        }
    }

    /**
     * Update partial transcript from accumulated audio.
     */
    private suspend fun updatePartialTranscript() {
        if (_isProcessing.value) return

        _isProcessing.value = true
        try {
            val audio = currentUtteranceBuffer.toShortArray()
            val result = nexaManager.runAsr(audio)

            result.onSuccess { text ->
                _partialTranscript.value = text.trim()
            }
        } finally {
            _isProcessing.value = false
        }
    }

    /**
     * Finalize utterance and get full transcript.
     */
    private suspend fun finalizeUtterance() {
        if (currentUtteranceBuffer.isEmpty()) return

        Log.d(TAG, "Finalizing utterance: ${currentUtteranceBuffer.size} samples")

        _isProcessing.value = true
        try {
            val audio = currentUtteranceBuffer.toShortArray()
            val result = nexaManager.runAsr(audio)

            result.onSuccess { text ->
                val trimmed = text.trim()
                if (trimmed.isNotBlank()) {
                    _transcript.value = trimmed
                    Log.d(TAG, "Transcript: $trimmed")
                }
            }

            result.onFailure { e ->
                Log.e(TAG, "ASR failed", e)
            }
        } finally {
            _isProcessing.value = false
            currentUtteranceBuffer.clear()
            _partialTranscript.value = ""
        }
    }

    /**
     * Calculate audio energy for VAD.
     */
    private fun calculateEnergy(samples: ShortArray): Float {
        if (samples.isEmpty()) return 0f

        var sum = 0.0
        for (sample in samples) {
            sum += sample.toDouble() * sample.toDouble()
        }
        return (sum / samples.size).toFloat()
    }

    /**
     * Transcribe audio without listening mode (one-shot).
     */
    suspend fun transcribe(audio: ShortArray): Result<String> {
        return nexaManager.runAsr(audio)
    }

    /**
     * Clear current transcript.
     */
    fun clearTranscript() {
        _transcript.value = ""
        _partialTranscript.value = ""
    }

    /**
     * Release resources.
     */
    fun release() {
        stopListening()
        scope.cancel()
    }
}

/**
 * Configuration for VoicePipeline.
 */
data class VoiceConfig(
    val sampleRate: Int = 16000,               // Hz
    val chunkDurationMs: Int = 750,            // Audio chunk duration
    val vadThreshold: Float = 500f,            // Energy threshold for VAD
    val silenceChunksToEnd: Int = 3,           // Silence chunks before end of utterance
    val maxBufferedChunks: Int = 10,           // Max buffered audio chunks
    val enablePartialResults: Boolean = true,  // Real-time partial transcripts
    val minSamplesForPartial: Int = 8000       // Min samples before partial transcript
) {
    val samplesPerChunk: Int get() = (sampleRate * chunkDurationMs) / 1000
}

/**
 * Extension to convert List<Short> to ShortArray.
 */
private fun List<Short>.toShortArray(): ShortArray {
    return ShortArray(size) { this[it] }
}
