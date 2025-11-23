package com.continuonxr.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.continuonxr.app.logging.Audio
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Thin wrapper around [AudioRecord] that emits PCM snippets into teleop logging.
 */
class AudioCapture(
    private val outputDir: File = File("/sdcard/continuonxr/audio"),
    private val sampleRateHz: Int = 16_000,
    private val chunkDurationMs: Long = 750,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
    private var recordJob: Job? = null
    private var audioRecord: AudioRecord? = null

    fun start(onAudio: (Audio) -> Unit) {
        if (recordJob != null) return
        outputDir.mkdirs()
        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )

        audioRecord = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.MIC)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRateHz)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize * 2)
            .build()

        val record = audioRecord ?: return
        recordJob = scope.launch {
            val buffer = ByteArray(bufferSize)
            record.startRecording()
            try {
                while (isActive) {
                    val chunkFile = File(outputDir, "audio-${System.currentTimeMillis()}.pcm")
                    FileOutputStream(chunkFile).use { out ->
                        val start = System.currentTimeMillis()
                        while (isActive && System.currentTimeMillis() - start < chunkDurationMs) {
                            val bytesRead = record.read(buffer, 0, buffer.size)
                            if (bytesRead > 0) {
                                out.write(buffer, 0, bytesRead)
                            }
                        }
                    }
                    onAudio(
                        Audio(
                            uri = chunkFile.absolutePath,
                            sampleRateHz = sampleRateHz,
                            numChannels = 1,
                            format = "pcm16",
                            frameId = chunkFile.nameWithoutExtension,
                        )
                    )
                }
            } finally {
                if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    record.stop()
                }
                record.release()
            }
        }
    }

    suspend fun stopAndJoin() {
        recordJob?.cancelAndJoin()
        recordJob = null
        audioRecord = null
    }

    fun stop() {
        recordJob?.cancel()
        recordJob = null
        audioRecord = null
    }
}
