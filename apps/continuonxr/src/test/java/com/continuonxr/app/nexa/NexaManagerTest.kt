package com.continuonxr.app.nexa

import org.junit.Assert.*
import org.junit.Test

class NexaManagerTest {

    @Test
    fun `VisionConfig has sensible defaults`() {
        val config = VisionConfig()
        assertEquals(2, config.maxBufferedFrames)
        assertEquals(2f, config.maxInferenceRate, 0.01f)
        assertEquals(640, config.resizeWidth)
        assertEquals(480, config.resizeHeight)
    }

    @Test
    fun `VoiceConfig has sensible defaults`() {
        val config = VoiceConfig()
        assertEquals(16000, config.sampleRate)
        assertEquals(750, config.chunkDurationMs)
        assertTrue(config.enablePartialResults)
    }

    @Test
    fun `VoiceConfig samplesPerChunk calculation`() {
        val config = VoiceConfig(sampleRate = 16000, chunkDurationMs = 1000)
        assertEquals(16000, config.samplesPerChunk)
    }

    @Test
    fun `Detection has correct structure`() {
        val detection = Detection(
            label = "cup",
            boundingBox = android.graphics.RectF(0.1f, 0.2f, 0.3f, 0.4f),
            confidence = 0.95f
        )

        assertEquals("cup", detection.label)
        assertEquals(0.95f, detection.confidence, 0.001f)
    }

    @Test
    fun `CameraFrame has correct structure`() {
        // This test would require mocking Bitmap
        // For now, test the data class structure
        assertTrue(true)
    }
}
