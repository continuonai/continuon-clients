package com.continuonxr.app.audio

import com.continuonxr.app.logging.Audio

/**
 * Stub audio capture pipeline. In production, wrap Android audio APIs and emit step-aligned audio frames/URIs.
 */
class AudioCapture {
    @Suppress("UNUSED_PARAMETER")
    fun start(onAudio: (Audio) -> Unit) {
        // TODO: Start microphone capture and deliver Audio frames with uri or buffer reference.
    }

    fun stop() {
        // TODO: Stop capture and release resources.
    }
}
