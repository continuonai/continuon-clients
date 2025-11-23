package com.continuonxr.app.logging

import java.io.File

/**
 * Stub uploader for RLDS episodes. Replace with real upload to Continuon Cloud.
 */
interface RldsUploader {
    fun upload(episodeDir: File)
}

class NoopRldsUploader : RldsUploader {
    override fun upload(episodeDir: File) {
        // Intentionally no-op in stub.
    }
}
