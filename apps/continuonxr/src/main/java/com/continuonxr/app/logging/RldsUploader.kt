package com.continuonxr.app.logging

import java.io.File
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/**
 * RLDS uploader that supports queuing, retries, and tagging for Continuon Cloud uploads.
 */
interface RldsUploader {
    fun enqueueUpload(episodeDir: File, metadata: EpisodeMetadata)
}

class NoopRldsUploader : RldsUploader {
    override fun enqueueUpload(episodeDir: File, metadata: EpisodeMetadata) {
        // Intentionally no-op in stub.
    }
}

class QueueingRldsUploader(
    private val transport: RldsUploadTransport,
    private val maxRetries: Int,
    private val retryBackoffMs: Long,
    private val executor: Executor = Executors.newSingleThreadExecutor(),
) : RldsUploader {
    override fun enqueueUpload(episodeDir: File, metadata: EpisodeMetadata) {
        executor.execute {
            var attempt = 0
            while (attempt < maxRetries) {
                val success = transport.upload(episodeDir, metadata)
                if (success) return@execute
                attempt += 1
                if (attempt < maxRetries) {
                    Thread.sleep(retryBackoffMs)
                }
            }
        }
    }
}

interface RldsUploadTransport {
    fun upload(episodeDir: File, metadata: EpisodeMetadata): Boolean
}
