package com.continuonxr.app.logging

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.file.Files
import java.util.concurrent.Executor

class QueueingRldsUploaderTest {
    @Test
    fun retriesUntilSuccess() {
        val attempts = mutableListOf<Int>()
        val transport = object : RldsUploadTransport {
            override fun upload(episodeDir: java.io.File, metadata: EpisodeMetadata): Boolean {
                attempts += attempts.size
                return attempts.size >= 2
            }
        }
        val uploader = QueueingRldsUploader(
            transport = transport,
            maxRetries = 3,
            retryBackoffMs = 0,
            executor = Executor { command -> command.run() },
        )

        uploader.enqueueUpload(tempEpisodeDir(), sampleMetadata())

        assertEquals(2, attempts.size)
    }

    @Test
    fun stopsAfterFailures() {
        val attempts = mutableListOf<Int>()
        val transport = object : RldsUploadTransport {
            override fun upload(episodeDir: java.io.File, metadata: EpisodeMetadata): Boolean {
                attempts += attempts.size
                return false
            }
        }
        val uploader = QueueingRldsUploader(
            transport = transport,
            maxRetries = 2,
            retryBackoffMs = 0,
            executor = Executor { command -> command.run() },
        )

        uploader.enqueueUpload(tempEpisodeDir(), sampleMetadata())

        assertEquals(2, attempts.size)
    }

    private fun tempEpisodeDir() = Files.createTempDirectory("queueing-uploader-test").toFile()

    private fun sampleMetadata() = EpisodeMetadata(
        xrMode = "trainer",
        controlRole = "human_teleop",
        environmentId = "test-environment",
    )
}
