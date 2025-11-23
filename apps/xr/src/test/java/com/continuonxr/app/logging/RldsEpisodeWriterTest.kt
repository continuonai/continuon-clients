package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.glove.GloveFrame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path

class RldsEpisodeWriterTest {
    @Test
    fun recordsStepCount() {
        val outputDir = Files.createTempDirectory("rlds-writer-test")
        val writer = RldsEpisodeWriter(
            LoggingConfig(
                episodeOutputDir = outputDir.toString(),
                uploadOnComplete = false,
                validateRlds = true,
            )
        )
        writer.startEpisode(
            EpisodeMetadata(
                xrMode = "trainer",
                controlRole = "human_teleop",
                environmentId = "test",
            )
        )
        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = GloveFrame(
                timestampNanos = 1L,
                flex = listOf(0f, 0f, 0f, 0f, 0f),
                fsr = listOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
                orientationQuat = listOf(0f, 0f, 0f, 1f),
                accel = listOf(0f, 0f, 0f),
            ),
            robotState = RobotState(timestampNanos = 1L),
            videoFrameId = "frame-1",
            depthFrameId = null,
        )
        val action = Action(command = listOf(0.1f, 0.2f), source = "human_teleop_xr")

        writer.recordStep(observation = observation, action = action)
        assertEquals(1, writer.recordedCount())
    }

    @Test
    fun writesMetadataAndStepsFiles() {
        val outputDir: Path = Files.createTempDirectory("rlds-writer-files")
        val writer = RldsEpisodeWriter(
            LoggingConfig(
                episodeOutputDir = outputDir.toString(),
                uploadOnComplete = false,
                validateRlds = true,
            )
        )
        writer.startEpisode(
            EpisodeMetadata(
                xrMode = "trainer",
                controlRole = "human_teleop",
                environmentId = "test",
            )
        )
        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = null,
            robotState = RobotState(timestampNanos = 1L),
        )
        val action = Action(command = listOf(0.1f), source = "human_teleop_xr")
        writer.recordStep(observation = observation, action = action)
        writer.completeEpisode()

        assertTrue(Files.exists(outputDir.resolve("metadata.json")))
        assertTrue(Files.exists(outputDir.resolve("steps.jsonl")))
    }

    @Test(expected = IllegalArgumentException::class)
    fun failsValidationWhenCommandMissing() {
        val writer = RldsEpisodeWriter(
            LoggingConfig(
                episodeOutputDir = Files.createTempDirectory("rlds-writer-validation").toString(),
                uploadOnComplete = false,
                validateRlds = true,
                failOnValidationError = true,
            )
        )
        writer.startEpisode(
            EpisodeMetadata(
                xrMode = "trainer",
                controlRole = "human_teleop",
                environmentId = "test",
            )
        )
        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = null,
            robotState = RobotState(timestampNanos = 1L),
        )
        writer.recordStep(observation = observation, action = Action(command = emptyList(), source = ""))
    }
}
