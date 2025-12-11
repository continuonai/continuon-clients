package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.GripperMode
import com.continuonxr.app.connectivity.ReferenceFrame
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.connectivity.Vector3
import com.continuonxr.app.glove.GloveFrame
import com.continuonxr.app.logging.Diagnostics
import com.continuonxr.app.logging.EpisodeDefaults
import com.continuonxr.app.logging.withDefaults
import com.continuonxr.app.logging.withVideoDepthTimestamps
import com.continuonxr.app.logging.withGloveDiagnostics
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
        writer.startEpisodeWithDefaults(
            EpisodeMetadata(
                xrMode = "trainer",
                controlRole = "human_teleop",
                environmentId = "",
                robotId = null,
                robotModel = null,
                frameConvention = null,
                startTimeUnixMs = null,
                durationMs = null,
            ),
            EpisodeDefaults(
                environmentId = "test",
                robotId = "robot-1",
                robotModel = "pi5-donkey",
                frameConvention = "base_link",
                appVersion = "1.0.0",
                brainVersion = "0.9.0",
                gloveFirmwareVersion = "0.1.0",
                tags = listOf("tag-a"),
            ),
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
            robotState = RobotState(timestampNanos = 1L, wallTimeMillis = 1L, frameId = "frame-1"),
            videoFrameId = "frame-1",
            depthFrameId = null,
        )
            .withVideoDepthTimestamps(videoTimestampNanos = 1L)
            .withGloveDiagnostics(glove = null)
        val action = Action(
            command = ControlCommand.EndEffectorVelocity(
                linearMps = Vector3(0.1f, 0.2f, 0.0f),
                angularRadS = Vector3(0f, 0f, 0f),
                referenceFrame = ReferenceFrame.BASE,
            ),
            source = "human_teleop_xr",
        )

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
                robotId = "robot-1",
            )
        )
        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = GloveFrame(
                timestampNanos = 1L,
                flex = List(5) { 0f },
                fsr = List(8) { 0f },
                orientationQuat = listOf(0f, 0f, 0f, 1f),
                accel = listOf(0f, 0f, 0f),
            ),
            robotState = RobotState(timestampNanos = 1L, wallTimeMillis = 1L, frameId = "frame-1"),
            videoFrameId = "frame-1",
        ).withVideoDepthTimestamps(videoTimestampNanos = 1L)
        val action = Action(
            command = ControlCommand.Gripper(
                mode = GripperMode.POSITION,
                positionM = 0.1f,
            ),
            source = "human_teleop_xr",
        )
        writer.recordStep(observation = observation, action = action)
        writer.completeEpisode()

        val metadata = Files.walk(outputDir)
            .filter { it.fileName.toString() == "metadata.json" }
            .findFirst()
        val steps = Files.walk(outputDir)
            .filter { it.fileName.toString() == "000000.jsonl" }
            .findFirst()
        assertTrue(metadata.isPresent)
        assertTrue(steps.isPresent)
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
                robotId = "robot-1",
            )
        )
        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = GloveFrame(
                timestampNanos = 1L,
                flex = List(5) { 0f },
                fsr = List(8) { 0f },
                orientationQuat = listOf(0f, 0f, 0f, 1f),
                accel = listOf(0f, 0f, 0f),
            ),
            robotState = RobotState(timestampNanos = 1L, wallTimeMillis = 1L, frameId = "frame-1"),
            videoFrameId = "frame-1",
        ).withVideoDepthTimestamps(videoTimestampNanos = 1L)
        writer.recordStep(
            observation = observation,
            action = Action(command = null, source = ""),
        )
    }
}
