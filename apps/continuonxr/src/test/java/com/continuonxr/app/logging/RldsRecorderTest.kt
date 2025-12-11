package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.ReferenceFrame
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.connectivity.Vector3
import com.continuonxr.app.glove.GloveDiagnostics
import com.continuonxr.app.glove.GloveFrame
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Test

class RldsRecorderTest {

    @Test
    fun appliesDefaultsAndDiagnostics() {
        val outputDir = Files.createTempDirectory("rlds-recorder").toString()
        val recorder = RldsRecorder.fromConfig(
            LoggingConfig(
                episodeOutputDir = outputDir,
                uploadOnComplete = false,
                environmentId = "lab-env",
                robotId = "robot-123",
                robotModel = "pi5-donkey",
                frameConvention = "base_link",
                appVersion = "1.2.3",
                brainVersion = "0.9.9",
                gloveFirmwareVersion = "0.1.1",
                tags = listOf("tag-a", "tag-b"),
            )
        )

        recorder.start(
            EpisodeMetadata(
                xrMode = "trainer",
                controlRole = "human_teleop",
                environmentId = "",
            )
        )

        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = GloveFrame(
                timestampNanos = 5_000_000L,
                flex = List(5) { 0f },
                fsr = List(8) { 0f },
                orientationQuat = listOf(0f, 0f, 0f, 1f),
                accel = listOf(0f, 0f, 0f),
                valid = false,
            ),
            robotState = RobotState(timestampNanos = 5_000_000L, wallTimeMillis = 10L, frameId = "frame-1"),
            videoFrameId = "frame-1",
            depthFrameId = "frame-1",
        )

        val action = Action(
            command = ControlCommand.EndEffectorVelocity(
                linearMps = Vector3(0f, 0f, 0f),
                angularRadS = Vector3(0f, 0f, 0f),
                referenceFrame = ReferenceFrame.BASE,
            ),
            source = "human_teleop_xr",
        )

        recorder.recordStep(
            observation = observation,
            action = action,
            gloveDiagnostics = GloveDiagnostics(mtu = 64, sampleRateHz = 100f, dropCount = 1, rssi = -60),
            videoTimestampNanos = 5_000_000L,
            depthTimestampNanos = 5_000_000L,
        )

        recorder.completeEpisode()
        assertEquals(1, recorder.recordedCount())
    }
}

