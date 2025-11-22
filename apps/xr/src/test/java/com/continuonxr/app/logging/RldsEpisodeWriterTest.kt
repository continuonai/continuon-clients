package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.glove.GloveFrame
import org.junit.Assert.assertEquals
import org.junit.Test

class RldsEpisodeWriterTest {
    @Test
    fun recordsStepCount() {
        val writer = RldsEpisodeWriter(
            LoggingConfig(
                episodeOutputDir = "/tmp/continuonxr/tests",
                uploadOnComplete = false,
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
                flex = floatArrayOf(0f, 0f, 0f, 0f, 0f),
                fsr = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
                orientationQuat = floatArrayOf(0f, 0f, 0f, 1f),
                accel = floatArrayOf(0f, 0f, 0f),
            ),
            robotState = RobotState(timestampNanos = 1L),
            videoFrameId = "frame-1",
            depthFrameId = null,
        )
        val action = Action(command = floatArrayOf(0.1f, 0.2f), source = "human_teleop_xr")

        writer.recordStep(observation = observation, action = action)

        assertEquals(1, writer.recordedCount())
    }
}

