package com.continuonxr.app.logging

import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.ReferenceFrame
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.connectivity.Vector3
import com.continuonxr.app.glove.GloveFrame
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RldsValidatorSchemaTest {
    private val validator = RldsValidator()

    @Test
    fun rejectsMissingGloveOrVideoFrame() {
        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = null,
            robotState = RobotState(timestampNanos = 1L, wallTimeMillis = 10L, frameId = "frame-1"),
            videoFrameId = null,
        )
        val issues = validator.validateStep(
            Step(
                observation = observation,
                action = Action(
                    command = ControlCommand.EndEffectorVelocity(
                        linearMps = Vector3(0f, 0f, 0f),
                        angularRadS = Vector3(0f, 0f, 0f),
                        referenceFrame = ReferenceFrame.BASE,
                        targetFrequencyHz = 100.0,
                    ),
                    source = "human_teleop_xr",
                ),
            ),
        )

        assertFalse(issues.isEmpty())
    }

    @Test
    fun acceptsSchemaCompliantStep() {
        val observation = Observation(
            headsetPose = Pose(),
            rightHandPose = Pose(),
            leftHandPose = null,
            gloveFrame = GloveFrame(
                timestampNanos = 1L,
                flex = List(5) { 0.5f },
                fsr = List(8) { 0.1f },
                orientationQuat = listOf(0f, 0f, 0f, 1f),
                accel = listOf(0f, 0f, 0f),
            ),
            robotState = RobotState(timestampNanos = 1L, wallTimeMillis = 10L, frameId = "frame-1"),
            videoFrameId = "frame-1",
        )
        val issues = validator.validateStep(
            Step(
                observation = observation,
                action = Action(
                    command = ControlCommand.EndEffectorVelocity(
                        linearMps = Vector3(0f, 0f, 0f),
                        angularRadS = Vector3(0f, 0f, 0f),
                        referenceFrame = ReferenceFrame.BASE,
                        targetFrequencyHz = 100.0,
                    ),
                    source = "human_teleop_xr",
                ),
            ),
        )

        assertTrue(issues.isEmpty())
    }
}
