package com.continuonxr.app.connectivity

import continuonxr.continuonbrain.v1.ContinuonbrainLink
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ControlCommandMappingTest {
    @Test
    fun mapsEndEffectorVelocityToProtoEnvelope() {
        val command = ControlCommand.EndEffectorVelocity(
            linearMps = Vector3(0.1f, -0.2f, 0.3f),
            angularRadS = Vector3(1f, 2f, 3f),
            referenceFrame = ReferenceFrame.TOOL,
            targetFrequencyHz = 50.0,
            safety = SafetyStatus(estopReleasedAck = true, safetyToken = "token-1"),
        )

        val envelope = command.toProto("client-1")

        assertEquals("client-1", envelope.clientId)
        assertEquals(ContinuonbrainLink.ControlMode.CONTROL_MODE_EE_VELOCITY, envelope.controlMode)
        assertEquals(0.1f, envelope.eeVelocity.linearMps.x)
        assertEquals(-0.2f, envelope.eeVelocity.linearMps.y)
        assertEquals(0.3f, envelope.eeVelocity.linearMps.z)
        assertEquals(1f, envelope.eeVelocity.angularRadS.x)
        assertEquals(ContinuonbrainLink.ReferenceFrame.REFERENCE_FRAME_TOOL, envelope.eeVelocity.referenceFrame)
        assertEquals(50.0, envelope.targetFrequencyHz, 1e-3)
        assertTrue(envelope.hasSafety())
        assertEquals(true, envelope.safety.estopReleasedAck)
        assertEquals("token-1", envelope.safety.safetyToken)
    }

    @Test
    fun mapsGripperWithPosition() {
        val command = ControlCommand.Gripper(
            mode = GripperMode.POSITION,
            positionM = 0.12f,
            targetFrequencyHz = 5.0,
        )

        val envelope = command.toProto("client-xyz")

        assertEquals(ContinuonbrainLink.ControlMode.CONTROL_MODE_GRIPPER, envelope.controlMode)
        assertEquals(ContinuonbrainLink.GripperMode.GRIPPER_MODE_POSITION, envelope.gripper.mode)
        assertEquals(0.12f, envelope.gripper.positionM)
        assertEquals(5.0, envelope.targetFrequencyHz, 1e-3)
    }
}
