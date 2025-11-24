package com.continuonxr.app

import continuonxr.continuonbrain.v1.ControlMode
import continuonxr.continuonbrain.v1.EeVelocityCommand
import continuonxr.continuonbrain.v1.ReferenceFrame
import continuonxr.continuonbrain.v1.SendCommandRequest
import continuonxr.rlds.v1.Episode
import continuonxr.rlds.v1.EpisodeMetadata
import continuonxr.rlds.v1.RobotState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ProtoSchemaTest {
    @Test
    fun robotStateDefaultsAreInitialized() {
        val state = RobotState.newBuilder().build()

        assertTrue(state.jointPositionsList.isEmpty())
        assertTrue(state.jointVelocitiesList.isEmpty())
        assertEquals("", state.frameId)
    }

    @Test
    fun commandEnvelopeOneofIsRespected() {
        val velocityCommand = EeVelocityCommand.newBuilder()
            .setReferenceFrame(ReferenceFrame.REFERENCE_FRAME_BASE)
            .build()

        val envelope = SendCommandRequest.newBuilder()
            .setClientId("test-client")
            .setControlMode(ControlMode.CONTROL_MODE_EE_VELOCITY)
            .setTargetFrequencyHz(30.0)
            .setEeVelocity(velocityCommand)
            .build()

        assertTrue(envelope.hasEeVelocity())
        assertFalse(envelope.hasJointDelta())
        assertEquals(ControlMode.CONTROL_MODE_EE_VELOCITY, envelope.controlMode)
    }

    @Test
    fun episodeMetadataAcceptsTags() {
        val episode = Episode.newBuilder()
            .setMetadata(
                EpisodeMetadata.newBuilder()
                    .addTags("lab")
                    .setXrMode("trainer")
                    .setControlRole("human_teleop")
                    .build()
            )
            .build()

        assertEquals(listOf("lab"), episode.metadata.tagsList)
        assertEquals("trainer", episode.metadata.xrMode)
    }
}
