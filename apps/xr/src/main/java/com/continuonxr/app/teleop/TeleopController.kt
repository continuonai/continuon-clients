package com.continuonxr.app.teleop

import com.continuonxr.app.connectivity.PixelBrainClient
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.glove.GloveBleClient
import com.continuonxr.app.glove.GloveFrame
import com.continuonxr.app.logging.*

/**
 * Coordinates teleoperation: input fusion -> command -> RLDS logging.
 */
class TeleopController(
    private val pixelBrainClient: PixelBrainClient,
    private val rldsWriter: RldsEpisodeWriter,
    private val gloveBleClient: GloveBleClient,
    private val inputFusion: InputFusion = DefaultInputFusion(),
    private val commandMapper: CommandMapper = DefaultCommandMapper(),
    private val timestampProvider: () -> Long = { System.nanoTime() },
) {
    fun startTeleopSession() {
        pixelBrainClient.connect()
        rldsWriter.startEpisode(
            EpisodeMetadata(
                xrMode = "trainer",
                controlRole = "human_teleop",
                environmentId = "lab-mock",
            )
        )
        gloveBleClient.connect(onFrame = { onGloveFrame(it) }, onDiagnostics = { /* TODO */ })
        pixelBrainClient.observeState { onRobotState(it) }
    }

    private fun onGloveFrame(frame: GloveFrame) {
        inputFusion.updateGlove(frame)
    }

    private fun onRobotState(state: RobotState) {
        val fused = inputFusion.currentObservation(state, timestampProvider())
        val command = commandMapper.map(fused)
        pixelBrainClient.sendCommand(command)
        rldsWriter.recordStep(
            observation = fused.toObservation(),
            action = Action(command = command, source = "human_teleop_xr"),
        )
    }
}

interface InputFusion {
    fun updateGlove(frame: GloveFrame)
    fun currentObservation(robotState: RobotState, timestampNanos: Long): FusedObservation
}

interface CommandMapper {
    fun map(fused: FusedObservation): FloatArray
}

data class FusedObservation(
    val headsetPose: Pose = Pose(),
    val rightHandPose: Pose = Pose(),
    val leftHandPose: Pose? = null,
    val gloveFrame: GloveFrame? = null,
    val robotState: RobotState? = null,
) {
    fun toObservation(): Observation = Observation(
        headsetPose = headsetPose,
        rightHandPose = rightHandPose,
        leftHandPose = leftHandPose,
        gloveFrame = gloveFrame,
        robotState = robotState,
    )
}

private class DefaultInputFusion : InputFusion {
    private var latestGlove: GloveFrame? = null

    override fun updateGlove(frame: GloveFrame) {
        latestGlove = frame
    }

    override fun currentObservation(robotState: RobotState, timestampNanos: Long): FusedObservation {
        // TODO: Add headset/hand poses from XR runtime.
        return FusedObservation(
            gloveFrame = latestGlove,
            robotState = robotState,
        )
    }
}

private class DefaultCommandMapper : CommandMapper {
    override fun map(fused: FusedObservation): FloatArray {
        // TODO: Map pose/gesture to normalized command vector expected by Robot API.
        return floatArrayOf()
    }
}

