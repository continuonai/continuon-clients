package com.continuonxr.app.teleop

import com.continuonxr.app.audio.AudioCapture
import com.continuonxr.app.connectivity.ContinuonBrainClient
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.RobotState
import com.continuonxr.app.connectivity.ReferenceFrame
import com.continuonxr.app.connectivity.Vector3
import com.continuonxr.app.glove.GloveBleClient
import com.continuonxr.app.glove.GloveFrame
import com.continuonxr.app.logging.*
import com.continuonxr.app.ui.UiContextTracker

/**
 * Coordinates teleoperation: input fusion -> command -> RLDS logging.
 */
class TeleopController(
    private val continuonBrainClient: ContinuonBrainClient,
    private val rldsWriter: RldsEpisodeWriter,
    private val gloveBleClient: GloveBleClient,
    private val audioCapture: AudioCapture = AudioCapture(),
    private val uiContextTracker: UiContextTracker = UiContextTracker(),
    private val inputFusion: InputFusion = DefaultInputFusion(),
    private val commandMapper: CommandMapper = DefaultCommandMapper(),
    private val timestampProvider: () -> Long = { System.nanoTime() },
) {
    fun startTeleopSession() {
        continuonBrainClient.connect()
        rldsWriter.startEpisode(
            EpisodeMetadata(
                xrMode = "trainer",
                controlRole = "human_teleop",
                environmentId = "lab-mock",
            )
        )
        gloveBleClient.connect(onFrame = { onGloveFrame(it) }, onDiagnostics = { /* TODO */ })
        audioCapture.start { onAudio(it) }
        continuonBrainClient.observeState { onRobotState(it) }
    }

    private fun onGloveFrame(frame: GloveFrame) {
        inputFusion.updateGlove(frame)
    }

    private fun onAudio(audio: Audio) {
        inputFusion.updateAudio(audio)
    }

    fun onGazeUpdate(gaze: Gaze) {
        inputFusion.updateGaze(gaze)
    }

    fun onHeadsetPose(pose: Pose) {
        inputFusion.updateHeadsetPose(pose)
    }

    fun onRightHandPose(pose: Pose) {
        inputFusion.updateRightHandPose(pose)
    }

    fun onLeftHandPose(pose: Pose?) {
        inputFusion.updateLeftHandPose(pose)
    }

    fun onUiContextUpdate(uiContext: UiContext) {
        uiContextTracker.update(uiContext)
        inputFusion.updateUiContext(uiContext)
    }

    private fun onRobotState(state: RobotState) {
        val fused = inputFusion.currentObservation(state, timestampProvider())
        val command = commandMapper.map(fused)
        continuonBrainClient.sendCommand(command)
        rldsWriter.recordStep(
            observation = fused.toObservation(uiContextTracker.current()),
            action = Action(command = command, source = "human_teleop_xr"),
        )
    }

    fun onUiAction(uiAction: UiAction) {
        val snapshot = inputFusion.snapshot()
        rldsWriter.recordStep(
            observation = snapshot.toObservation(uiContextTracker.current()),
            action = Action(command = null, source = "human_dev_xr", uiAction = uiAction),
        )
    }
}

interface InputFusion {
    fun updateGlove(frame: GloveFrame)
    fun updateAudio(audio: Audio)
    fun updateGaze(gaze: Gaze)
    fun updateHeadsetPose(pose: Pose)
    fun updateRightHandPose(pose: Pose)
    fun updateLeftHandPose(pose: Pose?)
    fun updateUiContext(uiContext: UiContext)
    fun currentObservation(robotState: RobotState, timestampNanos: Long): FusedObservation
    fun snapshot(): FusedObservation
}

interface CommandMapper {
    fun map(fused: FusedObservation): ControlCommand
}

data class FusedObservation(
    val headsetPose: Pose = Pose(),
    val rightHandPose: Pose = Pose(),
    val leftHandPose: Pose? = null,
    val gaze: Gaze? = null,
    val gloveFrame: GloveFrame? = null,
    val audio: Audio? = null,
    val uiContext: UiContext? = null,
    val robotState: RobotState? = null,
) {
    fun toObservation(activeUiContext: UiContext? = null): Observation = Observation(
        headsetPose = headsetPose,
        rightHandPose = rightHandPose,
        leftHandPose = leftHandPose,
        gaze = gaze,
        gloveFrame = gloveFrame,
        audio = audio,
        uiContext = activeUiContext ?: uiContext,
        robotState = robotState,
    )
}

private class DefaultInputFusion : InputFusion {
    private var latestGlove: GloveFrame? = null
    private var latestGaze: Gaze? = null
    private var latestAudio: Audio? = null
    private var latestUiContext: UiContext? = null
    private var latestHeadsetPose: Pose = Pose()
    private var latestRightHandPose: Pose = Pose()
    private var latestLeftHandPose: Pose? = null
    private var latestRobotState: RobotState? = null

    override fun updateGlove(frame: GloveFrame) {
        latestGlove = frame
    }

    override fun updateAudio(audio: Audio) {
        latestAudio = audio
    }

    override fun updateGaze(gaze: Gaze) {
        latestGaze = gaze
    }

    override fun updateHeadsetPose(pose: Pose) {
        latestHeadsetPose = pose
    }

    override fun updateRightHandPose(pose: Pose) {
        latestRightHandPose = pose
    }

    override fun updateLeftHandPose(pose: Pose?) {
        latestLeftHandPose = pose
    }

    override fun updateUiContext(uiContext: UiContext) {
        latestUiContext = uiContext
    }

    override fun currentObservation(robotState: RobotState, timestampNanos: Long): FusedObservation {
        // TODO: Add headset/hand poses from XR runtime.
        latestRobotState = robotState
        return FusedObservation(
            gloveFrame = latestGlove,
            robotState = robotState,
            gaze = latestGaze,
            audio = latestAudio,
            uiContext = latestUiContext,
            headsetPose = latestHeadsetPose,
            rightHandPose = latestRightHandPose,
            leftHandPose = latestLeftHandPose,
        )
    }

    override fun snapshot(): FusedObservation = FusedObservation(
        gloveFrame = latestGlove,
        robotState = latestRobotState,
        gaze = latestGaze,
        audio = latestAudio,
        uiContext = latestUiContext,
        headsetPose = latestHeadsetPose,
        rightHandPose = latestRightHandPose,
        leftHandPose = latestLeftHandPose,
    )
}

private class DefaultCommandMapper : CommandMapper {
    override fun map(fused: FusedObservation): ControlCommand {
        // TODO: Map pose/gesture to normalized command vector expected by Robot API.
        return ControlCommand.EndEffectorVelocity(
            linearMps = Vector3(0f, 0f, 0f),
            angularRadS = Vector3(0f, 0f, 0f),
            referenceFrame = ReferenceFrame.BASE,
            targetFrequencyHz = 20.0,
        )
    }
}
