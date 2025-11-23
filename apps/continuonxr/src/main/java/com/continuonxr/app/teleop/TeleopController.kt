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
import java.util.concurrent.TimeUnit
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
    private var running = false

    fun startTeleopSession() {
        if (running) return
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
        running = true
    }

    fun stopTeleopSession() {
        if (!running) return
        gloveBleClient.disconnect()
        audioCapture.stop()
        continuonBrainClient.close()
        rldsWriter.completeEpisode()
        running = false
    }

    private fun onGloveFrame(frame: GloveFrame) {
        inputFusion.updateGlove(frame)
    }

    private fun onAudio(audio: Audio) {
        inputFusion.updateAudio(audio)
    }

    fun onAudioUpdate(audio: Audio) = onAudio(audio)

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
        gloveFrame = gloveFrame ?: robotState?.let { emptyGlove(it.timestampNanos) },
        audio = audio,
        uiContext = activeUiContext ?: uiContext,
        robotState = robotState,
        videoFrameId = robotState?.frameId,
    )
}

private fun emptyGlove(timestampNanos: Long): GloveFrame = GloveFrame(
    timestampNanos = timestampNanos,
    flex = List(5) { 0f },
    fsr = List(8) { 0f },
    orientationQuat = listOf(0f, 0f, 0f, 1f),
    accel = listOf(0f, 0f, 0f),
    valid = false,
)

private class DefaultInputFusion : InputFusion {
    private val staleThresholdNs = TimeUnit.MILLISECONDS.toNanos(250)

    private var latestGlove: TimedValue<GloveFrame?> = TimedValue(null, 0L)
    private var latestGaze: TimedValue<Gaze?> = TimedValue(null, 0L)
    private var latestAudio: TimedValue<Audio?> = TimedValue(null, 0L)
    private var latestUiContext: TimedValue<UiContext?> = TimedValue(null, 0L)
    private var latestHeadsetPose: TimedValue<Pose> = TimedValue(Pose(), 0L)
    private var latestRightHandPose: TimedValue<Pose> = TimedValue(Pose(), 0L)
    private var latestLeftHandPose: TimedValue<Pose?> = TimedValue(null, 0L)
    private var latestRobotState: RobotState? = null

    override fun updateGlove(frame: GloveFrame) {
        latestGlove = TimedValue(frame, System.nanoTime())
    }

    override fun updateAudio(audio: Audio) {
        latestAudio = TimedValue(audio, System.nanoTime())
    }

    override fun updateGaze(gaze: Gaze) {
        latestGaze = TimedValue(gaze, System.nanoTime())
    }

    override fun updateHeadsetPose(pose: Pose) {
        latestHeadsetPose = TimedValue(pose, System.nanoTime())
    }

    override fun updateRightHandPose(pose: Pose) {
        latestRightHandPose = TimedValue(pose, System.nanoTime())
    }

    override fun updateLeftHandPose(pose: Pose?) {
        latestLeftHandPose = TimedValue(pose, System.nanoTime())
    }

    override fun updateUiContext(uiContext: UiContext) {
        latestUiContext = TimedValue(uiContext, System.nanoTime())
    }

    override fun currentObservation(robotState: RobotState, timestampNanos: Long): FusedObservation {
        latestRobotState = robotState
        val now = timestampNanos
        return FusedObservation(
            gloveFrame = latestGlove.valueIfFresh(now),
            robotState = robotState,
            gaze = latestGaze.valueIfFresh(now),
            audio = latestAudio.valueIfFresh(now),
            uiContext = latestUiContext.valueIfFresh(now),
            headsetPose = latestHeadsetPose.poseIfFresh(now),
            rightHandPose = latestRightHandPose.poseIfFresh(now),
            leftHandPose = latestLeftHandPose.poseIfFresh(now),
        )
    }

    override fun snapshot(): FusedObservation {
        val now = System.nanoTime()
        return FusedObservation(
            gloveFrame = latestGlove.valueIfFresh(now),
            robotState = latestRobotState,
            gaze = latestGaze.valueIfFresh(now),
            audio = latestAudio.valueIfFresh(now),
            uiContext = latestUiContext.valueIfFresh(now),
            headsetPose = latestHeadsetPose.poseIfFresh(now),
            rightHandPose = latestRightHandPose.poseIfFresh(now),
            leftHandPose = latestLeftHandPose.poseIfFresh(now),
        )
    }

    private fun <T> TimedValue<T>.valueIfFresh(now: Long): T? {
        if (timestampNanos == 0L) return value
        val isStale = now - timestampNanos > staleThresholdNs
        return if (isStale) null else value
    }

    private fun TimedValue<Pose>.poseIfFresh(now: Long): Pose {
        val base = value
        if (timestampNanos == 0L) return base
        val isStale = now - timestampNanos > staleThresholdNs
        return if (isStale) base.copy(valid = false) else base
    }

    private fun TimedValue<Pose?>.poseIfFresh(now: Long): Pose? {
        val pose = value ?: return null
        if (timestampNanos == 0L) return pose
        val isStale = now - timestampNanos > staleThresholdNs
        return if (isStale) pose.copy(valid = false) else pose
    }
}

private data class TimedValue<T>(val value: T, val timestampNanos: Long)

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
