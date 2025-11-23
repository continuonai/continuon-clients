package com.continuonxr.app.xr

import com.continuonxr.app.logging.Gaze
import com.continuonxr.app.logging.Pose
import com.continuonxr.app.teleop.TeleopController

/**
 * Entry point for XR runtimes to forward pose/gaze/audio into teleop.
 */
class XrInputProvider(
    private val teleopController: TeleopController,
    private val sceneCoreInputManager: SceneCoreInputManager = SceneCoreInputManager(teleopController),
) {
    fun start(streams: SceneCoreStreams = SceneCoreStreams.stub()) {
        sceneCoreInputManager.start(streams)
    }

    fun stop() {
        sceneCoreInputManager.stop()
    }

    fun onHeadsetPose(pose: Pose) = teleopController.onHeadsetPose(pose)
    fun onRightHandPose(pose: Pose) = teleopController.onRightHandPose(pose)
    fun onLeftHandPose(pose: Pose?) = teleopController.onLeftHandPose(pose)
    fun onGaze(gaze: Gaze) = teleopController.onGazeUpdate(gaze)
}
