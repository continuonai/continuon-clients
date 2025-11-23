package com.continuonxr.app.xr

import com.continuonxr.app.logging.Gaze
import com.continuonxr.app.logging.Pose
import com.continuonxr.app.teleop.TeleopController

/**
 * Placeholder for Jetpack XR / SceneCore integration.
 * Hook head/hand poses, gaze, and audio events into TeleopController.
 */
class SceneCoreInputManager(
    private val teleopController: TeleopController,
) {
    fun start() {
        // TODO: Register with SceneCore pipelines and subscribe to pose/gaze/audio streams.
    }

    fun stop() {
        // TODO: Unregister listeners and release resources.
    }

    // Callbacks to be wired from SceneCore
    fun onHeadPose(pose: Pose) = teleopController.onHeadsetPose(pose)
    fun onRightHandPose(pose: Pose) = teleopController.onRightHandPose(pose)
    fun onLeftHandPose(pose: Pose?) = teleopController.onLeftHandPose(pose)
    fun onGaze(gaze: Gaze) = teleopController.onGazeUpdate(gaze)
}

