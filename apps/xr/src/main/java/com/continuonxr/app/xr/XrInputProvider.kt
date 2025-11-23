package com.continuonxr.app.xr

import com.continuonxr.app.logging.Gaze
import com.continuonxr.app.logging.Pose
import com.continuonxr.app.teleop.TeleopController

/**
 * Stub XR input provider. Replace with Jetpack XR/SceneCore integrations to push head/hand poses and gaze.
 */
class XrInputProvider(
    private val teleopController: TeleopController,
) {
    // TODO: Inject SceneCore/Jetpack XR runtime once available.

    fun start() {
        // TODO: Hook into SceneCore pose/gaze/audio streams and call the handlers below.
    }

    fun stop() {
        // TODO: Unregister from XR runtime callbacks.
    }

    fun onHeadsetPose(pose: Pose) {
        teleopController.onHeadsetPose(pose)
    }

    fun onRightHandPose(pose: Pose) {
        teleopController.onRightHandPose(pose)
    }

    fun onLeftHandPose(pose: Pose) {
        teleopController.onLeftHandPose(pose)
    }

    fun onGaze(gaze: Gaze) {
        teleopController.onGazeUpdate(gaze)
    }
}
