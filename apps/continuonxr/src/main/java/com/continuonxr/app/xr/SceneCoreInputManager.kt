package com.continuonxr.app.xr

import com.continuonxr.app.logging.Audio
import com.continuonxr.app.logging.Gaze
import com.continuonxr.app.logging.Pose
import com.continuonxr.app.teleop.TeleopController
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.delay

/**
 * Bridges SceneCore/Jetpack XR pose, gaze, and audio feeds into the teleop controller.
 * Consumers provide `SceneCoreStreams` constructed from the XR runtime (pose providers, gaze rays, mic frames).
 */
class SceneCoreInputManager(
    private val teleopController: TeleopController,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Main),
) {
    private val jobs = mutableListOf<Job>()

    fun start(streams: SceneCoreStreams) {
        stop()
        jobs += scope.launch { streams.headsetPoses.collect { teleopController.onHeadsetPose(it) } }
        jobs += scope.launch { streams.rightHandPoses.collect { teleopController.onRightHandPose(it) } }
        jobs += scope.launch { streams.leftHandPoses.collect { teleopController.onLeftHandPose(it) } }
        jobs += scope.launch { streams.gaze.collect { teleopController.onGazeUpdate(it) } }
        streams.audio?.let { audioFlow ->
            jobs += scope.launch { audioFlow.collect { teleopController.onAudioUpdate(it) } }
        }
    }

    fun stop() {
        jobs.forEach { it.cancel() }
        jobs.clear()
    }
}

data class SceneCoreStreams(
    val headsetPoses: Flow<Pose>,
    val rightHandPoses: Flow<Pose>,
    val leftHandPoses: Flow<Pose?>,
    val gaze: Flow<Gaze>,
    val audio: Flow<Audio>? = null,
) {
    companion object {
        fun stub(sampleRateHz: Int = 60): SceneCoreStreams {
            val periodMs = 1000L / sampleRateHz
            return SceneCoreStreams(
                headsetPoses = repeatingPoseFlow(periodMs),
                rightHandPoses = repeatingPoseFlow(periodMs),
                leftHandPoses = repeatingOptionalPoseFlow(periodMs),
                gaze = flow {
                    while (true) {
                        emit(Gaze(origin = listOf(0f, 0f, 0f), direction = listOf(0f, 0f, -1f), confidence = 0.5f))
                        delay(periodMs)
                    }
                },
                audio = null,
            )
        }

        private fun repeatingPoseFlow(periodMs: Long): Flow<Pose> = flow {
            while (true) {
                emit(Pose(position = listOf(0f, 0f, 0.5f), orientationQuat = listOf(0f, 0f, 0f, 1f)))
                delay(periodMs)
            }
        }

        private fun repeatingOptionalPoseFlow(periodMs: Long): Flow<Pose?> = flow {
            var tick = 0
            while (true) {
                emit(
                    if (tick % 120 == 0) {
                        null
                    } else {
                        Pose(position = listOf(0f, 0f, 0.5f), orientationQuat = listOf(0f, 0f, 0f, 1f))
                    }
                )
                tick++
                delay(periodMs)
            }
        }
    }
}
