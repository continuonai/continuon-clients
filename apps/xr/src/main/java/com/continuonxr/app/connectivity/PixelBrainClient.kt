package com.continuonxr.app.connectivity

import com.continuonxr.app.config.ConnectivityConfig
import kotlinx.serialization.Serializable

/**
 * Stub for the PixelBrain/OS bridge client.
 * Responsible for opening gRPC/WebRTC streams for robot state and commands.
 */
class PixelBrainClient(private val config: ConnectivityConfig) {
    fun connect() {
        // TODO: Implement gRPC/WebRTC negotiation to PixelBrain/OS.
    }

    fun sendCommand(command: FloatArray) {
        // TODO: Marshal normalized command vector to Robot API.
    }

    fun observeState(onState: (RobotState) -> Unit) {
        // TODO: Subscribe to robot state stream.
    }
}

@Serializable
data class RobotState(
    val timestampNanos: Long,
    val jointPositions: FloatArray = floatArrayOf(),
    val endEffectorPose: Pose = Pose(),
    val gripperOpen: Boolean = false,
)

@Serializable
data class Pose(
    val position: FloatArray = floatArrayOf(0f, 0f, 0f),
    val orientationQuat: FloatArray = floatArrayOf(0f, 0f, 0f, 1f),
)
