package com.continuonxr.app.connectivity

import kotlinx.serialization.Serializable

/**
 * Typed representation of control commands sent to ContinuonBrain/OS.
 */
@Serializable
sealed class ControlCommand {
    abstract val targetFrequencyHz: Double?
    abstract val safety: SafetyStatus?

    @Serializable
    data class EndEffectorVelocity(
        val linearMps: Vector3,
        val angularRadS: Vector3,
        val referenceFrame: ReferenceFrame,
        override val targetFrequencyHz: Double? = null,
        override val safety: SafetyStatus? = null,
    ) : ControlCommand()

    @Serializable
    data class JointDelta(
        val deltaRadians: List<Float>,
        override val targetFrequencyHz: Double? = null,
        override val safety: SafetyStatus? = null,
    ) : ControlCommand()

    @Serializable
    data class Gripper(
        val mode: GripperMode,
        val positionM: Float? = null,
        val velocityMps: Float? = null,
        override val targetFrequencyHz: Double? = null,
        override val safety: SafetyStatus? = null,
    ) : ControlCommand()
}

@Serializable
data class Vector3(
    val x: Float,
    val y: Float,
    val z: Float,
)

@Serializable
enum class ReferenceFrame { BASE, TOOL }

@Serializable
enum class GripperMode { POSITION, VELOCITY }

@Serializable
data class SafetyStatus(
    val estopReleasedAck: Boolean = false,
    val safetyToken: String? = null,
)
