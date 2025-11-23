package com.continuonxr.app.logging

import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.GripperMode

/**
 * Lightweight validator to enforce required RLDS fields before persistence.
 */
class RldsValidator {
    fun validateEpisodeMetadata(metadata: EpisodeMetadata): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()
        if (metadata.xrMode.isBlank()) issues += ValidationIssue.error("xr_mode is required")
        if (metadata.controlRole.isBlank()) issues += ValidationIssue.error("control_role is required")
        if (!ALLOWED_XR_MODES.contains(metadata.xrMode)) {
            issues += ValidationIssue.error("xr_mode must be one of ${ALLOWED_XR_MODES.joinToString()}")
        }
        if (!ALLOWED_CONTROL_ROLES.contains(metadata.controlRole)) {
            issues += ValidationIssue.error("control_role must be one of ${ALLOWED_CONTROL_ROLES.joinToString()}")
        }
        if (metadata.environmentId.isBlank()) issues += ValidationIssue.error("environment_id is required")
        return issues
    }

    fun validateStep(step: Step): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()
        issues += validateObservation(step.observation)
        issues += validateAction(step.action)
        return issues
    }

    private fun validateObservation(observation: Observation): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()
        if (observation.robotState == null) {
            issues += ValidationIssue.error("robotState is required for synchronization")
        }
        if (observation.headsetPose.position.size != 3) issues += ValidationIssue.error("headsetPose.position must have 3 elements")
        if (observation.headsetPose.orientationQuat.size != 4) issues += ValidationIssue.error("headsetPose.orientationQuat must have 4 elements")
        if (observation.rightHandPose.position.size != 3) issues += ValidationIssue.error("rightHandPose.position must have 3 elements")
        if (observation.rightHandPose.orientationQuat.size != 4) issues += ValidationIssue.error("rightHandPose.orientationQuat must have 4 elements")
        observation.leftHandPose?.let {
            if (it.position.size != 3) issues += ValidationIssue.error("leftHandPose.position must have 3 elements")
            if (it.orientationQuat.size != 4) issues += ValidationIssue.error("leftHandPose.orientationQuat must have 4 elements")
        }
        observation.gaze?.let {
            if (it.origin.size != 3) issues += ValidationIssue.error("gaze.origin must have 3 elements")
            if (it.direction.size != 3) issues += ValidationIssue.error("gaze.direction must have 3 elements")
            val magnitude = kotlin.math.sqrt(it.direction.map { value -> value * value }.sum())
            if (magnitude == 0f) {
                issues += ValidationIssue.error("gaze.direction must be normalized (non-zero)")
            } else {
                val normalized = kotlin.math.abs(magnitude - 1f) < 1e-3
                if (!normalized) issues += ValidationIssue.error("gaze.direction must be normalized to unit length")
            }
        }
        observation.robotState?.let { robot ->
            if (robot.wallTimeMillis == null) {
                issues += ValidationIssue.error("robotState.wallTimeMillis is required for alignment")
            }
            if (robot.frameId.isNullOrBlank()) {
                issues += ValidationIssue.error("robotState.frameId is required for alignment")
            }
            if (observation.videoFrameId.isNullOrBlank()) {
                issues += ValidationIssue.error("videoFrameId is required when robotState is present")
            }
            observation.videoFrameId?.let { videoFrameId ->
                if (robot.frameId != null && robot.frameId != videoFrameId) {
                    issues += ValidationIssue.error("robotState.frameId must match videoFrameId when both are set")
                }
            }
            observation.depthFrameId?.let { depthFrameId ->
                if (robot.frameId != null && robot.frameId != depthFrameId) {
                    issues += ValidationIssue.error("robotState.frameId must match depthFrameId when both are set")
                }
            }
        }
        observation.gloveFrame?.let { glove ->
            if (glove.flex.size != 5) issues += ValidationIssue.error("glove.flex must have 5 elements")
            if (glove.fsr.size != 8) issues += ValidationIssue.error("glove.fsr must have 8 elements")
            if (glove.orientationQuat.size != 4) issues += ValidationIssue.error("glove.orientationQuat must have 4 elements")
            if (glove.accel.size != 3) issues += ValidationIssue.error("glove.accel must have 3 elements")
        } ?: issues.add(ValidationIssue.error("gloveFrame is required by PRD 3.2"))
        observation.audio?.let { audio ->
            if (audio.sampleRateHz <= 0) issues += ValidationIssue.error("audio.sampleRateHz must be > 0")
            if (audio.numChannels <= 0) issues += ValidationIssue.error("audio.numChannels must be > 0")
        }
        if (observation.robotState != null && observation.gloveFrame != null) {
            val delta = kotlin.math.abs(
                observation.robotState.timestampNanos - observation.gloveFrame.timestampNanos,
            )
            if (delta > ALIGNMENT_TOLERANCE_NANOS) {
                issues += ValidationIssue.error("robotState and gloveFrame must align within 5 ms (delta=${'$'}delta ns)")
            }
        }
        return issues
    }

    private fun validateAction(action: Action): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()
        if (action.command == null && action.uiAction == null) {
            issues += ValidationIssue.error("action.command must be provided unless uiAction is set")
        }
        action.command?.let { command ->
            when (command) {
                is ControlCommand.EndEffectorVelocity -> {
                    // No additional validation beyond presence.
                }
                is ControlCommand.JointDelta -> {
                    if (command.deltaRadians.isEmpty()) issues += ValidationIssue.error("jointDelta.deltaRadians must not be empty")
                }
                is ControlCommand.Gripper -> {
                    when (command.mode) {
                        GripperMode.POSITION -> if (command.positionM == null) issues += ValidationIssue.error("gripper.positionM is required for POSITION mode")
                        GripperMode.VELOCITY -> if (command.velocityMps == null) issues += ValidationIssue.error("gripper.velocityMps is required for VELOCITY mode")
                    }
                }
            }
        }
        if (action.source.isBlank()) issues += ValidationIssue.error("action.source must not be blank")
        return issues
    }
}

data class ValidationIssue(
    val severity: Severity,
    val message: String,
) {
    enum class Severity { ERROR, WARNING }

    companion object {
        fun error(msg: String) = ValidationIssue(Severity.ERROR, msg)
        fun warn(msg: String) = ValidationIssue(Severity.WARNING, msg)
    }
}

private const val ALIGNMENT_TOLERANCE_NANOS = 5_000_000L
private val ALLOWED_XR_MODES = setOf("trainer", "workstation", "observer")
private val ALLOWED_CONTROL_ROLES = setOf("human_teleop", "human_supervisor", "human_dev_xr")
