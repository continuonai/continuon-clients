package com.continuonxr.app.logging

/**
 * Lightweight validator to enforce required RLDS fields before persistence.
 */
class RldsValidator {
    fun validateEpisodeMetadata(metadata: EpisodeMetadata): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()
        if (metadata.xrMode.isBlank()) issues += ValidationIssue.error("xrMode is required")
        if (metadata.controlRole.isBlank()) issues += ValidationIssue.error("controlRole is required")
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
        }
        observation.gloveFrame?.let { glove ->
            if (glove.flex.size != 5) issues += ValidationIssue.error("glove.flex must have 5 elements")
            if (glove.fsr.size != 8) issues += ValidationIssue.error("glove.fsr must have 8 elements")
            if (glove.orientationQuat.size != 4) issues += ValidationIssue.error("glove.orientationQuat must have 4 elements")
            if (glove.accel.size != 3) issues += ValidationIssue.error("glove.accel must have 3 elements")
        }
        observation.audio?.let { audio ->
            if (audio.sampleRateHz <= 0) issues += ValidationIssue.error("audio.sampleRateHz must be > 0")
            if (audio.numChannels <= 0) issues += ValidationIssue.error("audio.numChannels must be > 0")
        }
        return issues
    }

    private fun validateAction(action: Action): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()
        if (action.command.isEmpty()) issues += ValidationIssue.error("action.command must not be empty")
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
