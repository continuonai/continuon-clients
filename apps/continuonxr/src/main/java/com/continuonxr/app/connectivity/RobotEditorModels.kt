package com.continuonxr.app.connectivity

import continuonxr.continuonbrain.v1.ContinuonbrainLink
import kotlinx.serialization.Serializable

@Serializable
data class CapabilityManifest(
    val robotModel: String = "",
    val softwareVersions: RobotSoftwareVersions = RobotSoftwareVersions(),
    val safety: SafetyFeatures = SafetyFeatures(),
    val skills: List<Skill> = emptyList(),
    val sensors: List<Sensor> = emptyList(),
    val source: String = "",
    val availableCmsSnapshots: List<CmsSnapshot> = emptyList(),
    val safetySignals: List<SafetySignalDefinition> = emptyList(),
)

@Serializable
data class RobotSoftwareVersions(
    val runtime: String = "",
    val studio: String = "",
    val hopeCmsBundle: String = "",
    val gloveFirmware: String = "",
)

@Serializable
data class SafetyFeatures(
    val envelopesSupported: Boolean = false,
    val estopSupported: Boolean = false,
    val safetyHeadPresent: Boolean = false,
    val rateLimiterPresent: Boolean = false,
    val defaultEnvelopes: List<String> = emptyList(),
)

@Serializable
data class SkillParameter(
    val name: String = "",
    val description: String = "",
    val type: String = "",
    val defaultValue: String = "",
    val required: Boolean = false,
    val enumValues: List<String> = emptyList(),
)

@Serializable
data class Skill(
    val id: String = "",
    val name: String = "",
    val parameters: List<SkillParameter> = emptyList(),
    val requiredModalities: List<String> = emptyList(),
    val safetyTags: List<String> = emptyList(),
    val documentationUri: String = "",
)

@Serializable
data class Sensor(
    val id: String = "",
    val modality: String = "",
    val sampleRateHz: Double = 0.0,
    val latencyMs: Double = 0.0,
    val frameIdDomain: String = "",
    val calibrationStatus: String = "",
    val vendor: String = "",
)

@Serializable
data class CmsSnapshot(
    val snapshotId: String = "",
    val policyVersion: String = "",
    val memoryPlaneVersion: String = "",
    val cmsBalance: String = "",
    val createdAt: String = "",
    val source: String = "",
    val lastTrainingSummaryId: String = "",
)

@Serializable
data class SafetySignalDefinition(
    val id: String = "",
    val label: String = "",
    val description: String = "",
    val unit: String = "",
    val tags: List<String> = emptyList(),
)

@Serializable
data class SafetySignal(
    val id: String = "",
    val value: Double = 0.0,
    val severity: String = "",
    val source: String = "",
    val frameId: String = "",
    val label: String = "",
    val confidence: Double = 0.0,
)

@Serializable
data class RobotEditorTelemetry(
    val robotState: RobotState,
    val diagnostics: EditorDiagnostics = EditorDiagnostics(),
    val safetyState: SafetyState = SafetyState(),
    val hopeCmsSignals: HopeCmsSignals = HopeCmsSignals(),
    val safetySignals: List<SafetySignal> = emptyList(),
    val cmsSnapshot: CmsSnapshot? = null,
)

@Serializable
data class EditorDiagnostics(
    val latencyMs: Float = 0f,
    val packetLossRate: Float = 0f,
    val bleRssi: Int = 0,
    val mockMode: Boolean = false,
    val uplinkKbps: Float = 0f,
    val downlinkKbps: Float = 0f,
)

@Serializable
data class SafetyState(
    val estopEngaged: Boolean = false,
    val rateLimited: Boolean = false,
    val envelopeViolated: Boolean = false,
    val predictedCollisionHorizonS: Double = 0.0,
    val safetyHeadState: String = "",
    val activeEnvelopes: List<String> = emptyList(),
)

@Serializable
data class HopeCmsSignals(
    val fast: HopeFastSignals = HopeFastSignals(),
    val mid: HopeMidSignals = HopeMidSignals(),
    val slow: HopeSlowSignals = HopeSlowSignals(),
)

@Serializable
data class HopeFastSignals(
    val hazardScores: List<Float> = emptyList(),
    val hazardLabels: List<String> = emptyList(),
    val safetyHeadOverride: Float = 0f,
)

@Serializable
data class HopeMidSignals(
    val intentLabel: String = "",
    val intentConfidence: Float = 0f,
    val suggestedSkills: List<String> = emptyList(),
)

@Serializable
data class HopeSlowSignals(
    val policyVersion: String = "",
    val memoryPlaneVersion: String = "",
    val cmsBalance: String = "",
    val lastTrainingSummaryId: String = "",
)

internal fun ContinuonbrainLink.CapabilityManifest.toDomain(): CapabilityManifest =
    CapabilityManifest(
        robotModel = robotModel,
        softwareVersions = softwareVersions.toDomain(),
        safety = safety.toDomain(),
        skills = skillsList.map { it.toDomain() },
        sensors = sensorsList.map { it.toDomain() },
        source = source,
        availableCmsSnapshots = availableCmsSnapshotsList.map { it.toDomain() },
        safetySignals = safetySignalsList.map { it.toDomain() },
    )

internal fun ContinuonbrainLink.RobotSoftwareVersions.toDomain() =
    RobotSoftwareVersions(
        runtime = runtime,
        studio = studio,
        hopeCmsBundle = hopeCmsBundle,
        gloveFirmware = gloveFirmware,
    )

internal fun ContinuonbrainLink.SafetyFeatures.toDomain() =
    SafetyFeatures(
        envelopesSupported = envelopesSupported,
        estopSupported = estopSupported,
        safetyHeadPresent = safetyHeadPresent,
        rateLimiterPresent = rateLimiterPresent,
        defaultEnvelopes = defaultEnvelopesList,
    )

internal fun ContinuonbrainLink.SkillParameter.toDomain() =
    SkillParameter(
        name = name,
        description = description,
        type = type,
        defaultValue = defaultValue,
        required = required,
        enumValues = enumValuesList,
    )

internal fun ContinuonbrainLink.Skill.toDomain() =
    Skill(
        id = id,
        name = name,
        parameters = parametersList.map { it.toDomain() },
        requiredModalities = requiredModalitiesList,
        safetyTags = safetyTagsList,
        documentationUri = documentationUri,
    )

internal fun ContinuonbrainLink.Sensor.toDomain() =
    Sensor(
        id = id,
        modality = modality,
        sampleRateHz = sampleRateHz,
        latencyMs = latencyMs,
        frameIdDomain = frameIdDomain,
        calibrationStatus = calibrationStatus,
        vendor = vendor,
    )

internal fun ContinuonbrainLink.CmsSnapshot.toDomain() =
    CmsSnapshot(
        snapshotId = snapshotId,
        policyVersion = policyVersion,
        memoryPlaneVersion = memoryPlaneVersion,
        cmsBalance = cmsBalance,
        createdAt = createdAt,
        source = source,
        lastTrainingSummaryId = lastTrainingSummaryId,
    )

internal fun ContinuonbrainLink.SafetySignalDefinition.toDomain() =
    SafetySignalDefinition(
        id = id,
        label = label,
        description = description,
        unit = unit,
        tags = tagsList,
    )

internal fun ContinuonbrainLink.SafetySignal.toDomain() =
    SafetySignal(
        id = id,
        value = value,
        severity = severity,
        source = source,
        frameId = frameId,
        label = label,
        confidence = confidence,
    )

internal fun ContinuonbrainLink.StreamRobotEditorTelemetryResponse.toDomain(): RobotEditorTelemetry =
    RobotEditorTelemetry(
        robotState = robotState.toDomain(),
        diagnostics = if (hasDiagnostics()) diagnostics.toDomain() else EditorDiagnostics(),
        safetyState = if (hasSafetyState()) safetyState.toDomain() else SafetyState(),
        hopeCmsSignals = if (hasHopeCmsSignals()) hopeCmsSignals.toDomain() else HopeCmsSignals(),
        safetySignals = safetySignalsList.map { it.toDomain() },
        cmsSnapshot = if (hasCmsSnapshot()) cmsSnapshot.toDomain() else null,
    )

internal fun ContinuonbrainLink.EditorDiagnostics.toDomain() =
    EditorDiagnostics(
        latencyMs = latencyMs,
        packetLossRate = packetLossRate,
        bleRssi = bleRssi,
        mockMode = mockMode,
        uplinkKbps = uplinkKbps,
        downlinkKbps = downlinkKbps,
    )

internal fun ContinuonbrainLink.SafetyState.toDomain() =
    SafetyState(
        estopEngaged = estopEngaged,
        rateLimited = rateLimited,
        envelopeViolated = envelopeViolated,
        predictedCollisionHorizonS = predictedCollisionHorizonS,
        safetyHeadState = safetyHeadState,
        activeEnvelopes = activeEnvelopesList,
    )

internal fun ContinuonbrainLink.HopeCmsSignals.toDomain() =
    HopeCmsSignals(
        fast = if (hasFast()) fast.toDomain() else HopeFastSignals(),
        mid = if (hasMid()) mid.toDomain() else HopeMidSignals(),
        slow = if (hasSlow()) slow.toDomain() else HopeSlowSignals(),
    )

internal fun ContinuonbrainLink.HopeFastSignals.toDomain() =
    HopeFastSignals(
        hazardScores = hazardScoresList,
        hazardLabels = hazardLabelsList,
        safetyHeadOverride = safetyHeadOverride,
    )

internal fun ContinuonbrainLink.HopeMidSignals.toDomain() =
    HopeMidSignals(
        intentLabel = intentLabel,
        intentConfidence = intentConfidence,
        suggestedSkills = suggestedSkillsList,
    )

internal fun ContinuonbrainLink.HopeSlowSignals.toDomain() =
    HopeSlowSignals(
        policyVersion = policyVersion,
        memoryPlaneVersion = memoryPlaneVersion,
        cmsBalance = cmsBalance,
        lastTrainingSummaryId = lastTrainingSummaryId,
    )

internal fun ContinuonbrainLink.RobotState.toDomain(): RobotState =
    RobotState(
        timestampNanos = timestampNanos,
        jointPositions = jointPositionsList,
        endEffectorPose = Pose(
            position = endEffectorPose.positionList,
            orientationQuat = endEffectorPose.orientationQuatList,
        ),
        gripperOpen = gripperOpen,
        frameId = frameId,
        jointVelocities = jointVelocitiesList,
        jointEfforts = jointEffortsList,
        endEffectorTwist = endEffectorTwistList,
        wallTimeMillis = wallTimeMillis,
    )
