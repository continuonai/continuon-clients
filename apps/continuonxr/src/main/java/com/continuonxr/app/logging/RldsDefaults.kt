package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig

data class EpisodeDefaults(
    val environmentId: String? = null,
    val robotId: String? = null,
    val robotModel: String? = null,
    val frameConvention: String? = null,
    val appVersion: String? = null,
    val brainVersion: String? = null,
    val gloveFirmwareVersion: String? = null,
    val tags: List<String> = emptyList(),
)

fun EpisodeMetadata.withDefaults(defaults: EpisodeDefaults?): EpisodeMetadata {
    if (defaults == null) return this
    return copy(
        environmentId = environmentId.ifBlank { defaults.environmentId ?: environmentId },
        robotId = robotId ?: defaults.robotId,
        robotModel = robotModel ?: defaults.robotModel,
        frameConvention = frameConvention ?: defaults.frameConvention,
        software = software ?: SoftwareInfo(
            xrAppVersion = defaults.appVersion,
            brainVersion = defaults.brainVersion,
            gloveFirmwareVersion = defaults.gloveFirmwareVersion,
        ),
        tags = (tags + defaults.tags).distinct(),
    )
}

fun EpisodeDefaults.fromLoggingConfig(loggingConfig: LoggingConfig): EpisodeDefaults =
    EpisodeDefaults(
        environmentId = loggingConfig.environmentId,
        robotId = loggingConfig.robotId,
        robotModel = loggingConfig.robotModel,
        frameConvention = loggingConfig.frameConvention,
        appVersion = loggingConfig.appVersion,
        brainVersion = loggingConfig.brainVersion,
        gloveFirmwareVersion = loggingConfig.gloveFirmwareVersion,
        tags = loggingConfig.tags,
    )

