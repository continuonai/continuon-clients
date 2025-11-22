package com.continuonxr.app.config

/**
 * Simple config holder for early bootstrapping.
 * In Phase 1 this can be hydrated from a JSON file or feature flag service without rebuilds.
 */
data class AppConfig(
    val mode: Mode,
    val connectivity: ConnectivityConfig,
    val logging: LoggingConfig,
    val glove: GloveConfig,
)

enum class Mode { TRAINER, WORKSTATION, OBSERVER }

data class ConnectivityConfig(
    val pixelBrainHost: String,
    val pixelBrainPort: Int,
    val useWebRtc: Boolean,
    val cloudBaseUrl: String,
)

data class LoggingConfig(
    val episodeOutputDir: String,
    val uploadOnComplete: Boolean,
)

data class GloveConfig(
    val bleDeviceName: String,
    val minMtu: Int = 64,
    val targetSampleRateHz: Int = 100,
)

object AppConfigLoader {
    fun load(): AppConfig {
        // TODO: Load from disk/flags; using safe defaults for now.
        return AppConfig(
            mode = Mode.TRAINER,
            connectivity = ConnectivityConfig(
                pixelBrainHost = "127.0.0.1",
                pixelBrainPort = 50051,
                useWebRtc = false,
                cloudBaseUrl = "https://api.continuon.ai",
            ),
            logging = LoggingConfig(
                episodeOutputDir = "/sdcard/continuonxr/episodes",
                uploadOnComplete = false,
            ),
            glove = GloveConfig(
                bleDeviceName = "ContinuonGlove",
            ),
        )
    }
}

