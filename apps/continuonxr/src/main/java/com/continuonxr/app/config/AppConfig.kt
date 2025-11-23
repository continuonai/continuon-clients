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
    val continuonBrainHost: String,
    val continuonBrainPort: Int,
    val useWebRtc: Boolean,
    val cloudBaseUrl: String,
    val useMockContinuonBrain: Boolean = true,
    val useTls: Boolean = false,
    val authToken: String? = null,
    val signalingUrl: String? = null,
    val iceServers: List<String> = emptyList(),
)

data class LoggingConfig(
    val episodeOutputDir: String,
    val uploadOnComplete: Boolean,
    val validateRlds: Boolean = true,
    val failOnValidationError: Boolean = true,
    val uploadEndpoint: String? = null,
    val uploadAuthToken: String? = null,
)

data class GloveConfig(
    val bleDeviceName: String,
    val minMtu: Int = 64,
    val targetSampleRateHz: Int = 100,
    val serviceUuid: String = "0000feed-0000-1000-8000-00805f9b34fb",
    val characteristicUuid: String = "0000beef-0000-1000-8000-00805f9b34fb",
)

object AppConfigLoader {
    fun load(): AppConfig {
        // TODO: Load from disk/flags; using safe defaults for now.
        return AppConfig(
            mode = Mode.TRAINER,
            connectivity = ConnectivityConfig(
                continuonBrainHost = "127.0.0.1",
                continuonBrainPort = 50051,
                useWebRtc = false,
                cloudBaseUrl = "https://api.continuon.ai",
                signalingUrl = null,
                iceServers = emptyList(),
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
