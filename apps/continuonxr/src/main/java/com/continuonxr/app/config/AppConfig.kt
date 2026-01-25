package com.continuonxr.app.config

import android.content.Context
import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * Simple config holder for early bootstrapping.
 * In Phase 1 this can be hydrated from a JSON file or feature flag service without rebuilds.
 */
@Serializable
data class AppConfig(
    val mode: Mode,
    val connectivity: ConnectivityConfig,
    val logging: LoggingConfig,
    val glove: GloveConfig,
)

@Serializable
enum class Mode { TRAINER, WORKSTATION, OBSERVER }

@Serializable
data class ConnectivityConfig(
    val continuonBrainHost: String,
    val continuonBrainPort: Int,
    val useWebRtc: Boolean,
    val cloudBaseUrl: String,
    val useTls: Boolean = false,
    val authToken: String? = null,
    val signalingUrl: String? = null,
    val iceServers: List<String> = emptyList(),
)

@Serializable
data class LoggingConfig(
    val episodeOutputDir: String,
    val uploadOnComplete: Boolean,
    val writeEpisodesToDisk: Boolean = true,
    val validateRlds: Boolean = true,
    val failOnValidationError: Boolean = true,
    val uploadEndpoint: String? = null,
    val uploadAuthToken: String? = null,
    val maxUploadRetries: Int = 3,
    val uploadRetryBackoffMs: Long = 1_000,
    val environmentId: String? = null,
    val robotId: String? = null,
    val robotModel: String? = null,
    val frameConvention: String? = null,
    val appVersion: String? = null,
    val brainVersion: String? = null,
    val gloveFirmwareVersion: String? = null,
    val tags: List<String> = emptyList(),
)

@Serializable
data class GloveConfig(
    val bleDeviceName: String,
    val minMtu: Int = 64,
    val targetSampleRateHz: Int = 100,
    val serviceUuid: String = "7d0e1000-5c86-4c84-9c72-6fa4cbb8a9c5",
    val characteristicUuid: String = "7d0e1001-5c86-4c84-9c72-6fa4cbb8a9c5",
)

/**
 * Loads AppConfig from multiple sources with priority:
 * 1. /sdcard/continuonxr/config.json (external override)
 * 2. app-internal files dir config.json
 * 3. Built-in defaults
 */
object AppConfigLoader {
    private const val TAG = "AppConfigLoader"
    private const val CONFIG_FILENAME = "config.json"

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        prettyPrint = true
    }

    /**
     * Load config with context for file access.
     */
    fun load(context: Context): AppConfig {
        // Try external config first (for easy development/testing)
        val externalConfig = tryLoadFromPath("/sdcard/continuonxr/$CONFIG_FILENAME")
        if (externalConfig != null) {
            Log.i(TAG, "Loaded config from external storage")
            return externalConfig
        }

        // Try app-internal config
        val internalFile = File(context.filesDir, CONFIG_FILENAME)
        if (internalFile.exists()) {
            val config = tryLoadFromFile(internalFile)
            if (config != null) {
                Log.i(TAG, "Loaded config from internal storage")
                return config
            }
        }

        // Fall back to defaults
        Log.i(TAG, "Using default config")
        return createDefaultConfig()
    }

    /**
     * Load config without context (uses defaults).
     */
    fun load(): AppConfig {
        // Try external config
        val externalConfig = tryLoadFromPath("/sdcard/continuonxr/$CONFIG_FILENAME")
        if (externalConfig != null) {
            Log.i(TAG, "Loaded config from external storage")
            return externalConfig
        }

        Log.i(TAG, "Using default config")
        return createDefaultConfig()
    }

    /**
     * Save config to internal storage.
     */
    fun save(context: Context, config: AppConfig) {
        try {
            val file = File(context.filesDir, CONFIG_FILENAME)
            val jsonString = json.encodeToString(AppConfig.serializer(), config)
            file.writeText(jsonString)
            Log.i(TAG, "Saved config to ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save config", e)
        }
    }

    private fun tryLoadFromPath(path: String): AppConfig? {
        return try {
            val file = File(path)
            if (file.exists()) {
                tryLoadFromFile(file)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load config from $path", e)
            null
        }
    }

    private fun tryLoadFromFile(file: File): AppConfig? {
        return try {
            val jsonString = file.readText()
            json.decodeFromString<AppConfig>(jsonString)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse config from ${file.absolutePath}", e)
            null
        }
    }

    private fun createDefaultConfig(): AppConfig {
        return AppConfig(
            mode = Mode.TRAINER,
            connectivity = ConnectivityConfig(
                continuonBrainHost = "127.0.0.1",
                continuonBrainPort = 50051,
                useWebRtc = false,
                cloudBaseUrl = "https://api.continuonai.com",
                signalingUrl = null,
                iceServers = emptyList(),
            ),
            logging = LoggingConfig(
                episodeOutputDir = "/sdcard/continuonxr/episodes",
                uploadOnComplete = false,
                environmentId = "dev-lab",
                frameConvention = "base_link",
            ),
            glove = GloveConfig(
                bleDeviceName = "ContinuonGlove",
            ),
        )
    }
}
