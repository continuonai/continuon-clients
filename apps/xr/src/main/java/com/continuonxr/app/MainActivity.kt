package com.continuonxr.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.continuonxr.app.config.AppConfigLoader
import com.continuonxr.app.config.Mode
import com.continuonxr.app.connectivity.ContinuonBrainClient
import com.continuonxr.app.glove.GloveBleClient
import com.continuonxr.app.logging.RldsEpisodeWriter
import com.continuonxr.app.teleop.TeleopController
import com.continuonxr.app.xr.XrInputProvider

private val requiredPermissions = listOf(
    android.Manifest.permission.BLUETOOTH,
    android.Manifest.permission.BLUETOOTH_ADMIN,
    android.Manifest.permission.BLUETOOTH_CONNECT,
    android.Manifest.permission.BLUETOOTH_SCAN,
    android.Manifest.permission.ACCESS_FINE_LOCATION,
    android.Manifest.permission.ACCESS_COARSE_LOCATION,
    android.Manifest.permission.RECORD_AUDIO,
    android.Manifest.permission.INTERNET,
)

/**
 * Entry point for the ContinuonXR Android app.
 * This is a minimal stub to show module wiring; UI will be Jetpack XR/Compose in implementation.
 */
class MainActivity : ComponentActivity() {
    private val config = AppConfigLoader.load()
    private val continuonBrainClient = ContinuonBrainClient(config.connectivity)
    private val rldsWriter = RldsEpisodeWriter(config.logging)
    private val gloveBleClient by lazy { GloveBleClient(this, config.glove) }
    private val teleopController by lazy { TeleopController(continuonBrainClient, rldsWriter, gloveBleClient) }
    private val xrInputProvider by lazy { XrInputProvider(teleopController) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    val started = remember { mutableStateOf(false) }
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("ContinuonXR (mode=${config.mode})")
                        Spacer(modifier = Modifier.height(12.dp))
                        when (config.mode) {
                            Mode.TRAINER -> {
                                Button(
                                    onClick = {
                                        if (hasAllPermissions()) {
                                            teleopController.startTeleopSession()
                                            xrInputProvider.start()
                                            started.value = true
                                        } else {
                                            requestMissingPermissions()
                                        }
                                    },
                                    enabled = !started.value,
                                ) { Text("Start Teleop Session") }
                            }
                            Mode.WORKSTATION -> startWorkstationShell()
                            Mode.OBSERVER -> startObserverShell()
                        }
                    }
                }
            }
        }
    }

    private fun startWorkstationShell() {
        // TODO: Launch workstation panels and attach RLDS UI event logging.
    }

    private fun startObserverShell() {
        // TODO: Render safety overlays and annotation tools.
    }

    private fun hasAllPermissions(): Boolean =
        requiredPermissions.all { perm ->
            ContextCompat.checkSelfPermission(this, perm) == android.content.pm.PackageManager.PERMISSION_GRANTED
        }

    private fun requestMissingPermissions() {
        ActivityCompat.requestPermissions(
            this,
            requiredPermissions.toTypedArray(),
            1001,
        )
    }
}
