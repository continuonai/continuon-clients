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
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.continuonxr.app.config.AppConfigLoader
import com.continuonxr.app.config.Mode
import com.continuonxr.app.connectivity.ContinuonBrainClient
import com.continuonxr.app.glove.GloveBleClient
import com.continuonxr.app.logging.RldsEpisodeWriter
import com.continuonxr.app.permissions.PermissionManager
import com.continuonxr.app.teleop.TeleopController
import com.continuonxr.app.xr.XrInputProvider

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
                                        if (PermissionManager.hasAllPermissions(this@MainActivity)) {
                                            if (started.value) {
                                                teleopController.stopTeleopSession()
                                                started.value = false
                                            } else {
                                                teleopController.startTeleopSession()
                                                xrInputProvider.start()
                                                started.value = true
                                            }
                                        } else {
                                            PermissionManager.requestMissingPermissions(this@MainActivity)
                                        }
                                    },
                                ) { Text(if (started.value) "Stop Teleop Session" else "Start Teleop Session") }
                            }
                            Mode.WORKSTATION -> {
                                Button(
                                    onClick = {
                                        if (started.value) {
                                            stopWorkstationShell()
                                            started.value = false
                                        } else {
                                            startWorkstationShell()
                                            started.value = true
                                        }
                                    },
                                ) { Text(if (started.value) "Stop Workstation Streams" else "Start Workstation Streams") }
                            }
                            Mode.OBSERVER -> {
                                Button(
                                    onClick = {
                                        if (started.value) {
                                            stopObserverShell()
                                            started.value = false
                                        } else {
                                            startObserverShell()
                                            started.value = true
                                        }
                                    },
                                ) { Text(if (started.value) "Stop Observer Streams" else "Start Observer Streams") }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun startWorkstationShell() {
        continuonBrainClient.connect()
        gloveBleClient.connect(onFrame = { /* Workstation glove warmup */ }, onDiagnostics = { })
    }

    private fun stopWorkstationShell() {
        continuonBrainClient.close()
        gloveBleClient.disconnect()
    }

    private fun startObserverShell() {
        continuonBrainClient.connect()
        continuonBrainClient.observeState { /* passive state stream for overlays */ }
    }

    private fun stopObserverShell() {
        continuonBrainClient.close()
    }
}
