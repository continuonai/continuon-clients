package com.continuonxr.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import com.continuonxr.app.config.Mode
import com.continuonxr.app.connectivity.PixelBrainClient
import com.continuonxr.app.glove.GloveBleClient
import com.continuonxr.app.logging.RldsEpisodeWriter
import com.continuonxr.app.teleop.TeleopController

/**
 * Entry point for the ContinuonXR Android app.
 * This is a minimal stub to show module wiring; UI will be Jetpack XR/Compose in implementation.
 */
class MainActivity : ComponentActivity() {
    private val config = AppConfigLoader.load()
    private val pixelBrainClient = PixelBrainClient(config.connectivity)
    private val rldsWriter = RldsEpisodeWriter(config.logging)
    private val gloveBleClient = GloveBleClient(config.glove)
    private val teleopController = TeleopController(pixelBrainClient, rldsWriter, gloveBleClient)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // TODO: Initialize XR runtime and render shell once Jetpack XR is wired.
        when (config.mode) {
            Mode.TRAINER -> teleopController.startTeleopSession()
            Mode.WORKSTATION -> startWorkstationShell()
            Mode.OBSERVER -> startObserverShell()
        }
    }

    private fun startWorkstationShell() {
        // TODO: Launch workstation panels and attach RLDS UI event logging.
    }

    private fun startObserverShell() {
        // TODO: Render safety overlays and annotation tools.
    }
}

