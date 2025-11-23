package com.continuonxr.app.connectivity

import com.continuonxr.app.config.ConnectivityConfig

class ContinuonBrainWebRtcClient(private val config: ConnectivityConfig) {
    fun connect() {
        // TODO: Implement WebRTC signaling using config.signalingUrl and config.iceServers.
    }
    fun sendCommand(command: List<Float>) {
        // TODO: Send command over data channel.
    }
    fun observeState(onState: (RobotState) -> Unit) {
        // TODO: Hook state stream from ContinuonBrain over WebRTC.
    }
    fun close() {
        // TODO: Tear down WebRTC session.
    }
}
