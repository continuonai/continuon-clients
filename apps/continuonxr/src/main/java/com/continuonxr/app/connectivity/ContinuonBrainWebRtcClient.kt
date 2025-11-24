package com.continuonxr.app.connectivity

import android.util.Log
import com.continuonxr.app.config.ConnectivityConfig
import continuonxr.continuonbrain.v1.ContinuonbrainLink
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.decodeBase64
import okio.ByteString.Companion.encodeUtf8

/**
 * Lightweight WebRTC/WebSocket transport for ContinuonBrain Link proto frames.
 * This client expects the signaling endpoint to accept binary proto messages over WebSocket.
 */
class ContinuonBrainWebRtcClient(
    private val config: ConnectivityConfig,
    private val clientId: String,
    private val okHttpClient: OkHttpClient = OkHttpClient(),
) : WebSocketListener() {
    private var webSocket: WebSocket? = null
    private var stateCallback: ((RobotState) -> Unit)? = null

    fun connect(onState: ((RobotState) -> Unit)? = null) {
        onState?.let { stateCallback = it }
        val signalingUrl = config.signalingUrl
            ?: throw IllegalStateException("WebRTC path selected but signalingUrl is null")
        val request = Request.Builder().url(signalingUrl).build()
        webSocket = okHttpClient.newWebSocket(request, this)
    }

    fun sendCommand(command: ControlCommand) {
        val buffer = ByteString.of(*command.toProto(clientId).toByteArray())
        if (webSocket == null) {
            Log.w(TAG, "sendCommand called before WebRTC socket connected; dropping")
            return
        }
        webSocket?.send(buffer)
    }

    fun observeState(onState: (RobotState) -> Unit) {
        stateCallback = onState
    }

    fun close() {
        webSocket?.close(1000, "shutdown")
        okHttpClient.connectionPool.evictAll()
    }

    override fun onOpen(webSocket: WebSocket, response: Response) {
        super.onOpen(webSocket, response)
        val subscription = ContinuonbrainLink.StreamRobotStateRequest.newBuilder().setClientId(clientId).build()
        webSocket.send(ByteString.of(*subscription.toByteArray()))
    }

    override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
        handleIncoming(bytes)
    }

    override fun onMessage(webSocket: WebSocket, text: String) {
        val decoded = text.decodeBase64() ?: text.encodeUtf8()
        handleIncoming(decoded)
    }

    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
        Log.e(TAG, "WebRTC socket failure", t)
        super.onFailure(webSocket, t, response)
    }

    override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
        super.onClosed(webSocket, code, reason)
        Log.i(TAG, "WebRTC socket closed ($code): $reason")
    }

    private fun handleIncoming(bytes: ByteString) {
        runCatching {
            ContinuonbrainLink.StreamRobotStateResponse.parseFrom(bytes.toByteArray())
        }.onSuccess { response ->
            stateCallback?.invoke(response.toDomain())
        }.onFailure {
            Log.w(TAG, "Failed to decode robot state envelope", it)
        }
    }

    companion object {
        private const val TAG = "CBWebRtcClient"
    }
}
