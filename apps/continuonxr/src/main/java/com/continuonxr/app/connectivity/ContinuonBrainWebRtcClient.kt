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
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine

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
    private var editorTelemetryCallback: ((RobotEditorTelemetry) -> Unit)? = null
    private val pendingManifestCallbacks = mutableListOf<(CapabilityManifest) -> Unit>()
    private var stateSubscribed = false
    private var telemetrySubscribed = false

    fun connect(onState: ((RobotState) -> Unit)? = null, onTelemetry: ((RobotEditorTelemetry) -> Unit)? = null) {
        onState?.let { stateCallback = it }
        onTelemetry?.let { editorTelemetryCallback = it }
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
        subscribeToStateStream()
    }

    fun observeEditorTelemetry(onTelemetry: (RobotEditorTelemetry) -> Unit) {
        editorTelemetryCallback = onTelemetry
        subscribeToTelemetryStream()
    }

    suspend fun requestCapabilityManifest(): CapabilityManifest? = suspendCancellableCoroutine { continuation ->
        val socket = webSocket
        if (socket == null) {
            continuation.resume(null)
            return@suspendCancellableCoroutine
        }

        val callback: (CapabilityManifest) -> Unit = { manifest ->
            if (continuation.isActive) continuation.resume(manifest)
        }
        pendingManifestCallbacks += callback
        continuation.invokeOnCancellation { pendingManifestCallbacks.remove(callback) }
        val request = ContinuonbrainLink.GetCapabilityManifestRequest.newBuilder()
            .setClientId(clientId)
            .setIncludeMockCapabilities(true)
            .build()
        socket.send(ByteString.of(*request.toByteArray()))
    }

    fun close() {
        webSocket?.close(1000, "shutdown")
        okHttpClient.connectionPool.evictAll()
        stateSubscribed = false
        telemetrySubscribed = false
    }

    override fun onOpen(webSocket: WebSocket, response: Response) {
        super.onOpen(webSocket, response)
        stateSubscribed = false
        telemetrySubscribed = false
        subscribeToStateStream()
        subscribeToTelemetryStream()
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
        val payload = bytes.toByteArray()

        if (telemetrySubscribed && editorTelemetryCallback != null) {
            runCatching { ContinuonbrainLink.StreamRobotEditorTelemetryResponse.parseFrom(payload) }
                .onSuccess { response ->
                    if (response.hasDiagnostics() || response.hasSafetyState() || response.hasHopeCmsSignals()) {
                        editorTelemetryCallback?.invoke(response.toDomain())
                        return
                    }
                }
        }

        runCatching { ContinuonbrainLink.GetCapabilityManifestResponse.parseFrom(payload) }
            .onSuccess { response ->
                if (response.hasManifest()) {
                    deliverManifest(response.manifest.toDomain())
                    return
                }
            }

        runCatching { ContinuonbrainLink.CapabilityManifest.parseFrom(payload) }
            .onSuccess { manifest ->
                if (manifest.robotModel.isNotEmpty() || manifest.skillsCount > 0 || manifest.sensorsCount > 0) {
                    deliverManifest(manifest.toDomain())
                    return
                }
            }

        runCatching { ContinuonbrainLink.StreamRobotStateResponse.parseFrom(payload) }
            .onSuccess { response ->
                stateCallback?.invoke(response.toDomain())
            }
            .onFailure {
                Log.w(TAG, "Failed to decode robot state envelope", it)
            }
    }

    private fun subscribeToStateStream() {
        if (stateSubscribed) return
        val socket = webSocket ?: return
        val subscription = ContinuonbrainLink.StreamRobotStateRequest.newBuilder().setClientId(clientId).build()
        socket.send(ByteString.of(*subscription.toByteArray()))
        stateSubscribed = true
    }

    private fun subscribeToTelemetryStream() {
        if (telemetrySubscribed || editorTelemetryCallback == null) return
        val socket = webSocket ?: return
        val subscription = ContinuonbrainLink.StreamRobotEditorTelemetryRequest.newBuilder()
            .setClientId(clientId)
            .setIncludeDiagnostics(true)
            .setIncludeHopeCmsSignals(true)
            .build()
        socket.send(ByteString.of(*subscription.toByteArray()))
        telemetrySubscribed = true
    }

    private fun deliverManifest(manifest: CapabilityManifest) {
        if (pendingManifestCallbacks.isEmpty()) return
        val callbacks = pendingManifestCallbacks.toList()
        pendingManifestCallbacks.clear()
        callbacks.forEach { callback -> callback(manifest) }
    }

    companion object {
        private const val TAG = "CBWebRtcClient"
    }
}
