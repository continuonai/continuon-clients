package com.continuonxr.app.connectivity

import android.util.Log
import com.continuonxr.app.config.ConnectivityConfig
import continuonxr.continuonbrain.v1.ContinuonBrainBridgeServiceGrpc
import continuonxr.continuonbrain.v1.ContinuonbrainLink
import io.grpc.ClientInterceptors
import io.grpc.ManagedChannel
import io.grpc.Metadata
import io.grpc.okhttp.OkHttpChannelBuilder
import io.grpc.stub.StreamObserver
import java.lang.IllegalArgumentException
import kotlin.math.min
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.retryWhen
import kotlinx.coroutines.isActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.serialization.Serializable

/**
 * Stub for the ContinuonBrain/OS bridge client.
 * Responsible for opening gRPC/WebRTC streams for robot state and commands.
 */
class ContinuonBrainClient(
    private val config: ConnectivityConfig,
    private val coroutineScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
    private val streamFactory:
        ((ContinuonbrainLink.StreamRobotStateRequest, StreamObserver<ContinuonbrainLink.StreamRobotStateResponse>) -> Unit)? =
        null,
    private val editorTelemetryStreamFactory:
        ((
            ContinuonbrainLink.StreamRobotEditorTelemetryRequest,
            StreamObserver<ContinuonbrainLink.StreamRobotEditorTelemetryResponse>,
        ) -> Unit)? = null,
) {
    private var stateCallback: ((RobotState) -> Unit)? = null
    private var editorTelemetryCallback: ((RobotEditorTelemetry) -> Unit)? = null
    private var channel: ManagedChannel? = null
    private var stub: ContinuonBrainBridgeServiceGrpc.ContinuonBrainBridgeServiceStub? = null
    private var stateStreamStarted: Boolean = false
    private var stateStreamJob: Job? = null
    private var telemetryStreamStarted: Boolean = false
    private var telemetryStreamJob: Job? = null
    private var closed: Boolean = false
    private val clientId = "xr-client"
    private val authHeaderKey = Metadata.Key.of("Authorization", Metadata.ASCII_STRING_MARSHALLER)
    private val webRtcClient = ContinuonBrainWebRtcClient(config, clientId)

    fun connect() {
        closed = false
        if (config.useWebRtc) {
            webRtcClient.connect(stateCallback, editorTelemetryCallback)
        } else {
            connectGrpc()
            startStateStreamIfReady()
            startTelemetryStreamIfReady()
        }
    }

    fun sendCommand(command: ControlCommand) {
        if (config.useWebRtc) {
            webRtcClient.sendCommand(command)
            return
        }

        val envelope = command.toProto(clientId)
        val commandStub = stub ?: throw IllegalStateException("connect() must be called before sendCommand")
        commandStub.sendCommand(
            envelope,
            object : StreamObserver<ContinuonbrainLink.SendCommandResponse> {
                override fun onNext(value: ContinuonbrainLink.SendCommandResponse) {}
                override fun onError(t: Throwable) {}
                override fun onCompleted() {}
            }
        )
    }

    fun observeState(onState: (RobotState) -> Unit) {
        stateCallback = onState
        if (config.useWebRtc) {
            webRtcClient.observeState(onState)
        } else {
            startStateStreamIfReady()
        }
    }

    fun observeEditorTelemetry(onTelemetry: (RobotEditorTelemetry) -> Unit) {
        editorTelemetryCallback = onTelemetry
        if (config.useWebRtc) {
            webRtcClient.observeEditorTelemetry(onTelemetry)
        } else {
            startTelemetryStreamIfReady()
        }
    }

    suspend fun getCapabilityManifest(): CapabilityManifest? {
        if (config.useWebRtc) {
            return webRtcClient.requestCapabilityManifest()
        }

        val manifestRequest = ContinuonbrainLink.GetCapabilityManifestRequest.newBuilder()
            .setClientId(clientId)
            .setIncludeMockCapabilities(true)
            .build()
        val manifestStub = stub
            ?: throw IllegalStateException("connect() must be called before getCapabilityManifest")

        return suspendCancellableCoroutine { continuation ->
            manifestStub.getCapabilityManifest(
                manifestRequest,
                object : StreamObserver<ContinuonbrainLink.GetCapabilityManifestResponse> {
                    override fun onNext(value: ContinuonbrainLink.GetCapabilityManifestResponse) {
                        if (continuation.isActive) {
                            continuation.resume(value.manifest.toDomain())
                        }
                    }

                    override fun onError(t: Throwable) {
                        if (continuation.isActive) continuation.resumeWithException(t)
                    }

                    override fun onCompleted() {
                        if (continuation.isActive && !continuation.isCancelled) {
                            continuation.resume(null)
                        }
                    }
                },
            )
        }
    }

    fun close() {
        closed = true
        channel?.shutdownNow()
        stateStreamJob?.cancel()
        telemetryStreamJob?.cancel()
        coroutineScope.coroutineContext.cancelChildren()
        stateStreamStarted = false
        telemetryStreamStarted = false
        if (config.useWebRtc) {
            webRtcClient.close()
        }
    }

    private fun connectGrpc() {
        channel = OkHttpChannelBuilder
            .forAddress(config.continuonBrainHost, config.continuonBrainPort)
            .apply {
                if (config.useTls) {
                    useTransportSecurity()
                } else {
                    usePlaintext()
                }
            }
            .build()
        val baseStub = ContinuonBrainBridgeServiceGrpc.newStub(channel)
        stub = if (config.authToken != null) {
            val headers = Metadata()
            headers.put(authHeaderKey, "Bearer ${config.authToken}")
            ContinuonBrainBridgeServiceGrpc.newStub(ClientInterceptors.intercept(channel, MetadataClientInterceptor(headers)))
        } else {
            baseStub
        }
        startStateStreamIfReady()
    }

    private fun startStateStreamIfReady() {
        val callback = stateCallback ?: return
        if (stateStreamStarted || closed) return
        val streamStarter = streamFactory ?: stub?.let { streamStub ->
            { request: ContinuonbrainLink.StreamRobotStateRequest, observer: StreamObserver<ContinuonbrainLink.StreamRobotStateResponse> ->
                streamStub.streamRobotState(request, observer)
            }
        } ?: return
        stateStreamStarted = true
        val request = ContinuonbrainLink.StreamRobotStateRequest.newBuilder().setClientId(clientId).build()
        stateStreamJob?.cancel()
        stateStreamJob = robotStateFlow(streamStarter, request)
            .onEach { callback.invoke(it) }
            .retryWhen { cause, attempt ->
                if (!isActive || closed) return@retryWhen false
                stateStreamStarted = false
                val delayMs = calculateBackoffDelay(attempt)
                Log.i(TAG, "Robot state stream interrupted; retrying in ${delayMs}ms (attempt ${attempt + 1})", cause)
                delay(delayMs)
                stateStreamStarted = true
                true
            }
            .onCompletion { stateStreamStarted = false }
            .launchIn(coroutineScope)
    }

    private fun robotStateFlow(
        streamStarter: (
            ContinuonbrainLink.StreamRobotStateRequest,
            StreamObserver<ContinuonbrainLink.StreamRobotStateResponse>,
        ) -> Unit,
        request: ContinuonbrainLink.StreamRobotStateRequest,
    ): Flow<RobotState> = callbackFlow {
        val observer = object : StreamObserver<ContinuonbrainLink.StreamRobotStateResponse> {
            override fun onNext(value: ContinuonbrainLink.StreamRobotStateResponse) {
                trySend(value.toDomain())
            }

            override fun onError(t: Throwable) {
                close(t)
            }

            override fun onCompleted() {
                close()
            }
        }
        streamStarter.invoke(request, observer)
        awaitClose { }
    }

    private fun startTelemetryStreamIfReady() {
        val callback = editorTelemetryCallback ?: return
        if (telemetryStreamStarted || closed) return
        val streamStarter = editorTelemetryStreamFactory ?: stub?.let { streamStub ->
            { request: ContinuonbrainLink.StreamRobotEditorTelemetryRequest, observer: StreamObserver<ContinuonbrainLink.StreamRobotEditorTelemetryResponse> ->
                streamStub.streamRobotEditorTelemetry(request, observer)
            }
        } ?: return
        telemetryStreamStarted = true
        val request = ContinuonbrainLink.StreamRobotEditorTelemetryRequest.newBuilder()
            .setClientId(clientId)
            .setIncludeDiagnostics(true)
            .setIncludeHopeCmsSignals(true)
            .build()
        telemetryStreamJob?.cancel()
        telemetryStreamJob = editorTelemetryFlow(streamStarter, request)
            .onEach { callback.invoke(it) }
            .retryWhen { cause, attempt ->
                if (!isActive || closed) return@retryWhen false
                telemetryStreamStarted = false
                val delayMs = calculateBackoffDelay(attempt)
                Log.i(TAG, "Editor telemetry stream interrupted; retrying in ${delayMs}ms (attempt ${attempt + 1})", cause)
                delay(delayMs)
                telemetryStreamStarted = true
                true
            }
            .onCompletion { telemetryStreamStarted = false }
            .launchIn(coroutineScope)
    }

    private fun editorTelemetryFlow(
        streamStarter: (
            ContinuonbrainLink.StreamRobotEditorTelemetryRequest,
            StreamObserver<ContinuonbrainLink.StreamRobotEditorTelemetryResponse>,
        ) -> Unit,
        request: ContinuonbrainLink.StreamRobotEditorTelemetryRequest,
    ): Flow<RobotEditorTelemetry> = callbackFlow {
        val observer = object : StreamObserver<ContinuonbrainLink.StreamRobotEditorTelemetryResponse> {
            override fun onNext(value: ContinuonbrainLink.StreamRobotEditorTelemetryResponse) {
                trySend(value.toDomain())
            }

            override fun onError(t: Throwable) {
                close(t)
            }

            override fun onCompleted() {
                close()
            }
        }
        streamStarter.invoke(request, observer)
        awaitClose { }
    }

    private fun calculateBackoffDelay(attempt: Long): Long {
        val delayMs = INITIAL_RETRY_DELAY_MS * (1L shl min(attempt.toInt(), MAX_BACKOFF_EXPONENT))
        return min(delayMs, MAX_RETRY_DELAY_MS)
    }

    companion object {
        private const val TAG = "ContinuonBrainClient"
        private const val INITIAL_RETRY_DELAY_MS = 500L
        private const val MAX_RETRY_DELAY_MS = 5_000L
        private const val MAX_BACKOFF_EXPONENT = 3
    }
}

internal fun ControlCommand.toProto(clientId: String): ContinuonbrainLink.SendCommandRequest {
    val builder = ContinuonbrainLink.SendCommandRequest.newBuilder().setClientId(clientId)
    targetFrequencyHz?.let { builder.targetFrequencyHz = it }
    safety?.let {
        builder.safety = ContinuonbrainLink.SafetyStatus.newBuilder()
            .setEstopReleasedAck(it.estopReleasedAck)
            .apply { it.safetyToken?.let { token -> safetyToken = token } }
            .build()
    }
    return when (this) {
        is ControlCommand.EndEffectorVelocity -> {
            builder.controlMode = ContinuonbrainLink.ControlMode.CONTROL_MODE_EE_VELOCITY
            builder.eeVelocity = ContinuonbrainLink.EeVelocityCommand.newBuilder()
                .setLinearMps(linearMps.toProto())
                .setAngularRadS(angularRadS.toProto())
                .setReferenceFrame(referenceFrame.toProto())
                .build()
            builder.build()
        }
        is ControlCommand.JointDelta -> {
            require(deltaRadians.isNotEmpty()) { "Joint delta command requires at least one joint" }
            builder.controlMode = ContinuonbrainLink.ControlMode.CONTROL_MODE_JOINT_DELTA
            builder.jointDelta = ContinuonbrainLink.JointDeltaCommand.newBuilder()
                .addAllDeltaRadians(deltaRadians)
                .build()
            builder.build()
        }
        is ControlCommand.Gripper -> {
            builder.controlMode = ContinuonbrainLink.ControlMode.CONTROL_MODE_GRIPPER
            builder.gripper = ContinuonbrainLink.GripperCommand.newBuilder()
                .setMode(mode.toProto())
                .apply {
                    when (mode) {
                        GripperMode.POSITION -> positionM?.let { positionM = it }
                            ?: throw IllegalArgumentException("Gripper position mode requires positionM")
                        GripperMode.VELOCITY -> velocityMps?.let { velocityMps = it }
                            ?: throw IllegalArgumentException("Gripper velocity mode requires velocityMps")
                    }
                }
                .build()
            builder.build()
        }
    }
}

internal fun ContinuonbrainLink.StreamRobotStateResponse.toDomain(): RobotState {
    return state.toDomain()
}

internal fun Vector3.toProto(): ContinuonbrainLink.Vector3 =
    ContinuonbrainLink.Vector3.newBuilder().setX(x).setY(y).setZ(z).build()

internal fun ReferenceFrame.toProto(): ContinuonbrainLink.ReferenceFrame = when (this) {
    ReferenceFrame.BASE -> ContinuonbrainLink.ReferenceFrame.REFERENCE_FRAME_BASE
    ReferenceFrame.TOOL -> ContinuonbrainLink.ReferenceFrame.REFERENCE_FRAME_TOOL
}

internal fun GripperMode.toProto(): ContinuonbrainLink.GripperMode = when (this) {
    GripperMode.POSITION -> ContinuonbrainLink.GripperMode.GRIPPER_MODE_POSITION
    GripperMode.VELOCITY -> ContinuonbrainLink.GripperMode.GRIPPER_MODE_VELOCITY
}

@Serializable
data class RobotState(
    val timestampNanos: Long,
    val jointPositions: List<Float> = emptyList(),
    val endEffectorPose: Pose = Pose(),
    val gripperOpen: Boolean = false,
    val frameId: String? = null,
    val jointVelocities: List<Float> = emptyList(),
    val jointEfforts: List<Float> = emptyList(),
    val endEffectorTwist: List<Float> = emptyList(),
    val wallTimeMillis: Long? = null,
)

@Serializable
data class Pose(
    val position: List<Float> = listOf(0f, 0f, 0f),
    val orientationQuat: List<Float> = listOf(0f, 0f, 0f, 1f),
)
