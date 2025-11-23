package com.continuonxr.app.connectivity

import com.continuonxr.app.config.ConnectivityConfig
import continuonxr.continuonbrain.v1.ContinuonBrainBridgeGrpc
import continuonxr.continuonbrain.v1.ContinuonbrainLink
import io.grpc.ManagedChannel
import io.grpc.Metadata
import io.grpc.ClientInterceptors
import io.grpc.okhttp.OkHttpChannelBuilder
import io.grpc.stub.StreamObserver
import kotlinx.serialization.Serializable
import java.util.Timer
import kotlin.concurrent.scheduleAtFixedRate

/**
 * Stub for the ContinuonBrain/OS bridge client.
 * Responsible for opening gRPC/WebRTC streams for robot state and commands.
 */
class ContinuonBrainClient(private val config: ConnectivityConfig) {
    private var stateCallback: ((RobotState) -> Unit)? = null
    private var mockTimer: Timer? = null
    private var mockTick: Long = 0
    private var lastCommand: List<Float> = emptyList()
    private var channel: ManagedChannel? = null
    private var stub: ContinuonBrainBridgeGrpc.ContinuonBrainBridgeStub? = null
    private val clientId = "xr-client"
    private val authHeaderKey = Metadata.Key.of("Authorization", Metadata.ASCII_STRING_MARSHALLER)
    private val webRtcClient = ContinuonBrainWebRtcClient(config)

    fun connect() {
        when {
            config.useMockContinuonBrain -> startMockStream()
            config.useWebRtc -> webRtcClient.connect()
            else -> startGrpcStream()
        }
    }

    fun sendCommand(command: List<Float>) {
        lastCommand = command
        if (config.useWebRtc) {
            webRtcClient.sendCommand(command)
        } else {
            stub?.sendCommand(
                ContinuonbrainLink.CommandEnvelope.newBuilder()
                    .setClientId(clientId)
                    .addAllCommand(command)
                    .build(),
                object : StreamObserver<ContinuonbrainLink.CommandAck> {
                    override fun onNext(value: ContinuonbrainLink.CommandAck) {}
                    override fun onError(t: Throwable) {}
                    override fun onCompleted() {}
                }
            )
        }
    }

    fun observeState(onState: (RobotState) -> Unit) {
        stateCallback = onState
        if (config.useWebRtc && !config.useMockContinuonBrain) {
            webRtcClient.observeState(onState)
        }
    }

    fun close() {
        mockTimer?.cancel()
        channel?.shutdownNow()
        if (config.useWebRtc) {
            webRtcClient.close()
        }
    }

    private fun startMockStream() {
        mockTimer?.cancel()
        mockTimer = Timer("pb-mock", true).apply {
            scheduleAtFixedRate(0L, 50L) {
                emitMockState()
            }
        }
    }

    private fun emitMockState() {
        stateCallback?.invoke(
            RobotState(
                timestampNanos = System.nanoTime(),
                jointPositions = listOf(0.1f * (mockTick % 10), 0f, 0f, 0f, 0f, 0f),
                endEffectorPose = Pose(
                    position = listOf(0.0f, 0.0f, 0.5f),
                    orientationQuat = listOf(0f, 0f, 0f, 1f),
                ),
                gripperOpen = (mockTick % 20L) < 10L,
                frameId = "mock-frame-$mockTick",
                jointVelocities = listOf(0f, 0f, 0f, 0f, 0f, 0f),
                jointEfforts = listOf(0f, 0f, 0f, 0f, 0f, 0f),
                endEffectorTwist = listOf(0f, 0f, 0f, 0f, 0f, 0f),
                wallTimeMillis = System.currentTimeMillis(),
            )
        )
        mockTick++
    }

    private fun startGrpcStream() {
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
        val baseStub = ContinuonBrainBridgeGrpc.newStub(channel)
        stub = if (config.authToken != null) {
            val headers = Metadata()
            headers.put(authHeaderKey, "Bearer ${config.authToken}")
            ContinuonBrainBridgeGrpc.newStub(ClientInterceptors.intercept(channel, MetadataClientInterceptor(headers)))
        } else {
            baseStub
        }
        stub?.streamRobotState(
            ContinuonbrainLink.StateRequest.newBuilder().setClientId(clientId).build(),
            object : StreamObserver<ContinuonbrainLink.RobotStateEnvelope> {
                override fun onNext(value: ContinuonbrainLink.RobotStateEnvelope) {
                    stateCallback?.invoke(value.toDomain())
                }

                override fun onError(t: Throwable) {
                    // TODO: add retry/backoff
                }

                override fun onCompleted() {}
            }
        )
    }

    private fun ContinuonbrainLink.RobotStateEnvelope.toDomain(): RobotState {
        val proto = this.state
        return RobotState(
            timestampNanos = proto.timestampNanos,
            jointPositions = proto.jointPositionsList.map { it },
            endEffectorPose = Pose(
                position = proto.endEffectorPose.positionList.map { it },
                orientationQuat = proto.endEffectorPose.orientationQuatList.map { it },
            ),
            gripperOpen = proto.gripperOpen,
            frameId = proto.frameId,
            jointVelocities = proto.jointVelocitiesList.map { it },
            jointEfforts = proto.jointEffortsList.map { it },
            endEffectorTwist = proto.endEffectorTwistList.map { it },
            wallTimeMillis = proto.wallTimeMillis,
        )
    }
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
