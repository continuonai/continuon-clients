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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable

/**
 * Stub for the ContinuonBrain/OS bridge client.
 * Responsible for opening gRPC/WebRTC streams for robot state and commands.
 */
class ContinuonBrainClient(
    private val config: ConnectivityConfig,
    private val coroutineScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
    private var stateCallback: ((RobotState) -> Unit)? = null
    private var channel: ManagedChannel? = null
    private var stub: ContinuonBrainBridgeServiceGrpc.ContinuonBrainBridgeServiceStub? = null
    private var stateStreamStarted: Boolean = false
    private var stateStreamStarter: ((StreamObserver<ContinuonbrainLink.StreamRobotStateResponse>) -> Unit)? = null
    private var reconnectJob: Job? = null
    private val clientId = "xr-client"
    private val authHeaderKey = Metadata.Key.of("Authorization", Metadata.ASCII_STRING_MARSHALLER)
    private val webRtcClient = ContinuonBrainWebRtcClient(config, clientId)

    fun connect() {
        if (config.useWebRtc) {
            webRtcClient.connect()
        } else {
            connectGrpc()
            startStateStreamIfReady()
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

    fun close() {
        reconnectJob?.cancel()
        stateStreamStarted = false
        channel?.shutdownNow()
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
        val request = ContinuonbrainLink.StreamRobotStateRequest.newBuilder().setClientId(clientId).build()
        val stateStub = stub
        stateStreamStarter = stateStub?.let { stubWithHeaders ->
            { observer ->
                stubWithHeaders.streamRobotState(request, observer)
            }
        }
        startStateStreamIfReady()
    }

    private fun startStateStreamIfReady() {
        val callback = stateCallback ?: return
        val streamStarter = stateStreamStarter ?: return
        if (stateStreamStarted) return
        stateStreamStarted = true
        reconnectJob?.cancel()
        reconnectJob = null
        streamStarter.invoke(
            object : StreamObserver<ContinuonbrainLink.StreamRobotStateResponse> {
                override fun onNext(value: ContinuonbrainLink.StreamRobotStateResponse) {
                    callback.invoke(value.toDomain())
                }

                override fun onError(t: Throwable) {
                    Log.w(TAG, "State stream error; scheduling reconnect", t)
                    stateStreamStarted = false
                    scheduleStateStreamReconnect(t)
                }

                override fun onCompleted() {
                    stateStreamStarted = false
                    scheduleStateStreamReconnect()
                }
            }
        )
    }

    private fun scheduleStateStreamReconnect(cause: Throwable? = null) {
        if (reconnectJob?.isActive == true) return
        reconnectJob = coroutineScope.launch {
            var attempt = 0
            var delayMs = INITIAL_RECONNECT_DELAY_MS
            while (isActive && !stateStreamStarted) {
                delay(delayMs)
                startStateStreamIfReady()
                if (stateStreamStarted) {
                    Log.i(TAG, "Robot state stream reconnected after $attempt retries")
                    return@launch
                }
                attempt += 1
                delayMs = min(delayMs * 2, MAX_RECONNECT_DELAY_MS)
                Log.w(TAG, "State stream reconnect attempt $attempt failed; retrying in ${delayMs}ms", cause)
            }
        }
    }

    internal fun setStateStreamStarter(starter: (StreamObserver<ContinuonbrainLink.StreamRobotStateResponse>) -> Unit) {
        stateStreamStarter = starter
    }

    companion object {
        private const val TAG = "ContinuonBrainClient"
        private const val INITIAL_RECONNECT_DELAY_MS = 500L
        private const val MAX_RECONNECT_DELAY_MS = 5_000L
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
