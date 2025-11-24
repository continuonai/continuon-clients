package com.continuonxr.app.connectivity

import com.continuonxr.app.config.ConnectivityConfig
import continuonxr.continuonbrain.v1.ContinuonBrainBridgeServiceGrpc
import continuonxr.continuonbrain.v1.ContinuonbrainLink
import io.grpc.ClientInterceptors
import io.grpc.ManagedChannel
import io.grpc.Metadata
import io.grpc.okhttp.OkHttpChannelBuilder
import io.grpc.stub.StreamObserver
import kotlinx.serialization.Serializable
import java.lang.IllegalArgumentException

/**
 * Stub for the ContinuonBrain/OS bridge client.
 * Responsible for opening gRPC/WebRTC streams for robot state and commands.
 */
class ContinuonBrainClient(private val config: ConnectivityConfig) {
    private var stateCallback: ((RobotState) -> Unit)? = null
    private var channel: ManagedChannel? = null
    private var stub: ContinuonBrainBridgeServiceGrpc.ContinuonBrainBridgeServiceStub? = null
    private var stateStreamStarted: Boolean = false
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
        startStateStreamIfReady()
    }

    private fun startStateStreamIfReady() {
        val callback = stateCallback ?: return
        val stateStub = stub ?: return
        if (stateStreamStarted) return
        stateStreamStarted = true
        stateStub.streamRobotState(
            ContinuonbrainLink.StreamRobotStateRequest.newBuilder().setClientId(clientId).build(),
            object : StreamObserver<ContinuonbrainLink.StreamRobotStateResponse> {
                override fun onNext(value: ContinuonbrainLink.StreamRobotStateResponse) {
                    callback.invoke(value.toDomain())
                }

                override fun onError(t: Throwable) {
                    // TODO: add retry/backoff
                    stateStreamStarted = false
                }

                override fun onCompleted() {
                    stateStreamStarted = false
                }
            }
        )
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
