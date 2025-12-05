package com.continuonxr.app.connectivity

import com.continuonxr.app.config.ConnectivityConfig
import continuonxr.continuonbrain.v1.ContinuonbrainLink
import io.grpc.stub.StreamObserver
import kotlin.test.assertEquals
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.Test

class ContinuonBrainClientTest {

    @Test
    fun retriesStateStreamWithBackoffUntilClosed() = runTest {
        val streamFactory = RecordingStreamFactory()
        val client = ContinuonBrainClient(
            config = ConnectivityConfig(
                continuonBrainHost = "localhost",
                continuonBrainPort = 50051,
                useWebRtc = false,
                cloudBaseUrl = "http://localhost",
            ),
            coroutineScope = this,
            streamFactory = streamFactory::startStream,
        )

        client.observeState { }

        assertEquals(1, streamFactory.observers.size)

        streamFactory.observers.first().onError(RuntimeException("network glitch"))
        advanceTimeBy(500)

        assertEquals(2, streamFactory.observers.size)

        streamFactory.observers[1].onError(RuntimeException("still broken"))
        advanceTimeBy(1_000)

        assertEquals(3, streamFactory.observers.size)

        client.close()
        streamFactory.observers[2].onError(RuntimeException("after close"))
        advanceTimeBy(5_000)

        assertEquals(3, streamFactory.observers.size)
    }

    @Test
    fun deliversRobotStatesAcrossReconnects() = runTest {
        val streamFactory = RecordingStreamFactory()
        val client = ContinuonBrainClient(
            config = ConnectivityConfig(
                continuonBrainHost = "localhost",
                continuonBrainPort = 50051,
                useWebRtc = false,
                cloudBaseUrl = "http://localhost",
            ),
            coroutineScope = this,
            streamFactory = streamFactory::startStream,
        )
        val received = mutableListOf<RobotState>()

        client.observeState { received.add(it) }

        assertEquals(1, streamFactory.observers.size)

        val firstResponse = ContinuonbrainLink.StreamRobotStateResponse.newBuilder()
            .setState(
                ContinuonbrainLink.RobotState.newBuilder()
                    .setTimestampNanos(111)
                    .build()
            )
            .build()
        streamFactory.observers.first().onNext(firstResponse)

        assertEquals(listOf(111L), received.map { it.timestampNanos })

        streamFactory.observers.first().onError(RuntimeException("disconnect"))
        advanceTimeBy(600)

        assertEquals(2, streamFactory.observers.size)

        val secondResponse = ContinuonbrainLink.StreamRobotStateResponse.newBuilder()
            .setState(
                ContinuonbrainLink.RobotState.newBuilder()
                    .setTimestampNanos(222)
                    .build()
            )
            .build()
        streamFactory.observers.last().onNext(secondResponse)

        assertEquals(listOf(111L, 222L), received.map { it.timestampNanos })
    }

    @Test
    fun streamsEditorTelemetryResponses() = runTest {
        val telemetryFactory = RecordingTelemetryStreamFactory()
        val client = ContinuonBrainClient(
            config = ConnectivityConfig(
                continuonBrainHost = "localhost",
                continuonBrainPort = 50051,
                useWebRtc = false,
                cloudBaseUrl = "http://localhost",
            ),
            coroutineScope = this,
            editorTelemetryStreamFactory = telemetryFactory::startStream,
        )
        val received = mutableListOf<RobotEditorTelemetry>()

        client.observeEditorTelemetry { received.add(it) }

        val response = ContinuonbrainLink.StreamRobotEditorTelemetryResponse.newBuilder()
            .setRobotState(
                ContinuonbrainLink.RobotState.newBuilder()
                    .setTimestampNanos(333)
                    .setFrameId("studio_frame")
                    .build()
            )
            .setDiagnostics(
                ContinuonbrainLink.EditorDiagnostics.newBuilder()
                    .setLatencyMs(2.5f)
                    .setMockMode(true)
                    .build()
            )
            .setSafetyState(
                ContinuonbrainLink.SafetyState.newBuilder()
                    .setEstopEngaged(true)
                    .addActiveEnvelopes("workspace")
                    .build()
            )
            .addSafetySignals(
                ContinuonbrainLink.SafetySignal.newBuilder()
                    .setId("estop_override")
                    .setLabel("E-Stop Override")
                    .setSeverity("critical")
                    .setSource("safety_head")
                    .setValue(1.0)
                    .build()
            )
            .setHopeCmsSignals(
                ContinuonbrainLink.HopeCmsSignals.newBuilder()
                    .setMid(
                        ContinuonbrainLink.HopeMidSignals.newBuilder()
                            .setIntentLabel("pick_place")
                            .setIntentConfidence(0.82f)
                            .build()
                    )
                    .build()
            )
            .setCmsSnapshot(
                ContinuonbrainLink.CmsSnapshot.newBuilder()
                    .setSnapshotId("slow-loop-123")
                    .setPolicyVersion("policy-a")
                    .setMemoryPlaneVersion("memory-1")
                    .setCmsBalance("stable")
                    .setCreatedAt("2024-05-01T12:00:00Z")
                    .setSource("live")
                    .build()
            )
            .build()

        telemetryFactory.observers.first().onNext(response)

        assertEquals(1, received.size)
        val telemetry = received.first()
        assertEquals(333L, telemetry.robotState.timestampNanos)
        assertEquals("studio_frame", telemetry.robotState.frameId)
        assertEquals(true, telemetry.safetyState.estopEngaged)
        assertEquals(2.5f, telemetry.diagnostics.latencyMs)
        assertEquals("pick_place", telemetry.hopeCmsSignals.mid.intentLabel)
        assertEquals("estop_override", telemetry.safetySignals.first().id)
        assertEquals("slow-loop-123", telemetry.cmsSnapshot?.snapshotId)
        assertEquals(listOf("workspace"), telemetry.safetyState.activeEnvelopes)
    }
}

private class RecordingStreamFactory {
    val observers = mutableListOf<StreamObserver<ContinuonbrainLink.StreamRobotStateResponse>>()

    fun startStream(
        request: ContinuonbrainLink.StreamRobotStateRequest,
        observer: StreamObserver<ContinuonbrainLink.StreamRobotStateResponse>,
    ) {
        observers.add(observer)
    }
}

private class RecordingTelemetryStreamFactory {
    val observers = mutableListOf<StreamObserver<ContinuonbrainLink.StreamRobotEditorTelemetryResponse>>()

    fun startStream(
        request: ContinuonbrainLink.StreamRobotEditorTelemetryRequest,
        observer: StreamObserver<ContinuonbrainLink.StreamRobotEditorTelemetryResponse>,
    ) {
        observers.add(observer)
    }
}
