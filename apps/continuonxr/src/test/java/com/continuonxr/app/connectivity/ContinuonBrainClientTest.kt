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
