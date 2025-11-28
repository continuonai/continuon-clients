package com.continuonxr.app.connectivity

import com.continuonxr.app.config.ConnectivityConfig
import continuonxr.continuonbrain.v1.ContinuonbrainLink
import io.grpc.stub.StreamObserver
import kotlin.test.assertEquals
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.Test

class ContinuonBrainClientTest {

    @Test
    fun restartsStateStreamAfterErrorWithBackoff() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val clientScope = TestScope(dispatcher)
        val client = ContinuonBrainClient(config = defaultConfig(), coroutineScope = clientScope)

        val observers = mutableListOf<StreamObserver<ContinuonbrainLink.StreamRobotStateResponse>>()
        client.setStateStreamStarter { observer ->
            observers.add(observer)
        }

        val states = mutableListOf<RobotState>()
        client.observeState { states.add(it) }

        assertEquals(1, observers.size)

        observers.first().onError(RuntimeException("disconnect"))

        advanceTimeBy(600)

        assertEquals(2, observers.size)

        val response = ContinuonbrainLink.StreamRobotStateResponse.newBuilder()
            .setState(
                ContinuonbrainLink.RobotState.newBuilder()
                    .setTimestampNanos(123L)
                    .build()
            )
            .build()

        observers.last().onNext(response)

        assertEquals(listOf(123L), states.map { it.timestampNanos })
    }

    private fun defaultConfig() = ConnectivityConfig(
        continuonBrainHost = "127.0.0.1",
        continuonBrainPort = 50051,
        useWebRtc = false,
        cloudBaseUrl = "https://api.continuon.ai",
    )
}
