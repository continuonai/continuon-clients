package com.continuonxr.app.teleop

import com.continuonxr.app.input.FusedInput
import com.continuonxr.app.input.InputFusionEngine
import com.continuonxr.app.input.SensorSample

sealed class TeleopResult {
    data class Success(val fusedInput: FusedInput, val message: String) : TeleopResult()
    data class Failure(val reason: String) : TeleopResult()
}

enum class TeleopState {
    Disconnected,
    Connecting,
    Connected,
}

class TeleopController(private val fusionEngine: InputFusionEngine) {
    private var currentState: TeleopState = TeleopState.Disconnected

    fun connect(): TeleopState {
        currentState = TeleopState.Connecting
        // Simulate a cheap, optimistic connection for build validation.
        currentState = TeleopState.Connected
        return currentState
    }

    fun disconnect(): TeleopState {
        currentState = TeleopState.Disconnected
        return currentState
    }

    fun state(): TeleopState = currentState

    fun sendControl(samples: List<SensorSample>): TeleopResult {
        return when (currentState) {
            TeleopState.Disconnected -> TeleopResult.Failure("Not connected to teleop host")
            TeleopState.Connecting -> TeleopResult.Failure("Connection still warming up")
            TeleopState.Connected -> {
                val fused = fusionEngine.fuse(samples)
                TeleopResult.Success(
                    fusedInput = fused,
                    message = "Stub control packet prepared",
                )
            }
        }
    }
}
