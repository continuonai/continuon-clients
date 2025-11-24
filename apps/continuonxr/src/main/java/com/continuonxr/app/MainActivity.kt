package com.continuonxr.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.continuonxr.app.input.FusedInput
import com.continuonxr.app.input.InputFusionEngine
import com.continuonxr.app.input.SensorSample
import com.continuonxr.app.navigation.ModeRoute
import com.continuonxr.app.navigation.ModeScreen
import com.continuonxr.app.navigation.modeScreens
import com.continuonxr.app.teleop.TeleopController
import com.continuonxr.app.teleop.TeleopState

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ContinuonXRApp()
        }
    }
}

@Composable
private fun ContinuonXRApp() {
    MaterialTheme {
        val fusionEngine = remember { InputFusionEngine() }
        val teleopController = remember { TeleopController(fusionEngine) }
        var activeRoute by rememberSaveable { mutableStateOf(ModeRoute.ModeA) }

        LaunchedEffect(Unit) {
            teleopController.connect()
        }

        Scaffold { padding ->
            Surface(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
            ) {
                ModeShellLayout(
                    activeRoute = activeRoute,
                    onRouteSelected = { activeRoute = it },
                    fusionEngine = fusionEngine,
                    teleopController = teleopController,
                )
            }
        }
    }
}

@Composable
private fun ModeShellLayout(
    activeRoute: ModeRoute,
    onRouteSelected: (ModeRoute) -> Unit,
    fusionEngine: InputFusionEngine,
    teleopController: TeleopController,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "ContinuonXR — Mode Shells",
            style = MaterialTheme.typography.headlineMedium,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            modeScreens.forEach { screen ->
                Button(onClick = { onRouteSelected(screen.route) }) {
                    Text(text = screen.title)
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        ModeContent(
            screen = modeScreens.first { it.route == activeRoute },
            fusionEngine = fusionEngine,
            teleopController = teleopController,
        )
    }
}

@Composable
private fun ModeContent(
    screen: ModeScreen,
    fusionEngine: InputFusionEngine,
    teleopController: TeleopController,
) {
    val sampleInput = listOf(
        SensorSample(name = "left_hand", value = 0.42f, timestampNanos = 1_000L),
        SensorSample(name = "right_hand", value = 0.58f, timestampNanos = 2_000L),
    )
    val fused: FusedInput = fusionEngine.fuse(sampleInput)
    val teleopState = teleopController.state()

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(text = screen.title, style = MaterialTheme.typography.titleLarge)
        Text(text = screen.description)

        XRStatusBlock(
            fusedInput = fused,
            teleopState = teleopState,
        )
    }
}

@Composable
private fun XRStatusBlock(
    fusedInput: FusedInput,
    teleopState: TeleopState,
) {
    Column(
        modifier = Modifier
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(
            text = "Fused Input Confidence: ${fusedInput.confidence}",
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            text = "Fused Input Value: ${fusedInput.combinedValue}",
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            text = fusedInput.description,
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            text = "Teleop State: ${teleopState.name}",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
