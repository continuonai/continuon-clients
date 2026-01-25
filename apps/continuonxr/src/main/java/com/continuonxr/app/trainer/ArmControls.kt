package com.continuonxr.app.trainer

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.GripperMode
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * ArmControls provides sliders for 6-axis robot arm control and gripper.
 *
 * Features:
 * - 6 joint sliders (J1-J6) with angle display
 * - Gripper slider (0-100%)
 * - Home and Ready preset buttons
 * - Real-time joint position feedback
 */
@Composable
fun ArmControls(
    modifier: Modifier = Modifier,
    jointPositions: List<Float> = List(6) { 0f },
    gripperPosition: Float = 0f,
    onJointCommand: (jointIndex: Int, delta: Float) -> Unit,
    onGripperCommand: (position: Float) -> Unit,
    onPreset: (preset: ArmPreset) -> Unit,
    enabled: Boolean = true
) {
    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .verticalScroll(scrollState)
            .padding(8.dp)
    ) {
        // Header with presets
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Arm Control",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                IconButton(
                    onClick = { onPreset(ArmPreset.HOME) },
                    enabled = enabled
                ) {
                    Icon(
                        imageVector = Icons.Default.Home,
                        contentDescription = "Home"
                    )
                }

                IconButton(
                    onClick = { onPreset(ArmPreset.READY) },
                    enabled = enabled
                ) {
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = "Ready"
                    )
                }
            }
        }

        // Joint sliders
        JointSliderGroup(
            jointPositions = jointPositions,
            onJointChange = onJointCommand,
            enabled = enabled
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Gripper control
        GripperSlider(
            position = gripperPosition,
            onPositionChange = onGripperCommand,
            enabled = enabled
        )
    }
}

/**
 * Group of joint sliders.
 */
@Composable
private fun JointSliderGroup(
    jointPositions: List<Float>,
    onJointChange: (jointIndex: Int, delta: Float) -> Unit,
    enabled: Boolean
) {
    val jointConfigs = listOf(
        JointConfig("J1", "Base", -180f, 180f),
        JointConfig("J2", "Shoulder", -90f, 90f),
        JointConfig("J3", "Elbow", -135f, 135f),
        JointConfig("J4", "Wrist1", -180f, 180f),
        JointConfig("J5", "Wrist2", -90f, 90f),
        JointConfig("J6", "Wrist3", -180f, 180f)
    )

    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        jointConfigs.forEachIndexed { index, config ->
            val currentPosition = jointPositions.getOrElse(index) { 0f }

            JointSlider(
                config = config,
                currentPosition = currentPosition,
                onValueChange = { newValue ->
                    val delta = newValue - currentPosition
                    if (abs(delta) > 0.01f) {
                        onJointChange(index, delta)
                    }
                },
                enabled = enabled
            )
        }
    }
}

/**
 * Individual joint slider.
 */
@Composable
private fun JointSlider(
    config: JointConfig,
    currentPosition: Float,
    onValueChange: (Float) -> Unit,
    enabled: Boolean
) {
    var sliderValue by remember(currentPosition) {
        mutableFloatStateOf(currentPosition)
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Joint label
        Column(modifier = Modifier.width(48.dp)) {
            Text(
                text = config.name,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = config.description,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // Slider
        Slider(
            value = sliderValue,
            onValueChange = { newValue ->
                sliderValue = newValue
            },
            onValueChangeFinished = {
                onValueChange(sliderValue)
            },
            valueRange = config.minAngle..config.maxAngle,
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 8.dp),
            enabled = enabled,
            colors = SliderDefaults.colors(
                thumbColor = MaterialTheme.colorScheme.primary,
                activeTrackColor = MaterialTheme.colorScheme.primary,
                inactiveTrackColor = MaterialTheme.colorScheme.surfaceVariant
            )
        )

        // Value display
        Text(
            text = "${sliderValue.roundToInt()}°",
            style = MaterialTheme.typography.labelMedium,
            modifier = Modifier.width(40.dp)
        )
    }
}

/**
 * Gripper control slider.
 */
@Composable
private fun GripperSlider(
    position: Float,
    onPositionChange: (Float) -> Unit,
    enabled: Boolean
) {
    var sliderValue by remember(position) {
        mutableFloatStateOf(position)
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f))
            .padding(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Gripper",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = when {
                    sliderValue < 0.1f -> "Closed"
                    sliderValue > 0.9f -> "Open"
                    else -> "${(sliderValue * 100).roundToInt()}%"
                },
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Closed button
            OutlinedButton(
                onClick = {
                    sliderValue = 0f
                    onPositionChange(0f)
                },
                modifier = Modifier.width(64.dp),
                enabled = enabled
            ) {
                Text("Close")
            }

            // Slider
            Slider(
                value = sliderValue,
                onValueChange = { newValue ->
                    sliderValue = newValue
                },
                onValueChangeFinished = {
                    onPositionChange(sliderValue)
                },
                valueRange = 0f..1f,
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 8.dp),
                enabled = enabled,
                colors = SliderDefaults.colors(
                    thumbColor = MaterialTheme.colorScheme.secondary,
                    activeTrackColor = MaterialTheme.colorScheme.secondary,
                    inactiveTrackColor = MaterialTheme.colorScheme.surfaceVariant
                )
            )

            // Open button
            OutlinedButton(
                onClick = {
                    sliderValue = 1f
                    onPositionChange(1f)
                },
                modifier = Modifier.width(64.dp),
                enabled = enabled
            ) {
                Text("Open")
            }
        }
    }
}

/**
 * Configuration for a single joint.
 */
data class JointConfig(
    val name: String,
    val description: String,
    val minAngle: Float,
    val maxAngle: Float
)

/**
 * Arm presets.
 */
enum class ArmPreset {
    HOME,   // All joints at zero
    READY,  // Arm in ready position
    STOW    // Arm in stowed/folded position
}

/**
 * Convert arm preset to joint angles.
 */
fun ArmPreset.toJointAngles(): List<Float> {
    return when (this) {
        ArmPreset.HOME -> List(6) { 0f }
        ArmPreset.READY -> listOf(0f, -45f, 90f, -45f, 0f, 0f)
        ArmPreset.STOW -> listOf(0f, -90f, 135f, -45f, 0f, 0f)
    }
}

/**
 * Create a ControlCommand from joint deltas.
 */
fun createJointDeltaCommand(jointIndex: Int, delta: Float): ControlCommand {
    val deltas = MutableList(6) { 0f }
    deltas[jointIndex] = delta * (Math.PI.toFloat() / 180f)  // Convert to radians
    return ControlCommand.JointDelta(deltas)
}

/**
 * Create a ControlCommand for gripper.
 */
fun createGripperCommand(position: Float): ControlCommand {
    return ControlCommand.Gripper(
        position = position,
        mode = GripperMode.POSITION
    )
}
