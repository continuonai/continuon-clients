package com.continuonxr.app.trainer

import android.view.HapticFeedbackConstants
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * DriveControls provides a virtual joystick for robot base control.
 *
 * Features:
 * - 2-axis joystick (forward/back, left/right)
 * - Speed multiplier slider
 * - Emergency stop button (always visible)
 * - Haptic feedback on direction changes
 */
@Composable
fun DriveControls(
    modifier: Modifier = Modifier,
    onDriveCommand: (vx: Float, vy: Float, omega: Float) -> Unit,
    onStop: () -> Unit,
    enabled: Boolean = true
) {
    var speedMultiplier by remember { mutableFloatStateOf(0.5f) }

    Column(
        modifier = modifier.padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Joystick
        VirtualJoystick(
            modifier = Modifier.size(160.dp),
            onPositionChanged = { x, y ->
                if (enabled) {
                    // x = lateral, y = forward
                    val vx = y * speedMultiplier
                    val vy = -x * speedMultiplier  // Invert for left/right
                    onDriveCommand(vx, vy, 0f)
                }
            },
            onReleased = {
                onDriveCommand(0f, 0f, 0f)
            },
            enabled = enabled
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Speed slider
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Speed",
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.width(48.dp)
            )
            Slider(
                value = speedMultiplier,
                onValueChange = { speedMultiplier = it },
                valueRange = 0.25f..1f,
                steps = 2,
                modifier = Modifier.weight(1f),
                enabled = enabled
            )
            Text(
                text = "${(speedMultiplier * 100).toInt()}%",
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.width(40.dp)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Emergency stop button
        EmergencyStopButton(
            onClick = onStop,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

/**
 * Virtual joystick composable.
 */
@Composable
fun VirtualJoystick(
    modifier: Modifier = Modifier,
    size: Dp = 160.dp,
    onPositionChanged: (x: Float, y: Float) -> Unit,
    onReleased: () -> Unit = {},
    enabled: Boolean = true
) {
    val view = LocalView.current
    var knobPosition by remember { mutableStateOf(Offset.Zero) }
    var isDragging by remember { mutableStateOf(false) }
    var lastQuadrant by remember { mutableIntStateOf(-1) }

    val backgroundColor = if (enabled) {
        MaterialTheme.colorScheme.surfaceVariant
    } else {
        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
    }

    val knobColor = if (enabled) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
    }

    val borderColor = if (isDragging) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.outline
    }

    Box(
        modifier = modifier
            .background(backgroundColor, CircleShape)
            .pointerInput(enabled) {
                if (!enabled) return@pointerInput

                detectDragGestures(
                    onDragStart = { offset ->
                        isDragging = true
                        val center = Offset(size.toPx() / 2, size.toPx() / 2)
                        knobPosition = constrainToCircle(offset - center, size.toPx() / 2)
                        updatePosition(knobPosition, size.toPx() / 2, onPositionChanged)
                    },
                    onDragEnd = {
                        isDragging = false
                        knobPosition = Offset.Zero
                        onReleased()
                        lastQuadrant = -1
                    },
                    onDragCancel = {
                        isDragging = false
                        knobPosition = Offset.Zero
                        onReleased()
                        lastQuadrant = -1
                    },
                    onDrag = { change, dragAmount ->
                        change.consume()
                        val newPos = knobPosition + dragAmount
                        knobPosition = constrainToCircle(newPos, size.toPx() / 2 - 20.dp.toPx())

                        // Haptic feedback on quadrant change
                        val currentQuadrant = getQuadrant(knobPosition)
                        if (currentQuadrant != lastQuadrant && lastQuadrant != -1) {
                            view.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
                        }
                        lastQuadrant = currentQuadrant

                        updatePosition(knobPosition, size.toPx() / 2, onPositionChanged)
                    }
                )
            },
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val center = Offset(size.toPx() / 2, size.toPx() / 2)
            val radius = size.toPx() / 2 - 4.dp.toPx()

            // Outer circle
            drawCircle(
                color = borderColor,
                radius = radius,
                center = center,
                style = Stroke(width = 2.dp.toPx())
            )

            // Cross guides
            drawLine(
                color = borderColor.copy(alpha = 0.3f),
                start = Offset(center.x, 4.dp.toPx()),
                end = Offset(center.x, size.toPx() - 4.dp.toPx()),
                strokeWidth = 1.dp.toPx()
            )
            drawLine(
                color = borderColor.copy(alpha = 0.3f),
                start = Offset(4.dp.toPx(), center.y),
                end = Offset(size.toPx() - 4.dp.toPx(), center.y),
                strokeWidth = 1.dp.toPx()
            )

            // Knob
            val knobCenter = center + knobPosition
            val knobRadius = 24.dp.toPx()

            drawCircle(
                color = knobColor,
                radius = knobRadius,
                center = knobCenter
            )

            // Knob highlight
            drawCircle(
                color = Color.White.copy(alpha = 0.3f),
                radius = knobRadius - 4.dp.toPx(),
                center = knobCenter
            )
        }
    }
}

/**
 * Emergency stop button.
 */
@Composable
fun EmergencyStopButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val view = LocalView.current

    Button(
        onClick = {
            view.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            onClick()
        },
        modifier = modifier.height(56.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.error,
            contentColor = MaterialTheme.colorScheme.onError
        ),
        shape = MaterialTheme.shapes.medium
    ) {
        Text(
            text = "STOP",
            style = MaterialTheme.typography.titleMedium
        )
    }
}

/**
 * Constrain a point to be within a circle.
 */
private fun constrainToCircle(point: Offset, radius: Float): Offset {
    val distance = sqrt(point.x * point.x + point.y * point.y)
    return if (distance <= radius) {
        point
    } else {
        val angle = atan2(point.y, point.x)
        Offset(cos(angle) * radius, sin(angle) * radius)
    }
}

/**
 * Update position callback with normalized values.
 */
private fun updatePosition(
    position: Offset,
    maxRadius: Float,
    onPositionChanged: (x: Float, y: Float) -> Unit
) {
    val normalizedX = (position.x / maxRadius).coerceIn(-1f, 1f)
    val normalizedY = (-position.y / maxRadius).coerceIn(-1f, 1f)  // Invert Y
    onPositionChanged(normalizedX, normalizedY)
}

/**
 * Get quadrant (0-3) for haptic feedback.
 */
private fun getQuadrant(position: Offset): Int {
    val deadzone = 10f
    if (sqrt(position.x * position.x + position.y * position.y) < deadzone) {
        return -1  // Center
    }

    return when {
        position.x >= 0 && position.y < 0 -> 0  // Up-right
        position.x >= 0 && position.y >= 0 -> 1  // Down-right
        position.x < 0 && position.y >= 0 -> 2  // Down-left
        else -> 3  // Up-left
    }
}
