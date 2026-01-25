package com.continuonxr.app.trainer

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

/**
 * VoicePanel provides voice command interface.
 *
 * Features:
 * - Mic button (tap to listen / tap to stop)
 * - Real-time transcript display
 * - Listening indicator animation
 * - Command confirmation feedback
 */
@Composable
fun VoicePanel(
    transcript: String,
    isListening: Boolean,
    onStartListening: () -> Unit,
    onStopListening: () -> Unit,
    hasMicPermission: Boolean,
    onRequestPermission: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Mic button
        MicButton(
            isListening = isListening,
            hasMicPermission = hasMicPermission,
            onToggle = {
                if (!hasMicPermission) {
                    onRequestPermission()
                } else if (isListening) {
                    onStopListening()
                } else {
                    onStartListening()
                }
            }
        )

        Spacer(modifier = Modifier.width(12.dp))

        // Transcript display
        TranscriptDisplay(
            transcript = transcript,
            isListening = isListening,
            hasMicPermission = hasMicPermission,
            modifier = Modifier.weight(1f)
        )
    }
}

/**
 * Animated mic button.
 */
@Composable
private fun MicButton(
    isListening: Boolean,
    hasMicPermission: Boolean,
    onToggle: () -> Unit
) {
    // Pulse animation when listening
    val pulseAnim = rememberInfiniteTransition(label = "pulse")
    val scale by pulseAnim.animateFloat(
        initialValue = 1f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(500, easing = EaseInOutCubic),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    val backgroundColor by animateColorAsState(
        targetValue = when {
            !hasMicPermission -> MaterialTheme.colorScheme.surfaceVariant
            isListening -> MaterialTheme.colorScheme.error
            else -> MaterialTheme.colorScheme.primary
        },
        animationSpec = tween(300),
        label = "bgColor"
    )

    val contentColor by animateColorAsState(
        targetValue = when {
            !hasMicPermission -> MaterialTheme.colorScheme.onSurfaceVariant
            isListening -> MaterialTheme.colorScheme.onError
            else -> MaterialTheme.colorScheme.onPrimary
        },
        animationSpec = tween(300),
        label = "contentColor"
    )

    Box(
        modifier = Modifier
            .size(56.dp)
            .scale(if (isListening) scale else 1f)
    ) {
        // Outer ring when listening
        if (isListening) {
            val ringAlpha by pulseAnim.animateFloat(
                initialValue = 0.5f,
                targetValue = 0f,
                animationSpec = infiniteRepeatable(
                    animation = tween(1000),
                    repeatMode = RepeatMode.Restart
                ),
                label = "ringAlpha"
            )

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .scale(1.5f)
                    .background(
                        MaterialTheme.colorScheme.error.copy(alpha = ringAlpha),
                        CircleShape
                    )
            )
        }

        FilledIconButton(
            onClick = onToggle,
            modifier = Modifier.fillMaxSize(),
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = backgroundColor,
                contentColor = contentColor
            )
        ) {
            Icon(
                imageVector = if (isListening || hasMicPermission) Icons.Default.Mic
                else Icons.Default.MicOff,
                contentDescription = when {
                    !hasMicPermission -> "Mic permission required"
                    isListening -> "Stop listening"
                    else -> "Start listening"
                },
                modifier = Modifier.size(24.dp)
            )
        }
    }
}

/**
 * Transcript display area.
 */
@Composable
private fun TranscriptDisplay(
    transcript: String,
    isListening: Boolean,
    hasMicPermission: Boolean,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // Status text
        Text(
            text = when {
                !hasMicPermission -> "Tap mic to enable voice commands"
                isListening -> "Listening..."
                transcript.isNotBlank() -> "Last command:"
                else -> "Tap mic to speak"
            },
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(4.dp))

        // Transcript text
        if (transcript.isNotBlank()) {
            Text(
                text = transcript,
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        } else if (isListening) {
            // Listening indicator
            ListeningIndicator()
        } else {
            Text(
                text = "Say \"forward\", \"stop\", \"teach patrol\", etc.",
                style = MaterialTheme.typography.bodySmall,
                fontStyle = FontStyle.Italic,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
            )
        }
    }
}

/**
 * Animated listening indicator.
 */
@Composable
private fun ListeningIndicator() {
    val infiniteTransition = rememberInfiniteTransition(label = "listening")

    Row(
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(3) { index ->
            val delay = index * 150

            val animatedHeight by infiniteTransition.animateFloat(
                initialValue = 4f,
                targetValue = 16f,
                animationSpec = infiniteRepeatable(
                    animation = tween(
                        durationMillis = 600,
                        delayMillis = delay,
                        easing = EaseInOutCubic
                    ),
                    repeatMode = RepeatMode.Reverse
                ),
                label = "height$index"
            )

            Box(
                modifier = Modifier
                    .width(4.dp)
                    .height(animatedHeight.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(MaterialTheme.colorScheme.primary)
            )
        }
    }
}

/**
 * Voice command help overlay.
 */
@Composable
fun VoiceCommandHelp(
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.padding(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Voice Commands",
                style = MaterialTheme.typography.titleMedium
            )

            Spacer(modifier = Modifier.height(12.dp))

            CommandHelpItem("forward / back", "Drive robot")
            CommandHelpItem("left / right", "Turn robot")
            CommandHelpItem("stop", "Emergency stop")
            CommandHelpItem("arm up / down", "Move arm")
            CommandHelpItem("open / close gripper", "Gripper control")
            CommandHelpItem("teach [name]", "Start teaching mode")
            CommandHelpItem("done", "Save taught behavior")

            Spacer(modifier = Modifier.height(12.dp))

            TextButton(
                onClick = onDismiss,
                modifier = Modifier.align(Alignment.End)
            ) {
                Text("Got it")
            }
        }
    }
}

@Composable
private fun CommandHelpItem(command: String, description: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = "\"$command\"",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = description,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private val EaseInOutCubic = CubicBezierEasing(0.65f, 0f, 0.35f, 1f)
