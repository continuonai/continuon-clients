package com.continuonxr.app.trainer

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Circle
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.continuonxr.app.camera.CameraPreview
import com.continuonxr.app.camera.DetectionOverlay
import com.continuonxr.app.camera.rememberCameraPreviewState
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.nexa.*
import kotlinx.coroutines.launch

/**
 * TrainerScreen is the main UI for robot training.
 *
 * Layout (portrait):
 * ┌────────────────────────────────────┐
 * │  Camera Preview (with overlays)    │
 * │  [Describe]              [Rec●]    │
 * ├────────────────────────────────────┤
 * │ Voice Panel with transcript        │
 * ├──────────────┬─────────────────────┤
 * │  Joystick    │ J1-J6 sliders       │
 * │              │ Gripper slider      │
 * │  [STOP]      │                     │
 * └──────────────┴─────────────────────┘
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainerScreen(
    nexaManager: NexaManager,
    visionPipeline: VisionPipeline,
    voicePipeline: VoicePipeline,
    onCommand: (ControlCommand) -> Unit,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // State
    val cameraState = rememberCameraPreviewState()
    var isRecording by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }

    // Nexa state
    val detections by visionPipeline.detections.collectAsState()
    val sceneDescription by visionPipeline.sceneDescription.collectAsState()
    val isProcessingVision by visionPipeline.isProcessing.collectAsState()
    val transcript by voicePipeline.transcript.collectAsState()
    val isListening by voicePipeline.isListening.collectAsState()

    // Robot state (placeholder - would come from ContinuonBrainClient)
    var jointPositions by remember { mutableStateOf(List(6) { 0f }) }
    var gripperPosition by remember { mutableFloatStateOf(0.5f) }
    var isConnected by remember { mutableStateOf(false) }

    // Permission handling
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }

    var hasMicPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        hasCameraPermission = permissions[Manifest.permission.CAMERA] == true
        hasMicPermission = permissions[Manifest.permission.RECORD_AUDIO] == true
    }

    // Request permissions on first launch
    LaunchedEffect(Unit) {
        val permissionsToRequest = mutableListOf<String>()
        if (!hasCameraPermission) permissionsToRequest.add(Manifest.permission.CAMERA)
        if (!hasMicPermission) permissionsToRequest.add(Manifest.permission.RECORD_AUDIO)

        if (permissionsToRequest.isNotEmpty()) {
            permissionLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }

    Scaffold(
        modifier = modifier,
        topBar = {
            TopAppBar(
                title = { Text("Robot Trainer") },
                actions = {
                    // Connection indicator
                    Icon(
                        imageVector = Icons.Default.Circle,
                        contentDescription = if (isConnected) "Connected" else "Disconnected",
                        tint = if (isConnected) Color.Green else Color.Red,
                        modifier = Modifier.size(12.dp)
                    )

                    IconButton(onClick = { showSettings = true }) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Camera Preview Section (40% of screen)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.4f)
                    .clip(RoundedCornerShape(bottomStart = 16.dp, bottomEnd = 16.dp))
            ) {
                if (hasCameraPermission) {
                    CameraPreview(
                        modifier = Modifier.fillMaxSize(),
                        onFrameAvailable = { frame ->
                            cameraState.updateFrame(frame)
                            visionPipeline.submitFrame(frame)
                        },
                        enableFrameAnalysis = true,
                        frameInterval = 500L
                    )

                    // Detection overlay
                    DetectionOverlay(
                        detections = detections,
                        modifier = Modifier.fillMaxSize()
                    )

                    // Camera controls overlay
                    CameraOverlayControls(
                        onDescribe = {
                            cameraState.latestFrame?.let { frame ->
                                scope.launch {
                                    visionPipeline.describeScene(frame)
                                }
                            }
                        },
                        isProcessing = isProcessingVision,
                        isRecording = isRecording,
                        onToggleRecording = {
                            isRecording = !isRecording
                            if (isRecording) onStartRecording() else onStopRecording()
                        },
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .fillMaxWidth()
                            .padding(8.dp)
                    )

                    // Scene description popup
                    if (sceneDescription.isNotBlank()) {
                        SceneDescriptionCard(
                            description = sceneDescription,
                            onDismiss = { /* Clear description */ },
                            modifier = Modifier
                                .align(Alignment.TopCenter)
                                .padding(8.dp)
                        )
                    }
                } else {
                    // Permission needed
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(MaterialTheme.colorScheme.surfaceVariant),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                Icons.Default.CameraAlt,
                                contentDescription = null,
                                modifier = Modifier.size(48.dp)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text("Camera permission required")
                            TextButton(onClick = {
                                permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
                            }) {
                                Text("Grant Permission")
                            }
                        }
                    }
                }
            }

            // Voice Panel
            VoicePanel(
                transcript = transcript,
                isListening = isListening,
                onStartListening = { voicePipeline.startListening() },
                onStopListening = { voicePipeline.stopListening() },
                hasMicPermission = hasMicPermission,
                onRequestPermission = {
                    permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp)
            )

            // Controls Section (60% of screen)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.5f)
                    .padding(8.dp)
            ) {
                // Drive controls (left side)
                DriveControls(
                    modifier = Modifier
                        .weight(0.4f)
                        .fillMaxHeight(),
                    onDriveCommand = { vx, vy, omega ->
                        // Drive commands go to base controller
                        // For now, we'll use EE velocity as placeholder
                        onCommand(
                            ControlCommand.EndEffectorVelocity(
                                vx = vx, vy = vy, vz = 0f,
                                wx = 0f, wy = 0f, wz = omega
                            )
                        )
                    },
                    onStop = {
                        onCommand(
                            ControlCommand.EndEffectorVelocity(
                                vx = 0f, vy = 0f, vz = 0f,
                                wx = 0f, wy = 0f, wz = 0f
                            )
                        )
                    },
                    enabled = isConnected
                )

                // Arm controls (right side)
                ArmControls(
                    modifier = Modifier
                        .weight(0.6f)
                        .fillMaxHeight(),
                    jointPositions = jointPositions,
                    gripperPosition = gripperPosition,
                    onJointCommand = { jointIndex, delta ->
                        onCommand(createJointDeltaCommand(jointIndex, delta))
                    },
                    onGripperCommand = { position ->
                        gripperPosition = position
                        onCommand(createGripperCommand(position))
                    },
                    onPreset = { preset ->
                        jointPositions = preset.toJointAngles()
                        // Send preset command
                    },
                    enabled = isConnected
                )
            }
        }
    }
}

/**
 * Camera overlay controls (describe button, recording indicator).
 */
@Composable
private fun CameraOverlayControls(
    onDescribe: () -> Unit,
    isProcessing: Boolean,
    isRecording: Boolean,
    onToggleRecording: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .background(
                Color.Black.copy(alpha = 0.5f),
                RoundedCornerShape(8.dp)
            )
            .padding(8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Describe button
        Button(
            onClick = onDescribe,
            enabled = !isProcessing,
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                contentColor = MaterialTheme.colorScheme.onPrimaryContainer
            )
        ) {
            if (isProcessing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text("Describe")
        }

        // Recording indicator
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .clip(RoundedCornerShape(8.dp))
                .background(
                    if (isRecording) Color.Red.copy(alpha = 0.8f)
                    else Color.Gray.copy(alpha = 0.5f)
                )
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Icon(
                Icons.Default.Circle,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(12.dp)
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = if (isRecording) "REC" else "REC",
                color = Color.White,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
        }

        // Toggle recording button
        IconButton(onClick = onToggleRecording) {
            Icon(
                imageVector = if (isRecording) Icons.Default.Circle else Icons.Default.Circle,
                contentDescription = if (isRecording) "Stop Recording" else "Start Recording",
                tint = if (isRecording) Color.Red else Color.White
            )
        }
    }
}

/**
 * Scene description card overlay.
 */
@Composable
private fun SceneDescriptionCard(
    description: String,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.9f)
        )
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = "Scene Description",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}
