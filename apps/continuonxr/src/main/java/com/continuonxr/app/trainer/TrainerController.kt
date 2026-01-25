package com.continuonxr.app.trainer

import android.util.Log
import com.continuonxr.app.connectivity.ContinuonBrainClient
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.nexa.NexaManager
import com.continuonxr.app.nexa.VisionPipeline
import com.continuonxr.app.nexa.VoicePipeline
import com.continuonxr.app.voice.CommandParser
import com.continuonxr.app.voice.ParsedCommand
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * TrainerController coordinates the trainer UI components:
 * - Voice commands from VoicePipeline
 * - Manual controls (joystick, sliders)
 * - Vision feedback from VisionPipeline
 * - Robot communication via ContinuonBrainClient
 *
 * Acts as the "brain" connecting all trainer components.
 */
class TrainerController(
    private val nexaManager: NexaManager,
    private val brainClient: ContinuonBrainClient,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
) {
    // Pipelines
    val visionPipeline = VisionPipeline(nexaManager)
    val voicePipeline = VoicePipeline(nexaManager)

    // Command parser with updatable behaviors
    private var commandParser = CommandParser()

    // Teaching mode state
    private val _teachingState = MutableStateFlow<TeachingState>(TeachingState.Idle)
    val teachingState: StateFlow<TeachingState> = _teachingState.asStateFlow()

    // Known behaviors
    private val _knownBehaviors = MutableStateFlow<Set<String>>(emptySet())
    val knownBehaviors: StateFlow<Set<String>> = _knownBehaviors.asStateFlow()

    // Connection state
    private val _isConnected = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected.asStateFlow()

    // Command history for teaching
    private val teachingBuffer = mutableListOf<ControlCommand>()

    // Stored behaviors
    private val behaviors = mutableMapOf<String, LearnedBehavior>()

    init {
        // Listen to voice transcripts and parse commands
        scope.launch {
            voicePipeline.transcript.collect { transcript ->
                if (transcript.isNotBlank()) {
                    handleVoiceCommand(transcript)
                }
            }
        }
    }

    /**
     * Initialize the controller and NexaSDK models.
     */
    suspend fun initialize(): Result<Unit> {
        Log.d(TAG, "Initializing TrainerController")

        return try {
            // Initialize NexaSDK
            nexaManager.initialize().getOrThrow()

            // Load models in parallel
            coroutineScope {
                launch { nexaManager.loadVlm() }
                launch { nexaManager.loadAsr() }
            }

            // Connect to robot
            connectToRobot()

            Log.d(TAG, "TrainerController initialized")
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize", e)
            Result.failure(e)
        }
    }

    /**
     * Connect to the robot via ContinuonBrainClient.
     */
    private suspend fun connectToRobot() {
        Log.d(TAG, "Connecting to robot...")
        try {
            // Connect the gRPC/WebRTC client
            brainClient.connect()

            // Set up robot state observation
            brainClient.observeState { state ->
                Log.v(TAG, "Robot state: joints=${state.jointPositions.size}, gripper=${state.gripperOpen}")
            }

            _isConnected.value = true
            Log.d(TAG, "Connected to robot")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to robot", e)
            _isConnected.value = false
            throw e
        }
    }

    /**
     * Send a control command to the robot.
     */
    fun sendCommand(command: ControlCommand) {
        scope.launch {
            try {
                // If teaching, record the command
                if (_teachingState.value is TeachingState.Recording) {
                    teachingBuffer.add(command)
                    Log.d(TAG, "Recorded command for teaching: $command")
                }

                // Send to robot
                brainClient.sendCommand(command)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send command", e)
            }
        }
    }

    /**
     * Handle a voice command transcript.
     */
    private fun handleVoiceCommand(transcript: String) {
        Log.d(TAG, "Processing voice: $transcript")

        val parsed = commandParser.parse(transcript)
        if (parsed == null) {
            Log.d(TAG, "No command recognized from: $transcript")
            return
        }

        Log.d(TAG, "Parsed command: $parsed")

        when (parsed) {
            is ParsedCommand.Stop -> handleStop()
            is ParsedCommand.Drive -> handleDrive(parsed)
            is ParsedCommand.Arm -> handleArm(parsed)
            is ParsedCommand.Gripper -> handleGripper(parsed)
            is ParsedCommand.ArmHome -> handleArmPreset(ArmPreset.HOME)
            is ParsedCommand.ArmReady -> handleArmPreset(ArmPreset.READY)
            is ParsedCommand.TeachStart -> startTeaching(parsed.name)
            is ParsedCommand.TeachEnd -> endTeaching()
            is ParsedCommand.TeachCancel -> cancelTeaching()
            is ParsedCommand.RunBehavior -> runBehavior(parsed.name)
            is ParsedCommand.LookAt -> handleLookAt(parsed.target)
        }
    }

    private fun handleStop() {
        Log.d(TAG, "STOP command")
        sendCommand(ControlCommand.EndEffectorVelocity(0f, 0f, 0f, 0f, 0f, 0f))
    }

    private fun handleDrive(cmd: ParsedCommand.Drive) {
        // Drive commands would go to base controller
        // For now, log and skip
        Log.d(TAG, "Drive command: vx=${cmd.vx}, vy=${cmd.vy}, omega=${cmd.omega}")
    }

    private fun handleArm(cmd: ParsedCommand.Arm) {
        sendCommand(ControlCommand.EndEffectorVelocity(
            vx = cmd.dx, vy = cmd.dy, vz = cmd.dz,
            wx = 0f, wy = 0f, wz = 0f
        ))
    }

    private fun handleGripper(cmd: ParsedCommand.Gripper) {
        sendCommand(createGripperCommand(cmd.position))
    }

    private fun handleArmPreset(preset: ArmPreset) {
        Log.d(TAG, "Arm preset: $preset")
        // Send joint position commands for preset
        val angles = preset.toJointAngles()
        // Would need absolute joint position command type
    }

    private fun handleLookAt(target: String) {
        Log.d(TAG, "Look at: $target")
        // Use vision pipeline to find object and point arm at it
        scope.launch {
            visionPipeline.detections.value.find { it.label.contains(target, ignoreCase = true) }
                ?.let { detection ->
                    Log.d(TAG, "Found $target at ${detection.boundingBox}")
                    // Calculate arm motion to point at object
                }
        }
    }

    /**
     * Start teaching mode for a new behavior.
     */
    private fun startTeaching(name: String) {
        if (_teachingState.value !is TeachingState.Idle) {
            Log.w(TAG, "Already in teaching mode")
            return
        }

        Log.d(TAG, "Starting teaching: $name")
        teachingBuffer.clear()
        _teachingState.value = TeachingState.Recording(name)
    }

    /**
     * End teaching and save the behavior.
     */
    private fun endTeaching() {
        val state = _teachingState.value
        if (state !is TeachingState.Recording) {
            Log.w(TAG, "Not in teaching mode")
            return
        }

        Log.d(TAG, "Ending teaching: ${state.behaviorName} with ${teachingBuffer.size} commands")

        // Save behavior
        val behavior = LearnedBehavior(
            name = state.behaviorName,
            commands = teachingBuffer.toList()
        )
        behaviors[state.behaviorName] = behavior

        // Update known behaviors
        _knownBehaviors.value = _knownBehaviors.value + state.behaviorName
        commandParser = commandParser.updateKnownBehaviors(_knownBehaviors.value)

        _teachingState.value = TeachingState.Idle
        teachingBuffer.clear()

        Log.d(TAG, "Saved behavior: ${behavior.name} with ${behavior.commands.size} commands")
    }

    /**
     * Cancel teaching without saving.
     */
    private fun cancelTeaching() {
        Log.d(TAG, "Canceling teaching")
        _teachingState.value = TeachingState.Idle
        teachingBuffer.clear()
    }

    /**
     * Run a previously learned behavior.
     * Plays back the recorded commands with timing.
     */
    private fun runBehavior(name: String) {
        val behavior = behaviors[name]
        if (behavior == null) {
            Log.w(TAG, "Unknown behavior: $name")
            return
        }

        if (_teachingState.value !is TeachingState.Idle) {
            Log.w(TAG, "Cannot run behavior while teaching")
            return
        }

        Log.d(TAG, "Running behavior: $name (${behavior.commands.size} commands)")
        _teachingState.value = TeachingState.Playing(name)

        scope.launch {
            try {
                for (command in behavior.commands) {
                    if (_teachingState.value !is TeachingState.Playing) {
                        Log.d(TAG, "Behavior playback interrupted")
                        break
                    }

                    brainClient.sendCommand(command)
                    // Small delay between commands for smooth playback
                    kotlinx.coroutines.delay(BEHAVIOR_COMMAND_DELAY_MS)
                }

                Log.d(TAG, "Behavior playback complete: $name")
            } catch (e: Exception) {
                Log.e(TAG, "Error during behavior playback", e)
            } finally {
                if (_teachingState.value is TeachingState.Playing) {
                    _teachingState.value = TeachingState.Idle
                }
            }
        }
    }

    /**
     * Stop behavior playback if running.
     */
    fun stopBehavior() {
        if (_teachingState.value is TeachingState.Playing) {
            Log.d(TAG, "Stopping behavior playback")
            _teachingState.value = TeachingState.Idle
            handleStop()  // Send stop command to robot
        }
    }

    companion object {
        private const val TAG = "TrainerController"
        private const val BEHAVIOR_COMMAND_DELAY_MS = 100L
    }

    /**
     * Release resources.
     */
    fun release() {
        Log.d(TAG, "Releasing TrainerController")
        scope.cancel()
        visionPipeline.release()
        voicePipeline.release()
        nexaManager.release()
    }
}

/**
 * Teaching mode state.
 */
sealed class TeachingState {
    object Idle : TeachingState()
    data class Recording(val behaviorName: String) : TeachingState()
    data class Playing(val behaviorName: String) : TeachingState()
}

/**
 * A learned behavior (sequence of commands).
 */
data class LearnedBehavior(
    val name: String,
    val commands: List<ControlCommand>,
    val createdAt: Long = System.currentTimeMillis()
)
