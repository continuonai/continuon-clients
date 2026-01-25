package com.continuonxr.app.voice

import android.util.Log
import com.continuonxr.app.connectivity.ControlCommand
import com.continuonxr.app.connectivity.ControlMode
import com.continuonxr.app.connectivity.GripperMode

/**
 * CommandParser extracts robot commands from voice transcripts.
 *
 * Supported command patterns:
 * - Drive: forward, back, left, right, turn left/right
 * - Stop: stop, halt, freeze, emergency
 * - Arm: arm up/down, arm left/right, arm forward/back
 * - Gripper: open gripper, close gripper, grip, release
 * - Teaching: teach [name], done, save, cancel
 * - Custom: [behavior name] (if previously taught)
 */
class CommandParser(
    private val knownBehaviors: Set<String> = emptySet()
) {
    companion object {
        private const val TAG = "CommandParser"

        // Default movement speeds
        const val DEFAULT_DRIVE_SPEED = 0.3f
        const val DEFAULT_TURN_SPEED = 0.5f
        const val DEFAULT_ARM_SPEED = 0.1f
        const val FAST_MULTIPLIER = 2.0f
        const val SLOW_MULTIPLIER = 0.5f
    }

    /**
     * Parse a transcript into a robot command.
     * Returns null if no command is recognized.
     */
    fun parse(transcript: String): ParsedCommand? {
        val normalized = normalizeTranscript(transcript)
        if (normalized.isBlank()) return null

        Log.d(TAG, "Parsing: '$normalized'")

        // Try each command type in order of priority
        return parseStop(normalized)
            ?: parseDrive(normalized)
            ?: parseArm(normalized)
            ?: parseGripper(normalized)
            ?: parseTeaching(normalized)
            ?: parseKnownBehavior(normalized)
            ?: parseLookAt(normalized)
    }

    /**
     * Normalize transcript for matching.
     */
    private fun normalizeTranscript(transcript: String): String {
        return transcript
            .lowercase()
            .replace(Regex("[^a-z0-9\\s]"), "")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    /**
     * Parse stop commands (highest priority).
     */
    private fun parseStop(text: String): ParsedCommand? {
        val stopPatterns = listOf(
            "stop", "halt", "freeze", "emergency", "emergency stop",
            "hold", "wait", "pause"
        )

        return if (stopPatterns.any { text.contains(it) }) {
            ParsedCommand.Stop
        } else null
    }

    /**
     * Parse drive commands.
     */
    private fun parseDrive(text: String): ParsedCommand? {
        val speedMultiplier = when {
            text.contains("fast") || text.contains("quick") -> FAST_MULTIPLIER
            text.contains("slow") || text.contains("careful") -> SLOW_MULTIPLIER
            else -> 1.0f
        }

        val baseSpeed = DEFAULT_DRIVE_SPEED * speedMultiplier

        // Forward/backward
        when {
            text.matches(Regex(".*\\b(go |move )?forward\\b.*")) ->
                return ParsedCommand.Drive(vx = baseSpeed, vy = 0f, omega = 0f)

            text.matches(Regex(".*\\b(go |move )?back(ward)?\\b.*")) ->
                return ParsedCommand.Drive(vx = -baseSpeed, vy = 0f, omega = 0f)
        }

        // Turn commands
        when {
            text.matches(Regex(".*\\bturn (to the )?left\\b.*")) ||
            text.matches(Regex(".*\\brotate left\\b.*")) ->
                return ParsedCommand.Drive(vx = 0f, vy = 0f, omega = DEFAULT_TURN_SPEED * speedMultiplier)

            text.matches(Regex(".*\\bturn (to the )?right\\b.*")) ||
            text.matches(Regex(".*\\brotate right\\b.*")) ->
                return ParsedCommand.Drive(vx = 0f, vy = 0f, omega = -DEFAULT_TURN_SPEED * speedMultiplier)
        }

        // Strafe commands
        when {
            text.matches(Regex(".*\\b(strafe |move |go )?left\\b.*")) ->
                return ParsedCommand.Drive(vx = 0f, vy = baseSpeed, omega = 0f)

            text.matches(Regex(".*\\b(strafe |move |go )?right\\b.*")) ->
                return ParsedCommand.Drive(vx = 0f, vy = -baseSpeed, omega = 0f)
        }

        return null
    }

    /**
     * Parse arm commands.
     */
    private fun parseArm(text: String): ParsedCommand? {
        if (!text.contains("arm")) return null

        val speedMultiplier = when {
            text.contains("fast") -> FAST_MULTIPLIER
            text.contains("slow") -> SLOW_MULTIPLIER
            else -> 1.0f
        }

        val speed = DEFAULT_ARM_SPEED * speedMultiplier

        return when {
            text.contains("up") -> ParsedCommand.Arm(dx = 0f, dy = 0f, dz = speed)
            text.contains("down") -> ParsedCommand.Arm(dx = 0f, dy = 0f, dz = -speed)
            text.contains("forward") -> ParsedCommand.Arm(dx = speed, dy = 0f, dz = 0f)
            text.contains("back") -> ParsedCommand.Arm(dx = -speed, dy = 0f, dz = 0f)
            text.contains("left") -> ParsedCommand.Arm(dx = 0f, dy = speed, dz = 0f)
            text.contains("right") -> ParsedCommand.Arm(dx = 0f, dy = -speed, dz = 0f)
            text.contains("home") -> ParsedCommand.ArmHome
            text.contains("ready") -> ParsedCommand.ArmReady
            else -> null
        }
    }

    /**
     * Parse gripper commands.
     */
    private fun parseGripper(text: String): ParsedCommand? {
        return when {
            text.matches(Regex(".*\\b(open|release)( the)?( gripper)?\\b.*")) ->
                ParsedCommand.Gripper(position = 1.0f)

            text.matches(Regex(".*\\b(close|grip|grab)( the)?( gripper)?\\b.*")) ->
                ParsedCommand.Gripper(position = 0.0f)

            text.matches(Regex(".*\\bhalf( open)?( gripper)?\\b.*")) ->
                ParsedCommand.Gripper(position = 0.5f)

            else -> null
        }
    }

    /**
     * Parse teaching mode commands.
     */
    private fun parseTeaching(text: String): ParsedCommand? {
        // Start teaching
        val teachMatch = Regex(".*\\bteach\\s+(\\w+)\\b.*").find(text)
        if (teachMatch != null) {
            val behaviorName = teachMatch.groupValues[1]
            return ParsedCommand.TeachStart(behaviorName)
        }

        // End teaching
        if (text.matches(Regex(".*\\b(done|finished|save|end teaching)\\b.*"))) {
            return ParsedCommand.TeachEnd
        }

        // Cancel teaching
        if (text.matches(Regex(".*\\b(cancel|abort|discard)\\b.*"))) {
            return ParsedCommand.TeachCancel
        }

        // Show teaching
        if (text.matches(Regex(".*\\bshow\\s+(\\w+)\\b.*"))) {
            val nameMatch = Regex("show\\s+(\\w+)").find(text)
            val name = nameMatch?.groupValues?.get(1) ?: return null
            return ParsedCommand.RunBehavior(name)
        }

        return null
    }

    /**
     * Parse known behavior names.
     */
    private fun parseKnownBehavior(text: String): ParsedCommand? {
        for (behavior in knownBehaviors) {
            if (text.contains(behavior.lowercase())) {
                return ParsedCommand.RunBehavior(behavior)
            }
        }
        return null
    }

    /**
     * Parse look-at commands.
     */
    private fun parseLookAt(text: String): ParsedCommand? {
        val lookMatch = Regex(".*\\blook at( the)?\\s+(.+)").find(text)
        if (lookMatch != null) {
            val target = lookMatch.groupValues[2].trim()
            return ParsedCommand.LookAt(target)
        }
        return null
    }

    /**
     * Update known behaviors for custom command recognition.
     */
    fun updateKnownBehaviors(behaviors: Set<String>): CommandParser {
        return CommandParser(behaviors)
    }
}

/**
 * Parsed command types.
 */
sealed class ParsedCommand {
    /** Emergency stop */
    object Stop : ParsedCommand()

    /** Drive command with velocity components */
    data class Drive(
        val vx: Float,    // Forward velocity (m/s)
        val vy: Float,    // Lateral velocity (m/s)
        val omega: Float  // Angular velocity (rad/s)
    ) : ParsedCommand()

    /** Arm end-effector velocity command */
    data class Arm(
        val dx: Float,  // X velocity
        val dy: Float,  // Y velocity
        val dz: Float   // Z velocity
    ) : ParsedCommand()

    /** Gripper position command */
    data class Gripper(
        val position: Float  // 0.0 = closed, 1.0 = open
    ) : ParsedCommand()

    /** Move arm to home position */
    object ArmHome : ParsedCommand()

    /** Move arm to ready position */
    object ArmReady : ParsedCommand()

    /** Start teaching a new behavior */
    data class TeachStart(val name: String) : ParsedCommand()

    /** End teaching and save behavior */
    object TeachEnd : ParsedCommand()

    /** Cancel teaching without saving */
    object TeachCancel : ParsedCommand()

    /** Run a previously taught behavior */
    data class RunBehavior(val name: String) : ParsedCommand()

    /** Look at a named object */
    data class LookAt(val target: String) : ParsedCommand()

    /**
     * Convert to ControlCommand for sending to robot.
     */
    fun toControlCommand(): ControlCommand? {
        return when (this) {
            is Stop -> ControlCommand.EndEffectorVelocity(
                vx = 0f, vy = 0f, vz = 0f,
                wx = 0f, wy = 0f, wz = 0f
            )
            is Drive -> null  // Drive commands go to base, not arm
            is Arm -> ControlCommand.EndEffectorVelocity(
                vx = dx, vy = dy, vz = dz,
                wx = 0f, wy = 0f, wz = 0f
            )
            is Gripper -> ControlCommand.Gripper(
                position = position,
                mode = GripperMode.POSITION
            )
            else -> null
        }
    }
}
