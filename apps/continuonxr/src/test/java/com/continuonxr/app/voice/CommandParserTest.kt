package com.continuonxr.app.voice

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class CommandParserTest {

    private lateinit var parser: CommandParser

    @Before
    fun setup() {
        parser = CommandParser()
    }

    // Stop commands
    @Test
    fun `parse stop command`() {
        val result = parser.parse("stop")
        assertEquals(ParsedCommand.Stop, result)
    }

    @Test
    fun `parse emergency stop`() {
        val result = parser.parse("emergency stop")
        assertEquals(ParsedCommand.Stop, result)
    }

    @Test
    fun `parse halt command`() {
        val result = parser.parse("halt the robot")
        assertEquals(ParsedCommand.Stop, result)
    }

    // Drive commands
    @Test
    fun `parse forward command`() {
        val result = parser.parse("go forward")
        assertTrue(result is ParsedCommand.Drive)
        val drive = result as ParsedCommand.Drive
        assertTrue(drive.vx > 0)
        assertEquals(0f, drive.vy, 0.001f)
    }

    @Test
    fun `parse backward command`() {
        val result = parser.parse("move back")
        assertTrue(result is ParsedCommand.Drive)
        val drive = result as ParsedCommand.Drive
        assertTrue(drive.vx < 0)
    }

    @Test
    fun `parse turn left command`() {
        val result = parser.parse("turn left")
        assertTrue(result is ParsedCommand.Drive)
        val drive = result as ParsedCommand.Drive
        assertTrue(drive.omega > 0)
    }

    @Test
    fun `parse turn right command`() {
        val result = parser.parse("turn to the right")
        assertTrue(result is ParsedCommand.Drive)
        val drive = result as ParsedCommand.Drive
        assertTrue(drive.omega < 0)
    }

    @Test
    fun `parse fast forward command`() {
        val result = parser.parse("go forward fast")
        assertTrue(result is ParsedCommand.Drive)
        val drive = result as ParsedCommand.Drive
        // Fast should have higher speed
        assertTrue(drive.vx > CommandParser.DEFAULT_DRIVE_SPEED)
    }

    @Test
    fun `parse slow forward command`() {
        val result = parser.parse("move forward slowly")
        assertTrue(result is ParsedCommand.Drive)
        val drive = result as ParsedCommand.Drive
        // Slow should have lower speed
        assertTrue(drive.vx < CommandParser.DEFAULT_DRIVE_SPEED)
    }

    // Arm commands
    @Test
    fun `parse arm up command`() {
        val result = parser.parse("arm up")
        assertTrue(result is ParsedCommand.Arm)
        val arm = result as ParsedCommand.Arm
        assertTrue(arm.dz > 0)
    }

    @Test
    fun `parse arm down command`() {
        val result = parser.parse("arm down")
        assertTrue(result is ParsedCommand.Arm)
        val arm = result as ParsedCommand.Arm
        assertTrue(arm.dz < 0)
    }

    @Test
    fun `parse arm forward command`() {
        val result = parser.parse("move arm forward")
        assertTrue(result is ParsedCommand.Arm)
        val arm = result as ParsedCommand.Arm
        assertTrue(arm.dx > 0)
    }

    @Test
    fun `parse arm home command`() {
        val result = parser.parse("arm home")
        assertEquals(ParsedCommand.ArmHome, result)
    }

    @Test
    fun `parse arm ready command`() {
        val result = parser.parse("arm ready")
        assertEquals(ParsedCommand.ArmReady, result)
    }

    // Gripper commands
    @Test
    fun `parse open gripper command`() {
        val result = parser.parse("open gripper")
        assertTrue(result is ParsedCommand.Gripper)
        val gripper = result as ParsedCommand.Gripper
        assertEquals(1.0f, gripper.position, 0.001f)
    }

    @Test
    fun `parse close gripper command`() {
        val result = parser.parse("close the gripper")
        assertTrue(result is ParsedCommand.Gripper)
        val gripper = result as ParsedCommand.Gripper
        assertEquals(0.0f, gripper.position, 0.001f)
    }

    @Test
    fun `parse grip command`() {
        val result = parser.parse("grip")
        assertTrue(result is ParsedCommand.Gripper)
        val gripper = result as ParsedCommand.Gripper
        assertEquals(0.0f, gripper.position, 0.001f)
    }

    @Test
    fun `parse release command`() {
        val result = parser.parse("release")
        assertTrue(result is ParsedCommand.Gripper)
        val gripper = result as ParsedCommand.Gripper
        assertEquals(1.0f, gripper.position, 0.001f)
    }

    // Teaching commands
    @Test
    fun `parse teach command`() {
        val result = parser.parse("teach patrol")
        assertTrue(result is ParsedCommand.TeachStart)
        val teach = result as ParsedCommand.TeachStart
        assertEquals("patrol", teach.name)
    }

    @Test
    fun `parse done command`() {
        val result = parser.parse("done")
        assertEquals(ParsedCommand.TeachEnd, result)
    }

    @Test
    fun `parse cancel command`() {
        val result = parser.parse("cancel")
        assertEquals(ParsedCommand.TeachCancel, result)
    }

    // Known behaviors
    @Test
    fun `parse known behavior command`() {
        val parserWithBehaviors = CommandParser(setOf("patrol", "pickup"))
        val result = parserWithBehaviors.parse("run patrol")
        assertTrue(result is ParsedCommand.RunBehavior)
        val behavior = result as ParsedCommand.RunBehavior
        assertEquals("patrol", behavior.name)
    }

    // Look at commands
    @Test
    fun `parse look at command`() {
        val result = parser.parse("look at the cup")
        assertTrue(result is ParsedCommand.LookAt)
        val lookAt = result as ParsedCommand.LookAt
        assertEquals("cup", lookAt.target)
    }

    // Edge cases
    @Test
    fun `parse empty string returns null`() {
        val result = parser.parse("")
        assertNull(result)
    }

    @Test
    fun `parse whitespace returns null`() {
        val result = parser.parse("   ")
        assertNull(result)
    }

    @Test
    fun `parse unrecognized command returns null`() {
        val result = parser.parse("do something random")
        assertNull(result)
    }

    @Test
    fun `parse handles mixed case`() {
        val result = parser.parse("STOP")
        assertEquals(ParsedCommand.Stop, result)
    }

    @Test
    fun `parse handles punctuation`() {
        val result = parser.parse("Stop!")
        assertEquals(ParsedCommand.Stop, result)
    }

    // Priority tests
    @Test
    fun `stop takes priority over other commands`() {
        val result = parser.parse("forward and stop")
        assertEquals(ParsedCommand.Stop, result)
    }

    // Control command conversion
    @Test
    fun `stop converts to zero velocity command`() {
        val command = ParsedCommand.Stop.toControlCommand()
        assertNotNull(command)
    }

    @Test
    fun `gripper converts to control command`() {
        val command = ParsedCommand.Gripper(0.5f).toControlCommand()
        assertNotNull(command)
    }

    @Test
    fun `drive does not convert to control command`() {
        val command = ParsedCommand.Drive(0.5f, 0f, 0f).toControlCommand()
        assertNull(command)  // Drive goes to base, not arm
    }
}
