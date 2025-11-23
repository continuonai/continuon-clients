package com.continuonxr.app.glove

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GloveFrameParserTest {
    @Test
    fun parsesV1Format() {
        val raw = ByteArray(56) { 0 }
        val buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)

        buffer.put(0, 0x01)
        buffer.put(1, 0x03)
        buffer.putShort(2, 5)
        buffer.putInt(4, 123_456)

        val flexValues = listOf(0, 512, 1023, 256, 768)
        var offset = 8
        flexValues.forEach { value ->
            buffer.position(offset)
            buffer.putShort(value.toShort())
            offset += 2
        }

        val fsrValues = List(8) { index -> index * 100 }
        fsrValues.forEach { value ->
            buffer.position(offset)
            buffer.putShort(value.toShort())
            offset += 2
        }

        val quat = listOf(0, 0, 0, 10000) // w = 1.0 scaled by 1e4
        quat.forEach { value ->
            buffer.position(offset)
            buffer.putShort(value.toShort())
            offset += 2
        }

        val accelMg = listOf(1000, 0, -1000) // +/- 1 g
        accelMg.forEach { value ->
            buffer.position(offset)
            buffer.putShort(value.toShort())
            offset += 2
        }

        buffer.position(offset)
        buffer.putShort(2500.toShort()) // 25.00 C
        offset += 2
        buffer.position(offset)
        buffer.putShort(3700.toShort()) // 3.7V

        val frame = GloveFrameParser.parse(raw, timestampNanos = 1234L)

        assertNotNull(frame)
        frame!!

        assertEquals(5, frame.sequence)
        assertEquals(123_456L, frame.sampleTimeMicros)
        assertEquals(1.0f, frame.flex[2], 1e-3f) // 1023 normalized to 1.0
        assertEquals(0.098f, frame.fsr[1], 1e-3f) // 100 / 1023 ~= 0.0977
        assertEquals(1.0f, frame.orientationQuat[3], 1e-3f) // w normalized to 1.0
        assertEquals(9.80665f, frame.accel[0], 1e-3f) // 1000 mg -> 9.80665 m/s^2
        assertEquals(-9.80665f, frame.accel[2], 1e-3f)
        assertEquals(25.0f, frame.temperatureC)
        assertEquals(3700, frame.batteryMv)
        assertEquals(0x03, frame.statusFlags)
    }
}
