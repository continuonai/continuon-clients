package com.continuonxr.app.glove

import android.annotation.SuppressLint
import android.bluetooth.*
import android.content.Context
import android.os.Handler
import android.os.Looper
import com.continuonxr.app.config.GloveConfig
import kotlinx.serialization.Serializable
import java.util.UUID

/**
 * BLE ingest stub for Continuon Glove v0.
 * Handles MTU negotiation, frame parsing, and diagnostics.
 */
class GloveBleClient(
    private val context: Context,
    private val config: GloveConfig,
) {
    private val handler = Handler(Looper.getMainLooper())
    private val diagnosticsTracker = GloveDiagnosticsTracker(config.minMtu, config.targetSampleRateHz)
    private var gatt: BluetoothGatt? = null
    private var diagnosticsCallback: ((GloveDiagnostics) -> Unit)? = null
    private var frameCallback: ((GloveFrame) -> Unit)? = null
    private var mtuSatisfied: Boolean = false

    @SuppressLint("MissingPermission")
    fun connect(onFrame: (GloveFrame) -> Unit, onDiagnostics: (GloveDiagnostics) -> Unit) {
        frameCallback = onFrame
        diagnosticsCallback = onDiagnostics
        val adapter = (context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
        val device = adapter.bondedDevices.firstOrNull { it.name == config.bleDeviceName }
        device?.connectGatt(context, false, gattCallback)
    }

    @SuppressLint("MissingPermission")
    fun disconnect() {
        gatt?.close()
        gatt = null
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                gatt.requestMtu(config.minMtu)
                gatt.discoverServices()
                this@GloveBleClient.gatt = gatt
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                diagnosticsCallback?.invoke(diagnosticsTracker.snapshot())
            }
        }

        @SuppressLint("MissingPermission")
        override fun onMtuChanged(gatt: BluetoothGatt, mtu: Int, status: Int) {
            mtuSatisfied = mtu >= config.minMtu
            diagnosticsCallback?.invoke(diagnosticsTracker.onMtuNegotiated(mtu).snapshot())
            if (!mtuSatisfied) {
                gatt.disconnect()
            }
        }

        @SuppressLint("MissingPermission")
        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (!mtuSatisfied) return
            val characteristic = findDataCharacteristic(gatt) ?: return
            gatt.setCharacteristicNotification(characteristic, true)
            characteristic.descriptors.forEach {
                it.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                gatt.writeDescriptor(it)
            }
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
        ) {
            val payload = characteristic.value ?: return
            val now = System.nanoTime()
            val frame = GloveFrameParser.parse(payload, now) ?: GloveFrame(
                timestampNanos = now,
                flex = emptyList(),
                fsr = emptyList(),
                orientationQuat = emptyList(),
                accel = emptyList(),
                valid = false,
            )
            frameCallback?.invoke(frame)
            diagnosticsCallback?.invoke(diagnosticsTracker.onFrame(frame, now))
        }
    }

    private fun findDataCharacteristic(gatt: BluetoothGatt): BluetoothGattCharacteristic? {
        val serviceUuid = UUID.fromString(config.serviceUuid)
        val characteristicUuid = UUID.fromString(config.characteristicUuid)
        return gatt.getService(serviceUuid)?.getCharacteristic(characteristicUuid)
    }
}

@Serializable
data class GloveFrame(
    val timestampNanos: Long,
    val flex: List<Float>,          // size 5, normalized 0..1
    val fsr: List<Float>,           // size 8, normalized 0..1
    val orientationQuat: List<Float>, // size 4
    val accel: List<Float>,         // size 3, m/s^2
    val valid: Boolean = true,
    val sequence: Int? = null,
    val statusFlags: Int? = null,
    val sampleTimeMicros: Long? = null,
    val batteryMv: Int? = null,
    val temperatureC: Float? = null,
)

@Serializable
data class GloveDiagnostics(
    val mtu: Int,
    val sampleRateHz: Float,
    val dropCount: Int,
    val rssi: Int?,
)
