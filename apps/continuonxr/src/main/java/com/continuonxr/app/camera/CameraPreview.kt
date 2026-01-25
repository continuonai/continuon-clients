package com.continuonxr.app.camera

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.view.ViewGroup
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.continuonxr.app.nexa.CameraFrame
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

/**
 * CameraPreview composable that displays live camera feed using CameraX.
 *
 * @param modifier Modifier for the preview container
 * @param onFrameAvailable Callback when a new frame is available for processing
 * @param enableFrameAnalysis Whether to analyze frames (for object detection)
 * @param frameInterval Minimum interval between frame callbacks in ms (for rate limiting)
 */
@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    onFrameAvailable: ((CameraFrame) -> Unit)? = null,
    enableFrameAnalysis: Boolean = false,
    frameInterval: Long = 500L
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var previewView by remember { mutableStateOf<PreviewView?>(null) }

    Box(modifier = modifier) {
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    previewView = this
                }
            },
            modifier = Modifier.fillMaxSize()
        )
    }

    LaunchedEffect(previewView, enableFrameAnalysis) {
        previewView?.let { view ->
            setupCamera(
                context = context,
                lifecycleOwner = lifecycleOwner,
                previewView = view,
                onFrameAvailable = if (enableFrameAnalysis) onFrameAvailable else null,
                frameInterval = frameInterval
            )
        }
    }
}

private suspend fun setupCamera(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    onFrameAvailable: ((CameraFrame) -> Unit)?,
    frameInterval: Long
) {
    val cameraProvider = context.getCameraProvider()
    val executor = ContextCompat.getMainExecutor(context)

    val preview = Preview.Builder()
        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
        .build()
        .also { it.setSurfaceProvider(previewView.surfaceProvider) }

    val cameraSelector = CameraSelector.Builder()
        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
        .build()

    val useCases = mutableListOf<UseCase>(preview)

    // Add image analysis if frame callback is provided
    if (onFrameAvailable != null) {
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        val frameAnalyzer = FrameAnalyzer(
            onFrameAvailable = onFrameAvailable,
            minIntervalMs = frameInterval
        )

        imageAnalysis.setAnalyzer(Dispatchers.Default.asExecutor(), frameAnalyzer)
        useCases.add(imageAnalysis)
    }

    try {
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            *useCases.toTypedArray()
        )
        Log.d(TAG, "Camera bound successfully with ${useCases.size} use cases")
    } catch (e: Exception) {
        Log.e(TAG, "Camera binding failed", e)
    }
}

/**
 * Analyzes camera frames and converts them to CameraFrame for processing.
 */
private class FrameAnalyzer(
    private val onFrameAvailable: (CameraFrame) -> Unit,
    private val minIntervalMs: Long
) : ImageAnalysis.Analyzer {

    private var lastAnalysisTime = 0L

    override fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // Rate limiting
        if (currentTime - lastAnalysisTime < minIntervalMs) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap()
            if (bitmap != null) {
                val frame = CameraFrame(
                    bitmap = bitmap,
                    timestampNanos = System.nanoTime(),
                    width = imageProxy.width,
                    height = imageProxy.height
                )
                onFrameAvailable(frame)
                lastAnalysisTime = currentTime
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame analysis failed", e)
        } finally {
            imageProxy.close()
        }
    }
}

/**
 * Extension to convert ImageProxy to Bitmap.
 */
private fun ImageProxy.toBitmap(): Bitmap? {
    val planes = planes
    if (planes.isEmpty()) return null

    val buffer = planes[0].buffer
    val pixelStride = planes[0].pixelStride
    val rowStride = planes[0].rowStride
    val rowPadding = rowStride - pixelStride * width

    val bitmap = Bitmap.createBitmap(
        width + rowPadding / pixelStride,
        height,
        Bitmap.Config.ARGB_8888
    )
    bitmap.copyPixelsFromBuffer(buffer)

    // Crop to actual dimensions if needed
    return if (rowPadding > 0) {
        Bitmap.createBitmap(bitmap, 0, 0, width, height)
    } else {
        bitmap
    }
}

/**
 * Get CameraProvider suspending function.
 */
private suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { future ->
            future.addListener(
                { continuation.resume(future.get()) },
                ContextCompat.getMainExecutor(this)
            )
        }
    }

/**
 * State holder for camera preview.
 */
class CameraPreviewState {
    var latestFrame: CameraFrame? by mutableStateOf(null)
        private set

    var isCapturing: Boolean by mutableStateOf(false)
        private set

    fun updateFrame(frame: CameraFrame) {
        latestFrame = frame
    }

    fun startCapture() {
        isCapturing = true
    }

    fun stopCapture() {
        isCapturing = false
        latestFrame = null
    }
}

@Composable
fun rememberCameraPreviewState(): CameraPreviewState {
    return remember { CameraPreviewState() }
}

private const val TAG = "CameraPreview"
