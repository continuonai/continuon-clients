package com.continuonxr.app.camera

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp
import com.continuonxr.app.nexa.Detection

/**
 * DetectionOverlay draws bounding boxes and labels for detected objects.
 *
 * Features:
 * - Color-coded boxes by confidence
 * - Labels with confidence percentage
 * - Semi-transparent overlay
 * - Scales to match camera preview
 */
@Composable
fun DetectionOverlay(
    detections: List<Detection>,
    modifier: Modifier = Modifier,
    boxColors: DetectionColors = DetectionColors.default()
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        detections.forEach { detection ->
            // Get color based on confidence
            val boxColor = when {
                detection.confidence >= 0.8f -> boxColors.highConfidence
                detection.confidence >= 0.5f -> boxColors.mediumConfidence
                else -> boxColors.lowConfidence
            }

            val box = detection.boundingBox

            // Scale bounding box to canvas size
            // Note: This assumes detections are in pixel coordinates matching original image
            // For production, you'd need to account for image vs preview aspect ratio
            val scaleX = canvasWidth / 640f  // Assuming 640px source width
            val scaleY = canvasHeight / 480f // Assuming 480px source height

            val scaledLeft = box.left * scaleX
            val scaledTop = box.top * scaleY
            val scaledRight = box.right * scaleX
            val scaledBottom = box.bottom * scaleY

            // Draw bounding box
            drawRect(
                color = boxColor,
                topLeft = Offset(scaledLeft, scaledTop),
                size = Size(scaledRight - scaledLeft, scaledBottom - scaledTop),
                style = Stroke(width = 3.dp.toPx())
            )

            // Draw label background
            val labelText = "${detection.label} ${(detection.confidence * 100).toInt()}%"
            val labelHeight = 24.dp.toPx()
            val labelPadding = 4.dp.toPx()

            drawRect(
                color = boxColor.copy(alpha = 0.7f),
                topLeft = Offset(scaledLeft, scaledTop - labelHeight),
                size = Size(
                    minOf(scaledRight - scaledLeft, labelText.length * 10.dp.toPx()),
                    labelHeight
                )
            )

            // Draw label text
            drawContext.canvas.nativeCanvas.apply {
                val paint = android.graphics.Paint().apply {
                    color = android.graphics.Color.WHITE
                    textSize = 14.dp.toPx()
                    isAntiAlias = true
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }

                drawText(
                    labelText,
                    scaledLeft + labelPadding,
                    scaledTop - labelPadding,
                    paint
                )
            }

            // Draw corner markers for better visibility
            val cornerLength = 15.dp.toPx()
            val cornerStroke = 4.dp.toPx()

            // Top-left corner
            drawLine(
                color = boxColor,
                start = Offset(scaledLeft, scaledTop),
                end = Offset(scaledLeft + cornerLength, scaledTop),
                strokeWidth = cornerStroke
            )
            drawLine(
                color = boxColor,
                start = Offset(scaledLeft, scaledTop),
                end = Offset(scaledLeft, scaledTop + cornerLength),
                strokeWidth = cornerStroke
            )

            // Top-right corner
            drawLine(
                color = boxColor,
                start = Offset(scaledRight, scaledTop),
                end = Offset(scaledRight - cornerLength, scaledTop),
                strokeWidth = cornerStroke
            )
            drawLine(
                color = boxColor,
                start = Offset(scaledRight, scaledTop),
                end = Offset(scaledRight, scaledTop + cornerLength),
                strokeWidth = cornerStroke
            )

            // Bottom-left corner
            drawLine(
                color = boxColor,
                start = Offset(scaledLeft, scaledBottom),
                end = Offset(scaledLeft + cornerLength, scaledBottom),
                strokeWidth = cornerStroke
            )
            drawLine(
                color = boxColor,
                start = Offset(scaledLeft, scaledBottom),
                end = Offset(scaledLeft, scaledBottom - cornerLength),
                strokeWidth = cornerStroke
            )

            // Bottom-right corner
            drawLine(
                color = boxColor,
                start = Offset(scaledRight, scaledBottom),
                end = Offset(scaledRight - cornerLength, scaledBottom),
                strokeWidth = cornerStroke
            )
            drawLine(
                color = boxColor,
                start = Offset(scaledRight, scaledBottom),
                end = Offset(scaledRight, scaledBottom - cornerLength),
                strokeWidth = cornerStroke
            )
        }
    }
}

/**
 * Color scheme for detections.
 */
data class DetectionColors(
    val highConfidence: Color,
    val mediumConfidence: Color,
    val lowConfidence: Color
) {
    companion object {
        fun default() = DetectionColors(
            highConfidence = Color(0xFF4CAF50),    // Green
            mediumConfidence = Color(0xFFFF9800),  // Orange
            lowConfidence = Color(0xFFF44336)      // Red
        )

        fun monochrome() = DetectionColors(
            highConfidence = Color.White,
            mediumConfidence = Color.LightGray,
            lowConfidence = Color.Gray
        )
    }
}

/**
 * Overlay showing a crosshair for targeting.
 */
@Composable
fun CrosshairOverlay(
    modifier: Modifier = Modifier,
    color: Color = Color.White.copy(alpha = 0.7f)
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        val centerX = size.width / 2
        val centerY = size.height / 2
        val crosshairSize = 30.dp.toPx()
        val strokeWidth = 2.dp.toPx()

        // Horizontal line
        drawLine(
            color = color,
            start = Offset(centerX - crosshairSize, centerY),
            end = Offset(centerX + crosshairSize, centerY),
            strokeWidth = strokeWidth
        )

        // Vertical line
        drawLine(
            color = color,
            start = Offset(centerX, centerY - crosshairSize),
            end = Offset(centerX, centerY + crosshairSize),
            strokeWidth = strokeWidth
        )

        // Center dot
        drawCircle(
            color = color,
            radius = 3.dp.toPx(),
            center = Offset(centerX, centerY)
        )
    }
}

/**
 * Grid overlay for alignment.
 */
@Composable
fun GridOverlay(
    modifier: Modifier = Modifier,
    gridColor: Color = Color.White.copy(alpha = 0.2f),
    divisions: Int = 3
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        val cellWidth = size.width / divisions
        val cellHeight = size.height / divisions

        // Vertical lines
        for (i in 1 until divisions) {
            drawLine(
                color = gridColor,
                start = Offset(cellWidth * i, 0f),
                end = Offset(cellWidth * i, size.height),
                strokeWidth = 1.dp.toPx()
            )
        }

        // Horizontal lines
        for (i in 1 until divisions) {
            drawLine(
                color = gridColor,
                start = Offset(0f, cellHeight * i),
                end = Offset(size.width, cellHeight * i),
                strokeWidth = 1.dp.toPx()
            )
        }
    }
}
