"""
Hailo YOLOv8 Detection Worker - Subprocess-safe object detection.

This worker runs YOLOv8 inference on the Hailo-8 NPU in a separate process
to prevent SDK instability from affecting the main runtime.

Protocol:
- Read JPEG bytes from stdin
- Run YOLOv8 inference on Hailo NPU
- Output JSON with detections to stdout

Detection Output Format:
{
    "ok": true,
    "detections": [
        {
            "label": "person",
            "class_id": 0,
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2]  # pixel coordinates
        }
    ],
    "inference_time_ms": 12.5,
    "model": "yolov8s"
}
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np


def _json_out(payload: Dict[str, Any]) -> None:
    """Write JSON to stdout."""
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.flush()


# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Non-Maximum Suppression.

    Args:
        boxes: Array of [x1, y1, x2, y2] boxes
        scores: Confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess_yolov8_nms(
    output: Any,
    img_width: int,
    img_height: int,
    input_width: int = 640,
    input_height: int = 640,
    conf_threshold: float = 0.25,
    num_classes: int = 80,
) -> List[Dict[str, Any]]:
    """
    Post-process YOLOv8 NMS-processed output.

    Hailo YOLOv8 HEF with built-in NMS outputs a nested structure:
    - List[batch] -> List[classes] -> ndarray(num_dets, 5)
    - Each detection: [y_min, x_min, y_max, x_max, confidence] normalized 0-1

    Args:
        output: NMS-processed model output (nested list or array)
        img_width: Original image width
        img_height: Original image height
        input_width: Model input width
        input_height: Model input height
        conf_threshold: Confidence threshold
        num_classes: Number of classes

    Returns:
        List of detection dicts
    """
    detections = []

    # Handle nested list structure: [batch][class][detections]
    if isinstance(output, (list, tuple)):
        # Get first batch
        batch_data = output[0] if len(output) > 0 else []

        if isinstance(batch_data, (list, tuple)):
            # Iterate over classes
            for class_id, class_dets in enumerate(batch_data):
                if class_id >= num_classes:
                    break

                # class_dets is ndarray of shape (num_dets, 5)
                if isinstance(class_dets, np.ndarray) and class_dets.size > 0:
                    for det in class_dets:
                        if len(det) < 5:
                            continue

                        confidence = float(det[4])
                        if confidence < conf_threshold:
                            continue

                        # Coordinates: [y1, x1, y2, x2] normalized 0-1
                        y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]

                        # Scale to original image size
                        x1 = max(0, x_min * img_width)
                        y1 = max(0, y_min * img_height)
                        x2 = min(img_width, x_max * img_width)
                        y2 = min(img_height, y_max * img_height)

                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1:
                            continue

                        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

                        detections.append({
                            "label": label,
                            "class_id": class_id,
                            "confidence": round(confidence, 4),
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        })

    # Fallback for array format
    elif isinstance(output, np.ndarray):
        # Handle batch dimension
        if len(output.shape) == 4:
            output = output[0]

        if len(output.shape) == 3:
            n_classes, n_coords, max_dets = output.shape

            for class_id in range(min(n_classes, num_classes)):
                class_output = output[class_id]

                for det_idx in range(max_dets):
                    det = class_output[:, det_idx]
                    confidence = float(det[4]) if len(det) > 4 else 0

                    if confidence < conf_threshold:
                        continue

                    y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]
                    x1 = max(0, x_min * img_width)
                    y1 = max(0, y_min * img_height)
                    x2 = min(img_width, x_max * img_width)
                    y2 = min(img_height, y_max * img_height)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

                    detections.append({
                        "label": label,
                        "class_id": class_id,
                        "confidence": round(confidence, 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    })

    # Sort by confidence
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return detections


def postprocess_yolov8_raw(
    output: np.ndarray,
    img_width: int,
    img_height: int,
    input_width: int = 640,
    input_height: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 80,
) -> List[Dict[str, Any]]:
    """
    Post-process raw YOLOv8 output (without NMS).

    YOLOv8 raw output format: [batch, num_classes + 4, num_anchors]
    - First 4 values: cx, cy, w, h
    - Next num_classes values: class probabilities

    Args:
        output: Raw model output
        img_width: Original image width
        img_height: Original image height
        input_width: Model input width
        input_height: Model input height
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        num_classes: Number of classes

    Returns:
        List of detection dicts
    """
    # Handle different output shapes
    if len(output.shape) == 3:
        output = output[0]
    elif len(output.shape) == 1:
        expected_anchors = 8400
        expected_channels = 4 + num_classes
        if output.size == expected_anchors * expected_channels:
            output = output.reshape(expected_channels, expected_anchors)
        else:
            return []

    if output.shape[0] == 4 + num_classes:
        output = output.T

    num_anchors = output.shape[0]
    scale_x = img_width / input_width
    scale_y = img_height / input_height

    boxes_list = []
    scores_list = []
    class_ids_list = []

    for i in range(num_anchors):
        row = output[i]
        cx, cy, w, h = row[:4]
        class_scores = row[4:4 + num_classes]

        if len(class_scores) == 0:
            continue

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < conf_threshold:
            continue

        x1 = max(0, min((cx - w / 2) * scale_x, img_width))
        y1 = max(0, min((cy - h / 2) * scale_y, img_height))
        x2 = max(0, min((cx + w / 2) * scale_x, img_width))
        y2 = max(0, min((cy + h / 2) * scale_y, img_height))

        boxes_list.append([x1, y1, x2, y2])
        scores_list.append(confidence)
        class_ids_list.append(class_id)

    if not boxes_list:
        return []

    boxes_arr = np.array(boxes_list)
    scores_arr = np.array(scores_list)
    keep = nms(boxes_arr, scores_arr, iou_threshold)

    detections = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_list[idx]
        class_id = class_ids_list[idx]
        confidence = scores_list[idx]
        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        detections.append({
            "label": label,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    return detections


def run_yolov8_inference(
    jpeg_bytes: bytes,
    hef_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Run YOLOv8 inference on Hailo NPU.

    Args:
        jpeg_bytes: JPEG image bytes
        hef_path: Path to YOLOv8 HEF file
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold

    Returns:
        Detection results dict
    """
    try:
        import hailo_platform as hailo
    except ImportError as e:
        return {"ok": False, "error": f"hailo_platform import failed: {e}"}

    from PIL import Image

    if not hef_path.exists():
        return {"ok": False, "error": f"HEF not found: {hef_path}"}

    start_time = time.time()

    # Load and decode image
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    orig_width, orig_height = im.size

    # Load HEF and configure
    hef = hailo.HEF(str(hef_path))
    vdevice = hailo.VDevice()

    cfg_params = hailo.ConfigureParams.create_from_hef(
        hef, interface=hailo.HailoStreamInterface.PCIe
    )
    net_group = vdevice.configure(hef, cfg_params)[0]

    # Get input/output info
    input_infos = list(hef.get_input_vstream_infos())
    output_infos = list(hef.get_output_vstream_infos())

    if not input_infos or not output_infos:
        return {"ok": False, "error": "HEF missing vstream infos"}

    inp0 = input_infos[0]
    out0 = output_infos[0]
    inp_name = inp0.name
    out_name = out0.name

    # Determine input size from HEF
    inp_shape = tuple(int(x) for x in inp0.shape)
    if len(inp_shape) == 3:
        h, w, c = inp_shape
    elif len(inp_shape) == 4:
        _, h, w, c = inp_shape
    else:
        return {"ok": False, "error": f"Unsupported input shape: {inp_shape}"}

    # Resize image to model input size
    im_resized = im.resize((w, h), Image.BILINEAR)
    img_arr = np.array(im_resized, dtype=np.uint8)  # np.array makes writable copy

    # Prepare input buffer with batch dimension
    input_buffer = np.ascontiguousarray(img_arr.reshape((1,) + img_arr.shape))

    # Determine format types
    inp_type = getattr(getattr(inp0, "format", None), "type", None)
    out_type = getattr(getattr(out0, "format", None), "type", None)

    in_format_type = hailo.FormatType.UINT8
    out_format_type = hailo.FormatType.FLOAT32
    if str(out_type).endswith("UINT8"):
        out_format_type = hailo.FormatType.UINT8

    in_params = hailo.InputVStreamParams.make_from_network_group(
        net_group, format_type=in_format_type
    )
    out_params = hailo.OutputVStreamParams.make_from_network_group(
        net_group, format_type=out_format_type
    )

    # Run inference
    ng_params = net_group.create_params()
    with net_group.activate(ng_params):
        with hailo.InferVStreams(net_group, in_params, out_params) as infer:
            outputs = infer.infer({inp_name: input_buffer})

    # Get output
    out_data = outputs.get(out_name)
    if out_data is None:
        # Fall back to first key
        key0 = next(iter(outputs.keys()))
        out_data = outputs[key0]

    inference_time_ms = (time.time() - start_time) * 1000

    # Detect output format and use appropriate postprocessor
    # Hailo NMS output is a nested list: [batch][class][detections_array]
    # Raw output is numpy array: (batch, 84, 8400) or similar

    if isinstance(out_data, (list, tuple)):
        # NMS-processed nested list format
        detections = postprocess_yolov8_nms(
            out_data,
            img_width=orig_width,
            img_height=orig_height,
            input_width=w,
            input_height=h,
            conf_threshold=conf_threshold,
        )
    elif isinstance(out_data, np.ndarray):
        out_arr = out_data.astype(np.float32)

        if len(out_arr.shape) >= 3:
            shape = out_arr.shape[-3:]
            if shape[0] == 80 and shape[1] == 5:
                # NMS-processed array format
                detections = postprocess_yolov8_nms(
                    out_arr,
                    img_width=orig_width,
                    img_height=orig_height,
                    input_width=w,
                    input_height=h,
                    conf_threshold=conf_threshold,
                )
            else:
                # Raw format
                detections = postprocess_yolov8_raw(
                    out_arr,
                    img_width=orig_width,
                    img_height=orig_height,
                    input_width=w,
                    input_height=h,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
        else:
            detections = postprocess_yolov8_raw(
                out_arr,
                img_width=orig_width,
                img_height=orig_height,
                input_width=w,
                input_height=h,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
    else:
        detections = []

    return {
        "ok": True,
        "detections": detections,
        "inference_time_ms": round(inference_time_ms, 2),
        "model": hef_path.stem,
        "input_size": [w, h],
        "image_size": [orig_width, orig_height],
        "num_detections": len(detections),
    }


def main() -> None:
    """Main entry point for subprocess worker."""
    # Get config from environment
    hef_path = Path(
        os.environ.get(
            "CONTINUON_HAILO_HEF",
            "/opt/continuonos/brain/model/base_model/yolov8s.hef"
        )
    )
    conf_threshold = float(os.environ.get("CONTINUON_HAILO_CONF", "0.25"))
    iou_threshold = float(os.environ.get("CONTINUON_HAILO_IOU", "0.45"))

    # Read JPEG from stdin
    jpeg_bytes = sys.stdin.buffer.read()
    if not jpeg_bytes:
        _json_out({"ok": False, "error": "No input JPEG bytes"})
        return

    try:
        result = run_yolov8_inference(
            jpeg_bytes,
            hef_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        _json_out(result)
    except Exception as e:
        _json_out({"ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
