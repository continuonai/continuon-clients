"""
Hailo YOLOv5-seg Segmentation Worker - Subprocess-safe instance segmentation.

This worker runs YOLOv5-seg inference on the Hailo-8 NPU in a separate process
to prevent SDK instability from affecting the main runtime.

Protocol:
- Read JPEG bytes from stdin
- Run YOLOv5-seg inference on Hailo NPU
- Output JSON with detections and masks to stdout

Segmentation Output Format:
{
    "ok": true,
    "detections": [
        {
            "label": "person",
            "class_id": 0,
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2],
            "mask": [[x1,y1], [x2,y2], ...],  # polygon points
            "mask_area": 12345
        }
    ],
    "inference_time_ms": 15.2,
    "model": "yolov5n_seg_h8"
}
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _json_out(payload: Dict[str, Any]) -> None:
    """Write JSON to stdout."""
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.flush()


# COCO class names (same as detection)
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

# YOLOv5-seg anchor configuration (from yolov5seg.json)
ANCHORS = [
    [[116, 90], [156, 198], [373, 326]],  # stride 32
    [[30, 61], [62, 45], [59, 119]],       # stride 16
    [[10, 13], [16, 30], [33, 23]],        # stride 8
]
STRIDES = [32, 16, 8]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.6
) -> List[int]:
    """Non-Maximum Suppression."""
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


def process_mask(
    protos: np.ndarray,
    mask_coefs: np.ndarray,
    bbox: List[float],
    img_shape: Tuple[int, int],
    mask_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Process segmentation mask from prototype and coefficients.

    Args:
        protos: Prototype masks from segmentation head [32, H, W]
        mask_coefs: Mask coefficients for this detection [32]
        bbox: Bounding box [x1, y1, x2, y2]
        img_shape: Original image shape (height, width)
        mask_threshold: Threshold for binary mask

    Returns:
        Binary mask array of shape (H, W) or None if failed
    """
    try:
        # Compute mask from linear combination of prototypes
        # mask = sigmoid(sum(coef_i * proto_i))
        mask = sigmoid(np.tensordot(mask_coefs, protos, axes=([0], [0])))

        # Resize mask to image size
        from PIL import Image
        mask_h, mask_w = mask.shape
        img_h, img_w = img_shape

        # Scale mask to image size
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((img_w, img_h), Image.BILINEAR)
        mask = np.array(mask_img) / 255.0

        # Crop to bounding box region
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        # Create full mask and fill bbox region
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        bbox_mask = mask[y1:y2, x1:x2] > mask_threshold
        full_mask[y1:y2, x1:x2] = bbox_mask.astype(np.uint8) * 255

        return full_mask

    except Exception as e:
        return None


def decode_yolov5_seg_output(
    det_outputs: List[np.ndarray],
    seg_output: np.ndarray,
    img_width: int,
    img_height: int,
    input_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
    num_classes: int = 80,
    num_masks: int = 32,
) -> List[Dict[str, Any]]:
    """
    Decode YOLOv5-seg outputs into detections with masks.

    YOLOv5-seg output structure:
    - Detection heads: 3 outputs at different scales
      Each: [batch, anchors, grid_h, grid_w, 5 + num_classes + num_masks]
      Where 5 = [cx, cy, w, h, obj_conf]
    - Segmentation head: prototype masks [batch, num_masks, mask_h, mask_w]

    Args:
        det_outputs: List of detection head outputs
        seg_output: Segmentation prototype output
        img_width: Original image width
        img_height: Original image height
        input_size: Model input size
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        num_classes: Number of classes
        num_masks: Number of mask prototypes

    Returns:
        List of detection dicts with masks
    """
    all_boxes = []
    all_scores = []
    all_class_ids = []
    all_mask_coefs = []

    scale_x = img_width / input_size
    scale_y = img_height / input_size

    # Process each detection head
    for head_idx, det_out in enumerate(det_outputs):
        stride = STRIDES[head_idx]
        anchors = ANCHORS[head_idx]

        # Handle different output shapes
        if len(det_out.shape) == 5:
            det_out = det_out[0]  # Remove batch dim

        # Shape: [num_anchors, grid_h, grid_w, 5 + num_classes + num_masks]
        if len(det_out.shape) == 4:
            na, gh, gw, nc = det_out.shape
        else:
            continue

        expected_nc = 5 + num_classes + num_masks
        if nc != expected_nc:
            # Try to reshape or skip
            continue

        # Decode predictions
        for a_idx in range(na):
            anchor_w, anchor_h = anchors[a_idx]

            for gy in range(gh):
                for gx in range(gw):
                    pred = det_out[a_idx, gy, gx]

                    # Object confidence
                    obj_conf = sigmoid(pred[4])
                    if obj_conf < conf_threshold:
                        continue

                    # Class probabilities
                    class_probs = sigmoid(pred[5:5 + num_classes])
                    class_id = int(np.argmax(class_probs))
                    class_conf = class_probs[class_id]

                    # Final confidence
                    confidence = obj_conf * class_conf
                    if confidence < conf_threshold:
                        continue

                    # Decode box
                    cx = (sigmoid(pred[0]) * 2 - 0.5 + gx) * stride
                    cy = (sigmoid(pred[1]) * 2 - 0.5 + gy) * stride
                    w = (sigmoid(pred[2]) * 2) ** 2 * anchor_w
                    h = (sigmoid(pred[3]) * 2) ** 2 * anchor_h

                    # Convert to corner format and scale
                    x1 = (cx - w / 2) * scale_x
                    y1 = (cy - h / 2) * scale_y
                    x2 = (cx + w / 2) * scale_x
                    y2 = (cy + h / 2) * scale_y

                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Mask coefficients
                    mask_coefs = pred[5 + num_classes:5 + num_classes + num_masks]

                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(confidence)
                    all_class_ids.append(class_id)
                    all_mask_coefs.append(mask_coefs)

    if not all_boxes:
        return []

    # Apply NMS
    boxes_arr = np.array(all_boxes)
    scores_arr = np.array(all_scores)
    keep = nms(boxes_arr, scores_arr, iou_threshold)

    # Get segmentation prototypes
    if len(seg_output.shape) == 4:
        protos = seg_output[0]  # Remove batch dim [num_masks, H, W]
    else:
        protos = seg_output

    # Build final detections
    detections = []
    for idx in keep[:20]:  # Limit to 20 detections
        x1, y1, x2, y2 = all_boxes[idx]
        class_id = all_class_ids[idx]
        confidence = all_scores[idx]
        mask_coefs = all_mask_coefs[idx]

        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

        det = {
            "label": label,
            "class_id": class_id,
            "confidence": round(float(confidence), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        }

        # Process mask
        if protos is not None:
            mask = process_mask(
                protos,
                mask_coefs,
                [x1, y1, x2, y2],
                (img_height, img_width),
            )
            if mask is not None:
                # Convert mask to polygon or RLE for efficient JSON
                mask_area = int(np.sum(mask > 0))
                det["mask_area"] = mask_area

                # Store mask as base64-encoded PNG for efficiency
                # Or compute contour polygon
                try:
                    import cv2
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        # Get largest contour
                        largest = max(contours, key=cv2.contourArea)
                        # Simplify polygon
                        epsilon = 0.01 * cv2.arcLength(largest, True)
                        approx = cv2.approxPolyDP(largest, epsilon, True)
                        polygon = approx.reshape(-1, 2).tolist()
                        det["polygon"] = polygon
                except:
                    pass

        detections.append(det)

    # Sort by confidence
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return detections


def run_yolov5_seg_inference(
    jpeg_bytes: bytes,
    hef_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Run YOLOv5-seg inference on Hailo NPU.

    Args:
        jpeg_bytes: JPEG image bytes
        hef_path: Path to YOLOv5-seg HEF file
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold

    Returns:
        Segmentation results dict
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

    # Load HEF
    hef = hailo.HEF(str(hef_path))
    vdevice = hailo.VDevice()

    cfg_params = hailo.ConfigureParams.create_from_hef(
        hef, interface=hailo.HailoStreamInterface.PCIe
    )
    net_group = vdevice.configure(hef, cfg_params)[0]

    # Get input/output info
    input_infos = list(hef.get_input_vstream_infos())
    output_infos = list(hef.get_output_vstream_infos())

    if not input_infos:
        return {"ok": False, "error": "HEF missing input vstream"}

    inp0 = input_infos[0]
    inp_name = inp0.name

    # Get input shape
    inp_shape = tuple(int(x) for x in inp0.shape)
    if len(inp_shape) == 3:
        h, w, c = inp_shape
    elif len(inp_shape) == 4:
        _, h, w, c = inp_shape
    else:
        return {"ok": False, "error": f"Unsupported input shape: {inp_shape}"}

    # Resize image
    im_resized = im.resize((w, h), Image.BILINEAR)
    img_arr = np.array(im_resized, dtype=np.uint8)
    input_buffer = np.ascontiguousarray(img_arr.reshape((1,) + img_arr.shape))

    # Configure streams
    in_params = hailo.InputVStreamParams.make_from_network_group(
        net_group, format_type=hailo.FormatType.UINT8
    )
    out_params = hailo.OutputVStreamParams.make_from_network_group(
        net_group, format_type=hailo.FormatType.FLOAT32
    )

    # Run inference
    ng_params = net_group.create_params()
    with net_group.activate(ng_params):
        with hailo.InferVStreams(net_group, in_params, out_params) as infer:
            outputs = infer.infer({inp_name: input_buffer})

    inference_time_ms = (time.time() - start_time) * 1000

    # Parse outputs - YOLOv5-seg has 4 outputs:
    # 3 detection heads + 1 segmentation head
    output_names = [info.name for info in output_infos]

    # Identify outputs by name pattern or shape
    det_outputs = []
    seg_output = None

    for name in output_names:
        out_data = outputs.get(name)
        if out_data is None:
            continue

        out_arr = np.array(out_data, dtype=np.float32)

        # Segmentation output is typically smaller and has shape like [1, 32, 160, 160]
        # Detection outputs have shape like [1, 3, H, W, 117] (3 anchors, 117 = 5 + 80 + 32)
        if "seg" in name.lower() or "proto" in name.lower():
            seg_output = out_arr
        elif len(out_arr.shape) >= 4:
            # Check if it looks like a detection output
            if out_arr.shape[-1] > 100:  # 5 + num_classes + num_masks
                det_outputs.append(out_arr)
            elif out_arr.shape[1] == 32:  # Prototype mask dimension
                seg_output = out_arr
            else:
                det_outputs.append(out_arr)

    # If we couldn't identify outputs by name, try by shape
    if not det_outputs and seg_output is None:
        for name, out_data in outputs.items():
            out_arr = np.array(out_data, dtype=np.float32)
            if len(out_arr.shape) == 4 and out_arr.shape[1] == 32:
                seg_output = out_arr
            else:
                det_outputs.append(out_arr)

    # Decode outputs
    if det_outputs:
        detections = decode_yolov5_seg_output(
            det_outputs,
            seg_output,
            img_width=orig_width,
            img_height=orig_height,
            input_size=w,
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
        "num_outputs": len(outputs),
    }


def main() -> None:
    """Main entry point for subprocess worker."""
    # Get config from environment
    hef_path = Path(
        os.environ.get(
            "CONTINUON_HAILO_SEG_HEF",
            "/usr/share/hailo-models/yolov5n_seg_h8.hef"
        )
    )
    conf_threshold = float(os.environ.get("CONTINUON_HAILO_CONF", "0.25"))
    iou_threshold = float(os.environ.get("CONTINUON_HAILO_IOU", "0.6"))

    # Read JPEG from stdin
    jpeg_bytes = sys.stdin.buffer.read()
    if not jpeg_bytes:
        _json_out({"ok": False, "error": "No input JPEG bytes"})
        return

    try:
        result = run_yolov5_seg_inference(
            jpeg_bytes,
            hef_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        _json_out(result)
    except Exception as e:
        import traceback
        _json_out({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


if __name__ == "__main__":
    main()
