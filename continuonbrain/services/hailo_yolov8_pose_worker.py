"""
Hailo YOLOv8-Pose Worker - Subprocess-safe human pose estimation.

This worker runs YOLOv8-pose inference on the Hailo-8 NPU in a separate process
to prevent SDK instability from affecting the main runtime.

Protocol:
- Read JPEG bytes from stdin
- Run YOLOv8-pose inference on Hailo NPU
- Output JSON with poses and keypoints to stdout

Pose Output Format:
{
    "ok": true,
    "poses": [
        {
            "person_id": 0,
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2],
            "keypoints": [
                {"name": "nose", "x": 320, "y": 100, "conf": 0.95},
                {"name": "left_eye", "x": 310, "y": 90, "conf": 0.92},
                ...
            ]
        }
    ],
    "inference_time_ms": 12.5,
    "model": "yolov8s_pose"
}

COCO Keypoints (17):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _json_out(payload: Dict[str, Any]) -> None:
    """Write JSON to stdout."""
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.flush()


# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Skeleton connections for visualization (pairs of keypoint indices)
SKELETON = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # torso
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
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


def postprocess_yolov8_pose(
    outputs: Dict[str, Any],
    img_width: int,
    img_height: int,
    input_width: int = 640,
    input_height: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_keypoints: int = 17,
) -> List[Dict[str, Any]]:
    """
    Post-process YOLOv8-pose output.

    YOLOv8-pose output format varies by HEF configuration:
    - With NMS: nested list [batch][class][detections]
    - Without NMS: array [batch, 56, 8400] where 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints)

    Each keypoint: (x, y, confidence)
    """
    poses = []
    scale_x = img_width / input_width
    scale_y = img_height / input_height

    # Get the main output tensor
    out_data = None
    for key, val in outputs.items():
        if val is not None:
            out_data = val
            break

    if out_data is None:
        return []

    # Handle NMS-processed nested list format
    if isinstance(out_data, (list, tuple)):
        # NMS output: [batch][class][detections_array]
        # For pose, there's only 1 class (person), so [batch][0][dets]
        batch_data = out_data[0] if len(out_data) > 0 else []

        if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
            # Get person class detections (class 0)
            person_dets = batch_data[0] if len(batch_data) > 0 else []

            if isinstance(person_dets, np.ndarray) and person_dets.size > 0:
                # Each detection: [y1, x1, y2, x2, conf, kp0_x, kp0_y, kp0_c, ...]
                for det_idx, det in enumerate(person_dets):
                    if len(det) < 5:
                        continue

                    confidence = float(det[4])
                    if confidence < conf_threshold:
                        continue

                    # Bounding box (normalized)
                    y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]
                    bbox = [
                        round(x_min * img_width, 1),
                        round(y_min * img_height, 1),
                        round(x_max * img_width, 1),
                        round(y_max * img_height, 1),
                    ]

                    # Extract keypoints
                    keypoints = []
                    kp_start = 5
                    for kp_idx in range(num_keypoints):
                        kp_offset = kp_start + kp_idx * 3
                        if kp_offset + 2 < len(det):
                            kp_x = float(det[kp_offset]) * img_width
                            kp_y = float(det[kp_offset + 1]) * img_height
                            kp_conf = float(det[kp_offset + 2])
                        else:
                            kp_x, kp_y, kp_conf = 0.0, 0.0, 0.0

                        keypoints.append({
                            "name": KEYPOINT_NAMES[kp_idx],
                            "x": round(kp_x, 1),
                            "y": round(kp_y, 1),
                            "conf": round(kp_conf, 3),
                        })

                    poses.append({
                        "person_id": det_idx,
                        "confidence": round(confidence, 3),
                        "bbox": bbox,
                        "keypoints": keypoints,
                    })

        return poses

    # Handle raw array format
    if isinstance(out_data, np.ndarray):
        out_arr = out_data.astype(np.float32)

        # Remove batch dimension if present
        if len(out_arr.shape) == 3:
            out_arr = out_arr[0]

        # Expected shape: (56, 8400) or (8400, 56)
        # 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints) = 4 + 1 + 51 = 56
        expected_channels = 4 + 1 + num_keypoints * 3  # 56 for 17 keypoints

        if out_arr.shape[0] == expected_channels:
            out_arr = out_arr.T  # Transpose to (num_anchors, channels)

        if out_arr.shape[1] != expected_channels:
            # Try to handle different formats
            return []

        num_anchors = out_arr.shape[0]

        boxes_list = []
        scores_list = []
        keypoints_list = []

        for i in range(num_anchors):
            row = out_arr[i]

            # YOLOv8 format: cx, cy, w, h, conf, kp0_x, kp0_y, kp0_c, ...
            cx, cy, w, h = row[0:4]
            confidence = float(row[4])

            if confidence < conf_threshold:
                continue

            # Convert center format to corner format and scale
            x1 = max(0, (cx - w / 2) * scale_x)
            y1 = max(0, (cy - h / 2) * scale_y)
            x2 = min(img_width, (cx + w / 2) * scale_x)
            y2 = min(img_height, (cy + h / 2) * scale_y)

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract keypoints
            keypoints = []
            for kp_idx in range(num_keypoints):
                kp_offset = 5 + kp_idx * 3
                kp_x = float(row[kp_offset]) * scale_x
                kp_y = float(row[kp_offset + 1]) * scale_y
                kp_conf = float(row[kp_offset + 2])

                keypoints.append({
                    "name": KEYPOINT_NAMES[kp_idx],
                    "x": round(kp_x, 1),
                    "y": round(kp_y, 1),
                    "conf": round(kp_conf, 3),
                })

            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(confidence)
            keypoints_list.append(keypoints)

        if not boxes_list:
            return []

        # Apply NMS
        boxes_arr = np.array(boxes_list)
        scores_arr = np.array(scores_list)
        keep = nms(boxes_arr, scores_arr, iou_threshold)

        for idx, keep_idx in enumerate(keep):
            x1, y1, x2, y2 = boxes_list[keep_idx]
            poses.append({
                "person_id": idx,
                "confidence": round(scores_list[keep_idx], 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "keypoints": keypoints_list[keep_idx],
            })

    # Sort by confidence
    poses.sort(key=lambda x: x["confidence"], reverse=True)

    return poses


def run_yolov8_pose_inference(
    jpeg_bytes: bytes,
    hef_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Run YOLOv8-pose inference on Hailo NPU.

    Args:
        jpeg_bytes: JPEG image bytes
        hef_path: Path to YOLOv8-pose HEF file
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold

    Returns:
        Pose results dict
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

    if not input_infos:
        return {"ok": False, "error": "HEF missing input vstream info"}

    inp0 = input_infos[0]
    inp_name = inp0.name

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
    img_arr = np.array(im_resized, dtype=np.uint8)

    # Prepare input buffer with batch dimension
    input_buffer = np.ascontiguousarray(img_arr.reshape((1,) + img_arr.shape))

    # Setup stream parameters
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

    # Post-process outputs
    poses = postprocess_yolov8_pose(
        outputs,
        img_width=orig_width,
        img_height=orig_height,
        input_width=w,
        input_height=h,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    # Extract wrist positions for AINA integration
    wrist_positions = []
    for pose in poses:
        left_wrist = next((kp for kp in pose["keypoints"] if kp["name"] == "left_wrist"), None)
        right_wrist = next((kp for kp in pose["keypoints"] if kp["name"] == "right_wrist"), None)

        if left_wrist and left_wrist["conf"] > 0.3:
            wrist_positions.append({
                "hand": "left",
                "x": left_wrist["x"],
                "y": left_wrist["y"],
                "conf": left_wrist["conf"],
                "person_id": pose["person_id"],
            })
        if right_wrist and right_wrist["conf"] > 0.3:
            wrist_positions.append({
                "hand": "right",
                "x": right_wrist["x"],
                "y": right_wrist["y"],
                "conf": right_wrist["conf"],
                "person_id": pose["person_id"],
            })

    return {
        "ok": True,
        "poses": poses,
        "wrist_positions": wrist_positions,
        "inference_time_ms": round(inference_time_ms, 2),
        "model": hef_path.stem,
        "input_size": [w, h],
        "image_size": [orig_width, orig_height],
        "num_poses": len(poses),
        "skeleton": SKELETON,
    }


def main() -> None:
    """Main entry point for subprocess worker."""
    parser = argparse.ArgumentParser(description="YOLOv8-Pose Hailo Worker")
    parser.add_argument(
        "--hef",
        type=str,
        default=os.environ.get(
            "CONTINUON_HAILO_POSE_HEF",
            "/usr/share/hailo-models/yolov8s_pose_h8.hef"
        ),
        help="Path to YOLOv8-pose HEF file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=float(os.environ.get("CONTINUON_HAILO_CONF", "0.25")),
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=float(os.environ.get("CONTINUON_HAILO_IOU", "0.45")),
        help="NMS IoU threshold"
    )
    args = parser.parse_args()

    hef_path = Path(args.hef)

    # Read JPEG from stdin
    jpeg_bytes = sys.stdin.buffer.read()
    if not jpeg_bytes:
        _json_out({"ok": False, "error": "No input JPEG bytes"})
        return

    try:
        result = run_yolov8_pose_inference(
            jpeg_bytes,
            hef_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        _json_out(result)
    except Exception as e:
        import traceback
        _json_out({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


if __name__ == "__main__":
    main()
