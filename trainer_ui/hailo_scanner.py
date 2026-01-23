"""
Hailo-8 NPU Scanner - Fast Neural Segmentation

Uses Hailo-8 NPU (26 TOPS) for real-time object detection and segmentation.
Much faster than CPU-based SAM3 (~45ms vs 10+ minutes).

Models available:
- yolov5n_seg_h8.hef: YOLOv5 nano segmentation (COCO classes)
- yolov8s_h8.hef: YOLOv8 small detection

Usage:
    from hailo_scanner import HailoScanner, scan_with_hailo

    scanner = HailoScanner()
    results = scanner.detect(image)
"""

import json
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import hailo_platform as hp
    HAS_HAILO = True
except ImportError:
    HAS_HAILO = False
    print("Hailo platform not available")


# COCO class names for YOLOv5/v8
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Map COCO classes to room asset types
COCO_TO_ASSET = {
    'chair': 'chair',
    'couch': 'couch',
    'potted plant': 'plant',
    'bed': 'furniture',
    'dining table': 'table',
    'tv': 'decoration',
    'laptop': 'decoration',
    'refrigerator': 'furniture',
    'clock': 'decoration',
    'vase': 'decoration',
    'book': 'decoration',
    'bottle': 'decoration',
    'cup': 'decoration',
    'bowl': 'decoration',
    'bench': 'furniture',
    'suitcase': 'furniture',
    'backpack': 'decoration',
    'umbrella': 'decoration',
    'handbag': 'decoration',
    'person': 'obstacle',  # People are obstacles for robot navigation
}

# Default colors for asset types
ASSET_COLORS = {
    'chair': 0x654321,
    'couch': 0x4a6fa5,
    'plant': 0x228b22,
    'table': 0x8b4513,
    'furniture': 0x805d40,
    'decoration': 0x9370db,
    'obstacle': 0xdc2626,
    'lamp': 0xffd700,
}


@dataclass
class HailoDetection:
    """A detection from Hailo inference."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    mask: Optional[np.ndarray] = None


@dataclass
class HailoScanResult:
    """Result from Hailo scanner."""
    scan_id: str
    timestamp: str
    detections: List[HailoDetection]
    assets: List[Dict[str, Any]]
    inference_time_ms: float
    model: str
    device: str


class HailoScanner:
    """
    Fast object detection using Hailo-8 NPU.
    """

    # Model paths
    MODELS = {
        'yolov5_seg': '/usr/share/hailo-models/yolov5n_seg_h8.hef',
        'yolov8_det': '/usr/share/hailo-models/yolov8s_h8.hef',
        'yolov6_det': '/usr/share/hailo-models/yolov6n_h8.hef',
        'fast_sam': '/tmp/fastsam/fast_sam_s.hef',  # FastSAM for zero-shot segmentation
    }

    def __init__(self, model_name: str = 'yolov5_seg'):
        self.model_name = model_name
        self.model_path = self.MODELS.get(model_name)

        if not HAS_HAILO:
            raise RuntimeError("Hailo platform not available")

        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        self.hef = hp.HEF(self.model_path)
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()

        # Input shape (H, W, C)
        self.input_shape = self.input_info.shape
        self.input_height = self.input_shape[0]
        self.input_width = self.input_shape[1]

        # Create device and configure
        self.target = hp.VDevice()
        configure_params = hp.ConfigureParams.create_from_hef(
            self.hef, interface=hp.HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_params = self.network_group.create_params()

        # Store quantization info for each output
        self.quant_params = {}
        for info in self.output_infos:
            self.quant_params[info.name] = {
                'scale': info.quant_info.qp_scale,
                'zero_point': info.quant_info.qp_zp
            }

        # Detection parameters (low threshold for room scanning)
        self.conf_threshold = 0.10
        self.iou_threshold = 0.45

        print(f"[HailoScanner] Loaded {model_name}")
        print(f"[HailoScanner] Input: {self.input_shape}")
        print(f"[HailoScanner] Outputs: {len(self.output_infos)}")
        for name, qp in self.quant_params.items():
            print(f"  {name}: scale={qp['scale']:.6f}, zp={qp['zero_point']:.1f}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required for preprocessing")

        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return resized.astype(np.uint8)

    def inference(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on preprocessed image."""
        # Ensure correct shape
        if image.shape != self.input_shape:
            image = self.preprocess(image)

        # Add batch dimension
        input_data = {self.input_info.name: np.expand_dims(image, 0)}

        # Run inference
        with self.network_group.activate(self.network_params):
            input_params = hp.InputVStreamParams.make(self.network_group, quantized=False)
            output_params = hp.OutputVStreamParams.make(self.network_group, quantized=False)

            with hp.InferVStreams(self.network_group, input_params, output_params) as pipeline:
                results = pipeline.infer(input_data)

        return results

    def postprocess_yolov5_seg(
        self,
        outputs: Dict[str, np.ndarray],
        orig_width: int,
        orig_height: int,
        debug: bool = False
    ) -> List[HailoDetection]:
        """Post-process YOLOv5 segmentation outputs."""
        detections = []
        max_conf_seen = 0.0
        total_checked = 0

        # YOLOv5 seg outputs:
        # conv48: (1, 80, 80, 351) - large objects
        # conv55: (1, 40, 40, 351) - medium objects
        # conv61: (1, 20, 20, 351) - small objects
        # conv63: (1, 160, 160, 32) - segmentation protos

        # For simplicity, we'll extract detections from the detection heads
        # 351 = 3 anchors * (5 + 80 + 32) = 3 * 117
        # 5 = x, y, w, h, conf
        # 80 = class probs
        # 32 = mask coefficients

        for output_name, output in outputs.items():
            if 'conv63' in output_name:
                continue  # Skip proto masks for now

            output = output[0]  # Remove batch dim
            h, w, c = output.shape

            # Dequantize the output
            qp = self.quant_params.get(output_name, {'scale': 1.0, 'zero_point': 0.0})
            output = (output.astype(np.float32) - qp['zero_point']) * qp['scale']

            if debug:
                print(f"[HailoScanner] Processing {output_name}: ({h}, {w}, {c})")
                print(f"  Dequantized: min={output.min():.3f}, max={output.max():.3f}")

            # Reshape to (h*w*3, 117)
            num_anchors = 3
            num_attrs = c // num_anchors  # Should be 117

            output = output.reshape(-1, num_attrs)
            total_checked += len(output)

            for det in output:
                # Extract detection info
                x, y, w_box, h_box, conf = det[:5]
                class_probs = det[5:85]
                # mask_coeffs = det[85:117]  # For segmentation

                # Apply sigmoid to confidence
                conf = 1 / (1 + np.exp(-conf))
                max_conf_seen = max(max_conf_seen, conf)

                if conf < self.conf_threshold:
                    continue

                # Get class with highest probability
                class_probs = 1 / (1 + np.exp(-class_probs))
                class_id = np.argmax(class_probs)
                class_conf = class_probs[class_id]

                final_conf = conf * class_conf

                # Track best detection for debugging
                if final_conf > 0.1 and debug:
                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                    print(f"    Candidate: {class_name} obj={conf:.2f} cls={class_conf:.2f} final={final_conf:.2f}")

                if final_conf < self.conf_threshold:
                    continue

                # Convert to bbox (normalized)
                x = 1 / (1 + np.exp(-x))
                y = 1 / (1 + np.exp(-y))
                w_box = 1 / (1 + np.exp(-w_box))
                h_box = 1 / (1 + np.exp(-h_box))

                x1 = max(0, x - w_box / 2)
                y1 = max(0, y - h_box / 2)
                x2 = min(1, x + w_box / 2)
                y2 = min(1, y + h_box / 2)

                if class_id < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[class_id]
                else:
                    class_name = f"class_{class_id}"

                detections.append(HailoDetection(
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(final_conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2))
                ))

        # Apply NMS
        detections = self._nms(detections)

        if debug:
            print(f"[HailoScanner] Checked {total_checked} candidates")
            print(f"[HailoScanner] Max confidence seen: {max_conf_seen:.4f}")
            print(f"[HailoScanner] Detections after NMS: {len(detections)}")

        return detections

    def _nms(self, detections: List[HailoDetection]) -> List[HailoDetection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._iou(best.bbox, d.bbox) < self.iou_threshold
            ]

        return keep

    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def detect(self, image: np.ndarray, debug: bool = False) -> List[HailoDetection]:
        """
        Detect objects in image.

        Args:
            image: BGR image (any size)
            debug: Print debug info

        Returns:
            List of detections
        """
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        preprocessed = self.preprocess(image)

        # Inference
        outputs = self.inference(preprocessed)

        if debug:
            print(f"[HailoScanner] Output keys: {list(outputs.keys())}")
            for name, arr in outputs.items():
                print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
                      f"min={arr.min():.3f}, max={arr.max():.3f}")

        # Postprocess based on model type
        if 'fast_sam' in self.model_name:
            detections = self.postprocess_fast_sam(outputs, orig_w, orig_h, debug=debug)
        elif 'seg' in self.model_name:
            detections = self.postprocess_yolov5_seg(outputs, orig_w, orig_h, debug=debug)
        else:
            detections = self.postprocess_yolov5_seg(outputs, orig_w, orig_h, debug=debug)

        return detections

    def postprocess_fast_sam(
        self,
        outputs: Dict[str, np.ndarray],
        orig_width: int,
        orig_height: int,
        debug: bool = False
    ) -> List[HailoDetection]:
        """
        Post-process FastSAM outputs.

        FastSAM uses YOLOv8-seg architecture with DFL (Distribution Focal Loss):
        - 64 channels = 4 * 16 (DFL-encoded bbox: x, y, w, h each with 16-bin distribution)
        - 1 channel = objectness (FastSAM detects 1 class: "object")
        - 32 channels = mask coefficients
        - 160x160x32 = prototype masks
        """
        detections = []
        max_conf_seen = 0.0

        # Group outputs by scale
        scales = {}
        proto_output = None

        for name, output in outputs.items():
            output = output[0]  # Remove batch dim
            h, w, c = output.shape

            if debug:
                print(f"[FastSAM] Output {name}: ({h}, {w}, {c})")

            # Prototype mask (160x160x32)
            if h == 160 and c == 32:
                proto_output = output
                continue

            # Group by grid size
            grid_key = h
            if grid_key not in scales:
                scales[grid_key] = {}

            # Dequantize
            qp = self.quant_params.get(name, {'scale': 1.0, 'zero_point': 0.0})
            output = (output.astype(np.float32) - qp['zero_point']) * qp['scale']

            if c == 64:
                scales[grid_key]['bbox'] = output
            elif c == 1:
                scales[grid_key]['obj'] = output
            elif c == 32:
                scales[grid_key]['mask'] = output

        if debug:
            print(f"[FastSAM] Scales: {list(scales.keys())}")
            print(f"[FastSAM] Proto: {proto_output.shape if proto_output is not None else None}")

        # DFL bins for bbox decoding
        dfl_bins = 16
        dfl_range = np.arange(dfl_bins, dtype=np.float32)

        # Process each scale with vectorized operations
        for grid_size, data in scales.items():
            if 'bbox' not in data or 'obj' not in data:
                continue

            bbox_data = data['bbox']  # (h, w, 64)
            obj_data = data['obj']    # (h, w, 1)
            h, w, _ = bbox_data.shape
            stride = 640 // h

            if debug:
                print(f"[FastSAM] Processing scale {grid_size}x{grid_size}, stride={stride}")
                print(f"  Bbox: min={bbox_data.min():.3f}, max={bbox_data.max():.3f}")
                print(f"  Obj: min={obj_data.min():.3f}, max={obj_data.max():.3f}")

            # Vectorized sigmoid on objectness
            obj_conf = 1 / (1 + np.exp(-obj_data[:, :, 0]))  # (h, w)
            max_conf_seen = max(max_conf_seen, obj_conf.max())

            # Find cells above threshold
            mask = obj_conf > self.conf_threshold
            if not mask.any():
                continue

            # Get indices of valid cells
            rows, cols = np.where(mask)
            confs = obj_conf[rows, cols]

            # Reshape bbox data for DFL: (h, w, 4, 16)
            bbox_reshaped = bbox_data.reshape(h, w, 4, dfl_bins)

            # Extract bbox data for valid cells: (N, 4, 16)
            valid_bbox = bbox_reshaped[rows, cols]

            # Vectorized DFL decode: softmax then weighted sum
            # Softmax: exp / sum(exp) along last axis
            exp_bbox = np.exp(valid_bbox - valid_bbox.max(axis=2, keepdims=True))  # Numerical stability
            dfl_softmax = exp_bbox / exp_bbox.sum(axis=2, keepdims=True)  # (N, 4, 16)
            bbox_decoded = (dfl_softmax * dfl_range).sum(axis=2)  # (N, 4)

            # Convert to pixel coordinates
            cx = (cols + 0.5) * stride
            cy = (rows + 0.5) * stride

            # bbox_decoded[:, 0:4] = [left, top, right, bottom] distances
            x1 = (cx - bbox_decoded[:, 0] * stride) / 640
            y1 = (cy - bbox_decoded[:, 1] * stride) / 640
            x2 = (cx + bbox_decoded[:, 2] * stride) / 640
            y2 = (cy + bbox_decoded[:, 3] * stride) / 640

            # Clamp to [0, 1]
            x1 = np.clip(x1, 0, 1)
            y1 = np.clip(y1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            y2 = np.clip(y2, 0, 1)

            # Filter valid boxes
            valid = (x2 > x1) & (y2 > y1) & ((x2 - x1) > 0.02) & ((y2 - y1) > 0.02)

            for i in range(len(confs)):
                if valid[i]:
                    detections.append(HailoDetection(
                        class_id=0,
                        class_name="object",
                        confidence=float(confs[i]),
                        bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]))
                    ))

        # Apply NMS with higher IoU threshold for overlapping objects
        detections = self._nms(detections)

        if debug:
            print(f"[FastSAM] Max confidence: {max_conf_seen:.4f}")
            print(f"[FastSAM] Detections after NMS: {len(detections)}")

        return detections

    def detections_to_assets(
        self,
        detections: List[HailoDetection],
        room_scale: float = 10.0,
        image_width: int = 640,
        image_height: int = 480
    ) -> List[Dict[str, Any]]:
        """Convert detections to 3D assets for room scanner."""
        assets = []

        for i, det in enumerate(detections):
            # Skip classes we don't map
            asset_type = COCO_TO_ASSET.get(det.class_name, 'decoration')

            # Calculate center position
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2

            # Convert to world coordinates
            world_x = (cx - 0.5) * room_scale
            world_z = (cy - 0.5) * room_scale

            # Estimate size from bbox
            bbox_w = det.bbox[2] - det.bbox[0]
            bbox_h = det.bbox[3] - det.bbox[1]

            obj_width = bbox_w * room_scale * 0.5
            obj_depth = bbox_h * room_scale * 0.5
            obj_height = max(obj_width, obj_depth) * 0.8  # Estimate height

            # Get color for asset type
            color = ASSET_COLORS.get(asset_type, 0x808080)

            asset = {
                'asset_id': f'hailo_{i}_{det.class_name}',
                'asset_type': asset_type,
                'position': {
                    'x': float(world_x),
                    'y': float(obj_height / 2),
                    'z': float(world_z)
                },
                'size': {
                    'width': float(max(0.3, obj_width)),
                    'height': float(max(0.3, obj_height)),
                    'depth': float(max(0.3, obj_depth))
                },
                'rotation': 0.0,
                'color': color,
                'geometry': 'box',
                'metadata': {
                    'source': 'hailo',
                    'model': self.model_name,
                    'coco_class': det.class_name,
                    'coco_id': det.class_id,
                    'confidence': det.confidence,
                }
            }
            assets.append(asset)

        return assets


# Singleton instances
_hailo_scanner: Optional[HailoScanner] = None
_fastsam_scanner: Optional[HailoScanner] = None


def get_fastsam_scanner() -> Optional[HailoScanner]:
    """Get or create FastSAM scanner singleton."""
    global _fastsam_scanner

    if not HAS_HAILO:
        return None

    if _fastsam_scanner is None:
        try:
            model_path = HailoScanner.MODELS.get('fast_sam')
            if not model_path or not Path(model_path).exists():
                print(f"[FastSAM] Model not found: {model_path}")
                print("[FastSAM] Download with: curl -L -o /tmp/fastsam/fast_sam_s.hef https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/fast_sam_s.hef")
                return None
            _fastsam_scanner = HailoScanner('fast_sam')
        except Exception as e:
            print(f"[FastSAM] Failed to initialize: {e}")
            return None

    return _fastsam_scanner


def get_hailo_scanner(model_name: str = 'yolov5_seg') -> Optional[HailoScanner]:
    """Get or create Hailo scanner singleton."""
    global _hailo_scanner

    if not HAS_HAILO:
        return None

    if _hailo_scanner is None:
        try:
            _hailo_scanner = HailoScanner(model_name)
        except Exception as e:
            print(f"[HailoScanner] Failed to initialize: {e}")
            return None

    return _hailo_scanner


def scan_with_hailo(
    image_data: str,
    room_scale: float = 10.0
) -> Dict[str, Any]:
    """
    Scan image using Hailo-8 NPU.

    Args:
        image_data: Base64-encoded image
        room_scale: World scale factor

    Returns:
        Scan results with assets
    """
    import base64

    scanner = get_hailo_scanner()
    if scanner is None:
        return {'error': 'Hailo scanner not available', 'assets': []}

    # Decode image
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {'error': 'Failed to decode image', 'assets': []}

    height, width = image.shape[:2]

    # Run detection
    start = time.time()
    detections = scanner.detect(image, debug=False)
    inference_time = (time.time() - start) * 1000
    print(f"[HailoScanner] {len(detections)} objects in {inference_time:.0f}ms")

    # Convert to assets
    assets = scanner.detections_to_assets(detections, room_scale, width, height)

    # Add floor and wall
    assets = [
        {
            'asset_id': 'floor_main',
            'asset_type': 'floor',
            'position': {'x': 0, 'y': 0, 'z': 0},
            'size': {'width': room_scale, 'height': 0.1, 'depth': room_scale},
            'color': 0x2d3748,
            'geometry': 'box'
        },
        {
            'asset_id': 'wall_back',
            'asset_type': 'wall',
            'position': {'x': 0, 'y': 1.5, 'z': -room_scale/2},
            'size': {'width': room_scale, 'height': 3.0, 'depth': 0.2},
            'color': 0x4a5568,
            'geometry': 'box'
        }
    ] + assets

    scan_id = f"hailo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Format compatible with room_scanner.py and server.py
    return {
        'scan_id': scan_id,
        'timestamp': datetime.now().isoformat(),
        'images_processed': 1,
        'room_dimensions': {'width': room_scale, 'height': 3.0, 'depth': room_scale},
        'processing_time_ms': inference_time,
        'detected_objects': [
            {
                'object_type': d.class_name,
                'confidence': d.confidence,
                'bounding_box': d.bbox,
                'center': [d.bbox[0] + d.bbox[2]//2, d.bbox[1] + d.bbox[3]//2],
                'area': d.bbox[2] * d.bbox[3],
                'color_rgb': [128, 128, 128],
                'estimated_depth': 0.5
            }
            for d in detections
        ],
        'generated_assets': assets,
        'metadata': {
            'segmentation_model': 'hailo_yolov5_seg',
            'device': 'Hailo-8',
            'room_scale': room_scale,
            'hailo_used': True,
            'opencv_available': True
        }
    }


def scan_with_fastsam(
    image_data: str,
    room_scale: float = 10.0
) -> Dict[str, Any]:
    """
    Scan image using FastSAM on Hailo-8 NPU.

    FastSAM provides fast zero-shot segmentation at 48 FPS on Hailo-8.
    Much better for room scanning than YOLOv5 detection.

    Args:
        image_data: Base64-encoded image
        room_scale: World scale factor

    Returns:
        Scan results with assets
    """
    import base64

    scanner = get_fastsam_scanner()
    if scanner is None:
        return {'error': 'FastSAM scanner not available', 'assets': []}

    # Decode image
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {'error': 'Failed to decode image', 'assets': []}

    height, width = image.shape[:2]

    # Set higher threshold for FastSAM (it finds many objects)
    scanner.conf_threshold = 0.50  # Higher threshold for cleaner results
    scanner.iou_threshold = 0.30   # More aggressive NMS for overlapping objects

    # Run detection
    start = time.time()
    detections = scanner.detect(image, debug=True)  # Debug on first run
    inference_time = (time.time() - start) * 1000
    print(f"[FastSAM] {len(detections)} objects in {inference_time:.0f}ms")

    # Convert to assets
    assets = scanner.detections_to_assets(detections, room_scale, width, height)

    # Add floor and wall
    assets = [
        {
            'asset_id': 'floor_main',
            'asset_type': 'floor',
            'position': {'x': 0, 'y': 0, 'z': 0},
            'size': {'width': room_scale, 'height': 0.1, 'depth': room_scale},
            'color': 0x2d3748,
            'geometry': 'box'
        },
        {
            'asset_id': 'wall_back',
            'asset_type': 'wall',
            'position': {'x': 0, 'y': 1.5, 'z': -room_scale/2},
            'size': {'width': room_scale, 'height': 3.0, 'depth': 0.2},
            'color': 0x4a5568,
            'geometry': 'box'
        }
    ] + assets

    scan_id = f"fastsam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        'scan_id': scan_id,
        'timestamp': datetime.now().isoformat(),
        'images_processed': 1,
        'room_dimensions': {'width': room_scale, 'height': 3.0, 'depth': room_scale},
        'processing_time_ms': inference_time,
        'detected_objects': [
            {
                'object_type': d.class_name,
                'confidence': d.confidence,
                'bounding_box': d.bbox,
                'center': [(d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2],
                'area': (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                'color_rgb': [128, 128, 128],
                'estimated_depth': 0.5
            }
            for d in detections
        ],
        'generated_assets': assets,
        'metadata': {
            'segmentation_model': 'fast_sam_s',
            'device': 'Hailo-8',
            'room_scale': room_scale,
            'fastsam_used': True,
            'expected_fps': 48
        }
    }


# =============================================================================
# Hybrid Hailo-8 + SAM3 Scanner
# =============================================================================
# Uses Hailo-8 for fast detection (~200ms) and SAM3 for precise segmentation
# Much faster than full SAM3 because SAM3 only segments detected regions

_sam3_model = None
_sam3_processor = None

def get_sam3_model():
    """Load SAM model (cached). Uses original SAM for better box prompt support."""
    global _sam3_model, _sam3_processor

    if _sam3_model is None:
        try:
            # Use original SAM which has proper box prompt support
            from transformers import SamProcessor, SamModel
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[HailoSAM] Loading SAM-ViT-Base on {device}...")
            print("[HailoSAM] Note: SAM on CPU is slow (~30-60s per object)")

            _sam3_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            _sam3_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

            print("[HailoSAM] SAM loaded successfully")
        except Exception as e:
            print(f"[HailoSAM] Failed to load SAM: {e}")
            return None, None

    return _sam3_model, _sam3_processor


async def scan_with_hailo_sam3(
    image_data: str,
    room_scale: float = 10.0
) -> Dict[str, Any]:
    """
    Hybrid scanner using Hailo-8 for detection + SAM3 for segmentation.

    Pipeline:
    1. Hailo-8 detects objects and provides bounding boxes (~200ms)
    2. SAM3 segments only the detected regions using box prompts
    3. Much faster than full SAM3 scan

    Args:
        image_data: Base64-encoded image
        room_scale: World scale factor

    Returns:
        Scan results with high-quality SAM3 segmentation masks
    """
    import base64
    import torch
    from PIL import Image
    from io import BytesIO

    start_time = time.time()

    # Get Hailo scanner
    scanner = get_hailo_scanner()
    if scanner is None:
        return {'error': 'Hailo scanner not available', 'assets': []}

    # Get SAM3 model
    model, processor = get_sam3_model()
    if model is None:
        return {'error': 'SAM3 model not available', 'assets': []}

    device = next(model.parameters()).device

    # Decode image
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if cv_image is None:
        return {'error': 'Failed to decode image', 'assets': []}

    height, width = cv_image.shape[:2]

    # Convert to PIL for SAM3
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Phase 1: Hailo detection (~200ms)
    hailo_start = time.time()
    detections = scanner.detect(cv_image, debug=False)
    hailo_time = (time.time() - hailo_start) * 1000
    print(f"[HailoSAM3] Hailo detected {len(detections)} objects in {hailo_time:.0f}ms")

    if not detections:
        # No objects detected, return basic structure
        return {
            'scan_id': f"hailo_sam3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'images_processed': 1,
            'room_dimensions': {'width': room_scale, 'height': 3.0, 'depth': room_scale},
            'processing_time_ms': (time.time() - start_time) * 1000,
            'detected_objects': [],
            'generated_assets': _generate_room_structure(room_scale),
            'metadata': {
                'hailo_used': True,
                'sam3_used': False,
                'hailo_time_ms': hailo_time,
                'device': 'Hailo-8 + CPU'
            }
        }

    # Phase 2: SAM3 segmentation using Hailo bounding boxes
    sam3_start = time.time()

    # Convert Hailo detections to SAM3 box format
    # SAM3 expects boxes as [[x1, y1, x2, y2], ...]
    input_boxes = []
    for det in detections:
        # det.bbox is (x, y, w, h) normalized or pixel
        x, y, w, h = det.bbox
        # Convert to pixel coordinates if normalized
        if max(x, y, w, h) <= 1.0:
            x1 = int(x * width)
            y1 = int(y * height)
            x2 = int((x + w) * width)
            y2 = int((y + h) * height)
        else:
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

        # Clamp to image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 > x1 and y2 > y1:
            input_boxes.append([x1, y1, x2, y2])

    print(f"[HailoSAM3] Processing {len(input_boxes)} boxes with SAM3...")

    all_assets = []
    all_detected = []

    # Process each detection with SAM3
    for i, (det, box) in enumerate(zip(detections, input_boxes)):
        try:
            # Use SAM3 with box prompt for this detection
            inputs = processor(
                images=pil_image,
                input_boxes=[[box]],  # Nested list for batch
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Get mask - SAM returns (batch, num_masks, 3, H, W)
            # Use post_process_masks with original_sizes and reshaped_input_sizes
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]  # Get first image's masks

            print(f"[HailoSAM] Detection {i}: masks shape = {masks.shape if hasattr(masks, 'shape') else len(masks)}")

            # SAM outputs shape: [3_masks_per_box, H, W] for each box
            # Take the first mask (highest quality)
            if len(masks) > 0:
                mask = masks[0]  # First mask variant
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()

                print(f"[HailoSAM] Mask shape: {mask.shape}, dtype: {mask.dtype}")

                # Calculate mask area and centroid (mask is boolean)
                mask_binary = mask.astype(bool) if mask.dtype != bool else mask
                mask_area = mask_binary.sum()
                print(f"[HailoSAM] Mask area: {mask_area:,} pixels")

                if mask_area > 100:  # Minimum area threshold
                    # Get centroid
                    y_coords, x_coords = np.where(mask_binary)
                    if len(x_coords) > 0:
                        cx = int(x_coords.mean())
                        cy = int(y_coords.mean())

                        # Get bounding box from mask
                        x1_m, x2_m = x_coords.min(), x_coords.max()
                        y1_m, y2_m = y_coords.min(), y_coords.max()

                        # Convert to world coordinates
                        norm_x = cx / width
                        norm_y = cy / height
                        world_x = (norm_x - 0.5) * room_scale
                        world_z = (norm_y - 0.5) * room_scale

                        # Object size from mask bounds
                        obj_width = (x2_m - x1_m) / width * room_scale * 0.5
                        obj_depth = (y2_m - y1_m) / height * room_scale * 0.5

                        # Get type info
                        from room_scanner import ASSET_TYPES
                        type_info = ASSET_TYPES.get(det.class_name, ASSET_TYPES.get("furniture"))

                        asset = {
                            "asset_id": f"hailo_sam3_{i}_{det.class_name}",
                            "asset_type": det.class_name,
                            "position": {
                                "x": world_x,
                                "y": type_info["default_size"][2] / 2,
                                "z": world_z
                            },
                            "size": {
                                "width": max(0.3, obj_width),
                                "height": type_info["default_size"][2],
                                "depth": max(0.3, obj_depth)
                            },
                            "rotation": 0.0,
                            "color": type_info["color"],
                            "geometry": type_info["geometry"],
                            "metadata": {
                                "source": "hailo_sam3",
                                "hailo_confidence": det.confidence,
                                "mask_area": int(mask_area),
                                "class_id": det.class_id
                            }
                        }
                        all_assets.append(asset)

                        all_detected.append({
                            "object_type": det.class_name,
                            "confidence": det.confidence,
                            "bounding_box": [int(x1_m), int(y1_m), int(x2_m - x1_m), int(y2_m - y1_m)],
                            "center": [cx, cy],
                            "area": int(mask_area),
                            "has_mask": True
                        })

        except Exception as e:
            print(f"[HailoSAM3] Error processing detection {i}: {e}")
            continue

    sam3_time = (time.time() - sam3_start) * 1000
    total_time = (time.time() - start_time) * 1000

    print(f"[HailoSAM3] SAM3 processed {len(all_assets)} objects in {sam3_time:.0f}ms")
    print(f"[HailoSAM3] Total pipeline: {total_time:.0f}ms")

    # Add room structure
    room_assets = _generate_room_structure(room_scale)

    return {
        'scan_id': f"hailo_sam3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'images_processed': 1,
        'room_dimensions': {'width': room_scale, 'height': 3.0, 'depth': room_scale},
        'processing_time_ms': total_time,
        'detected_objects': all_detected,
        'generated_assets': room_assets + all_assets,
        'metadata': {
            'hailo_used': True,
            'sam3_used': True,
            'hailo_time_ms': hailo_time,
            'sam3_time_ms': sam3_time,
            'hailo_detections': len(detections),
            'sam3_segmented': len(all_assets),
            'device': 'Hailo-8 + CPU'
        }
    }


def _generate_room_structure(room_scale: float) -> List[Dict]:
    """Generate floor and wall assets."""
    return [
        {
            'asset_id': 'floor_main',
            'asset_type': 'floor',
            'position': {'x': 0, 'y': 0, 'z': 0},
            'size': {'width': room_scale, 'height': 0.1, 'depth': room_scale},
            'color': 0x2d3748,
            'geometry': 'box'
        },
        {
            'asset_id': 'wall_back',
            'asset_type': 'wall',
            'position': {'x': 0, 'y': 1.5, 'z': -room_scale / 2},
            'size': {'width': room_scale, 'height': 3.0, 'depth': 0.2},
            'color': 0x4a5568,
            'geometry': 'box'
        }
    ]


if __name__ == '__main__':
    import sys
    import base64

    print("=== Hailo Scanner Test ===")

    if not HAS_HAILO:
        print("Hailo not available")
        sys.exit(1)

    # Test with synthetic image
    print("Creating test image...")
    test_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    # Add some colored rectangles
    cv2.rectangle(test_img, (100, 200), (300, 400), (139, 69, 19), -1)  # Brown
    cv2.rectangle(test_img, (400, 150), (550, 350), (34, 139, 34), -1)  # Green

    print("Initializing scanner...")
    scanner = HailoScanner()

    print("Running detection...")
    start = time.time()
    detections = scanner.detect(test_img)
    elapsed = (time.time() - start) * 1000

    print(f"\nInference time: {elapsed:.1f}ms")
    print(f"Detections: {len(detections)}")

    for det in detections[:5]:
        print(f"  - {det.class_name}: {det.confidence:.2f}")

    # Test full pipeline
    print("\nTesting full scan pipeline...")
    _, buffer = cv2.imencode('.jpg', test_img)
    img_b64 = base64.b64encode(buffer).decode()

    result = scan_with_hailo(img_b64)
    print(f"Scan ID: {result['scan_id']}")
    print(f"Detected: {len(result['detected_objects'])} objects")
    print(f"Assets: {len(result['generated_assets'])}")
    print(f"Time: {result['processing_time_ms']:.1f}ms")
    print(f"Hailo used: {result['metadata'].get('hailo_used', False)}")
