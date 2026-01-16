"""
Room Scanner - Image to 3D Asset Pipeline

Processes room images and generates 3D assets for the HomeScan simulator.
Uses OpenCV for image analysis and generates Three.js-compatible asset data.

Features:
- Color/edge analysis for room structure
- Simple object detection via contours
- Depth estimation from single images
- 3D asset generation for simulator

Usage:
    from room_scanner import RoomScanner, process_room_images

    scanner = RoomScanner()
    assets = scanner.process_images(image_data_list)
"""

import base64
import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from io import BytesIO

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("OpenCV not available - room scanner limited")


# Asset type definitions for Three.js
ASSET_TYPES = {
    "wall": {"color": 0x4a5568, "geometry": "box", "default_size": [0.2, 3.0, 2.5]},
    "floor": {"color": 0x2d3748, "geometry": "plane", "default_size": [10, 10, 0.1]},
    "furniture": {"color": 0x805d40, "geometry": "box", "default_size": [1.5, 1.0, 0.8]},
    "table": {"color": 0x8b4513, "geometry": "box", "default_size": [1.2, 0.8, 0.75]},
    "chair": {"color": 0x654321, "geometry": "box", "default_size": [0.5, 0.5, 1.0]},
    "couch": {"color": 0x4a6fa5, "geometry": "box", "default_size": [2.0, 0.9, 0.85]},
    "shelf": {"color": 0x8b7355, "geometry": "box", "default_size": [1.5, 0.3, 1.8]},
    "lamp": {"color": 0xffd700, "geometry": "cylinder", "default_size": [0.3, 0.3, 1.2]},
    "plant": {"color": 0x228b22, "geometry": "cone", "default_size": [0.4, 0.4, 0.8]},
    "obstacle": {"color": 0xdc2626, "geometry": "cylinder", "default_size": [0.5, 0.5, 1.5]},
    "decoration": {"color": 0x9370db, "geometry": "dodecahedron", "default_size": [0.5, 0.5, 0.5]},
}

# Color ranges for object detection (HSV)
COLOR_RANGES = {
    "brown": {"lower": [10, 50, 50], "upper": [30, 255, 200], "type": "furniture"},
    "blue": {"lower": [100, 50, 50], "upper": [130, 255, 255], "type": "couch"},
    "green": {"lower": [35, 50, 50], "upper": [85, 255, 255], "type": "plant"},
    "white": {"lower": [0, 0, 200], "upper": [180, 30, 255], "type": "wall"},
    "gray": {"lower": [0, 0, 80], "upper": [180, 30, 180], "type": "floor"},
}


@dataclass
class DetectedObject:
    """An object detected in the image."""
    object_type: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    area: int
    color_rgb: Tuple[int, int, int]
    estimated_depth: float = 0.5  # 0=close, 1=far


@dataclass
class Asset3D:
    """A 3D asset for the simulator."""
    asset_id: str
    asset_type: str
    position: Dict[str, float]  # x, y, z
    size: Dict[str, float]  # width, height, depth
    rotation: float = 0.0
    color: int = 0x808080
    geometry: str = "box"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomScanResult:
    """Result of room scanning."""
    scan_id: str
    timestamp: str
    images_processed: int
    detected_objects: List[DetectedObject]
    generated_assets: List[Asset3D]
    room_dimensions: Dict[str, float]
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoomScanner:
    """
    Processes room images and generates 3D assets.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("room_scans")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Processing parameters
        self.min_contour_area = 200  # Lowered for real camera images
        self.max_objects = 50
        self.depth_scale = 10.0  # World units
        self.blur_kernel = (3, 3)  # Smaller blur for sharper edges

    def decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image data to OpenCV format."""
        if not HAS_OPENCV:
            return None

        try:
            # Handle data URL format
            if "," in image_data:
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"[RoomScanner] Failed to decode image: {e}")
            return None

    def analyze_image(self, image: np.ndarray) -> List[DetectedObject]:
        """Analyze image and detect objects."""
        if not HAS_OPENCV:
            return []

        detected = []
        height, width = image.shape[:2]

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Edge detection for structure
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edges = cv2.Canny(blurred, 30, 120)  # Lower thresholds for better detection

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            # Get dominant color in region
            roi = image[y:y+h, x:x+w]
            avg_color = roi.mean(axis=(0, 1)).astype(int)
            color_rgb = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))

            # Determine object type based on color and shape
            object_type = self._classify_object(roi, hsv[y:y+h, x:x+w], w, h)

            # Estimate depth based on position (bottom = closer)
            depth = y / height  # 0=top/far, 1=bottom/close
            estimated_depth = 1.0 - depth  # Invert so 0=close, 1=far

            detected.append(DetectedObject(
                object_type=object_type,
                confidence=min(area / (width * height) * 10, 1.0),
                bounding_box=(x, y, w, h),
                center=center,
                area=area,
                color_rgb=color_rgb,
                estimated_depth=estimated_depth,
            ))

        # Sort by area (largest first) and limit
        detected.sort(key=lambda d: d.area, reverse=True)
        return detected[:self.max_objects]

    def _classify_object(
        self,
        roi: np.ndarray,
        roi_hsv: np.ndarray,
        width: int,
        height: int
    ) -> str:
        """Classify detected region as object type."""
        if not HAS_OPENCV:
            return "decoration"

        # Aspect ratio analysis
        aspect = width / max(height, 1)

        # Color analysis
        for color_name, color_range in COLOR_RANGES.items():
            lower = np.array(color_range["lower"])
            upper = np.array(color_range["upper"])
            mask = cv2.inRange(roi_hsv, lower, upper)
            coverage = mask.sum() / (width * height * 255)

            if coverage > 0.3:
                return color_range["type"]

        # Shape-based classification
        if aspect > 2.0:
            return "shelf" if height > width else "table"
        elif aspect < 0.5:
            return "lamp"
        elif width > 100 and height > 80:
            return "furniture"
        else:
            return "decoration"

    def generate_assets(
        self,
        detected_objects: List[DetectedObject],
        image_width: int,
        image_height: int,
        room_scale: float = 10.0
    ) -> List[Asset3D]:
        """Convert detected objects to 3D assets."""
        assets = []

        for i, obj in enumerate(detected_objects):
            asset_type = obj.object_type
            type_info = ASSET_TYPES.get(asset_type, ASSET_TYPES["decoration"])

            # Convert image coordinates to 3D world coordinates
            norm_x = obj.center[0] / image_width  # 0-1
            norm_y = obj.center[1] / image_height  # 0-1

            # Map to world coordinates (centered)
            world_x = (norm_x - 0.5) * room_scale
            world_z = (norm_y - 0.5) * room_scale
            world_y = type_info["default_size"][2] / 2  # Place on ground

            # Scale based on detected size
            scale_factor = math.sqrt(obj.area) / 100
            size = type_info["default_size"].copy()
            size = [s * max(0.5, min(2.0, scale_factor)) for s in size]

            # Create asset
            asset = Asset3D(
                asset_id=f"asset_{i}_{asset_type}",
                asset_type=asset_type,
                position={"x": world_x, "y": world_y, "z": world_z},
                size={"width": size[0], "height": size[2], "depth": size[1]},
                rotation=random.uniform(0, math.pi * 2),
                color=type_info["color"],
                geometry=type_info["geometry"],
                metadata={
                    "source": "room_scan",
                    "confidence": obj.confidence,
                    "original_color": obj.color_rgb,
                    "depth": obj.estimated_depth,
                }
            )
            assets.append(asset)

        return assets

    def estimate_room_dimensions(
        self,
        detected_objects: List[DetectedObject],
        image_width: int,
        image_height: int
    ) -> Dict[str, float]:
        """Estimate room dimensions from detected objects."""
        # Simple estimation based on object spread
        if not detected_objects:
            return {"width": 10.0, "height": 3.0, "depth": 10.0}

        # Calculate bounding box of all objects
        min_x = min(obj.bounding_box[0] for obj in detected_objects)
        max_x = max(obj.bounding_box[0] + obj.bounding_box[2] for obj in detected_objects)
        min_y = min(obj.bounding_box[1] for obj in detected_objects)
        max_y = max(obj.bounding_box[1] + obj.bounding_box[3] for obj in detected_objects)

        # Scale to world units
        spread_x = (max_x - min_x) / image_width
        spread_y = (max_y - min_y) / image_height

        return {
            "width": max(5.0, spread_x * 15),
            "height": 3.0,  # Standard room height
            "depth": max(5.0, spread_y * 15),
        }

    def process_images(
        self,
        image_data_list: List[str],
        room_scale: float = 10.0
    ) -> RoomScanResult:
        """
        Process multiple room images and generate assets.

        Args:
            image_data_list: List of base64-encoded images
            room_scale: Scale factor for world units

        Returns:
            RoomScanResult with detected objects and generated assets
        """
        import time
        start_time = time.time()

        scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_detected = []
        all_assets = []
        image_dims = (640, 480)  # Default

        for idx, image_data in enumerate(image_data_list):
            image = self.decode_image(image_data)
            if image is None:
                continue

            image_dims = (image.shape[1], image.shape[0])

            # Detect objects
            detected = self.analyze_image(image)
            all_detected.extend(detected)

            # Generate assets
            assets = self.generate_assets(
                detected,
                image_dims[0],
                image_dims[1],
                room_scale
            )
            all_assets.extend(assets)

        # De-duplicate assets by proximity
        all_assets = self._deduplicate_assets(all_assets)

        # Estimate room dimensions
        room_dims = self.estimate_room_dimensions(
            all_detected,
            image_dims[0],
            image_dims[1]
        )

        # Add floor and walls
        room_assets = self._generate_room_structure(room_dims)
        all_assets = room_assets + all_assets

        processing_time = (time.time() - start_time) * 1000

        result = RoomScanResult(
            scan_id=scan_id,
            timestamp=datetime.now().isoformat(),
            images_processed=len(image_data_list),
            detected_objects=all_detected,
            generated_assets=all_assets,
            room_dimensions=room_dims,
            processing_time_ms=processing_time,
            metadata={
                "opencv_available": HAS_OPENCV,
                "room_scale": room_scale,
            }
        )

        # Save result
        self._save_result(result)

        return result

    def _deduplicate_assets(
        self,
        assets: List[Asset3D],
        min_distance: float = 1.0
    ) -> List[Asset3D]:
        """Remove duplicate assets that are too close together."""
        if not assets:
            return assets

        unique = [assets[0]]

        for asset in assets[1:]:
            is_duplicate = False
            for existing in unique:
                dist = math.sqrt(
                    (asset.position["x"] - existing.position["x"]) ** 2 +
                    (asset.position["z"] - existing.position["z"]) ** 2
                )
                if dist < min_distance and asset.asset_type == existing.asset_type:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(asset)

        return unique

    def _generate_room_structure(self, room_dims: Dict[str, float]) -> List[Asset3D]:
        """Generate floor and wall assets for the room."""
        assets = []
        w, h, d = room_dims["width"], room_dims["height"], room_dims["depth"]

        # Floor
        assets.append(Asset3D(
            asset_id="floor_main",
            asset_type="floor",
            position={"x": 0, "y": 0, "z": 0},
            size={"width": w, "height": 0.1, "depth": d},
            color=0x2d3748,
            geometry="box",
        ))

        # Walls (optional - can be added for enclosed spaces)
        wall_thickness = 0.2

        # Back wall
        assets.append(Asset3D(
            asset_id="wall_back",
            asset_type="wall",
            position={"x": 0, "y": h / 2, "z": -d / 2},
            size={"width": w, "height": h, "depth": wall_thickness},
            color=0x4a5568,
            geometry="box",
        ))

        return assets

    def _save_result(self, result: RoomScanResult) -> Path:
        """Save scan result to disk."""
        result_file = self.output_dir / f"{result.scan_id}.json"

        # Convert to serializable format
        data = {
            "scan_id": result.scan_id,
            "timestamp": result.timestamp,
            "images_processed": result.images_processed,
            "room_dimensions": result.room_dimensions,
            "processing_time_ms": result.processing_time_ms,
            "metadata": result.metadata,
            "detected_objects": [
                {
                    "object_type": obj.object_type,
                    "confidence": obj.confidence,
                    "bounding_box": obj.bounding_box,
                    "center": obj.center,
                    "area": obj.area,
                    "color_rgb": obj.color_rgb,
                    "estimated_depth": obj.estimated_depth,
                }
                for obj in result.detected_objects
            ],
            "generated_assets": [asdict(asset) for asset in result.generated_assets],
        }

        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)

        return result_file


def process_room_images(
    image_data_list: List[str],
    room_scale: float = 10.0,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to process room images.

    Args:
        image_data_list: List of base64-encoded images
        room_scale: Scale factor for world units
        output_dir: Optional output directory

    Returns:
        Dictionary with scan results for JSON response
    """
    scanner = RoomScanner(Path(output_dir) if output_dir else None)
    result = scanner.process_images(image_data_list, room_scale)

    return {
        "scan_id": result.scan_id,
        "images_processed": result.images_processed,
        "objects_detected": len(result.detected_objects),
        "assets_generated": len(result.generated_assets),
        "room_dimensions": result.room_dimensions,
        "processing_time_ms": result.processing_time_ms,
        "assets": [asdict(asset) for asset in result.generated_assets],
    }


# Singleton scanner
_room_scanner: Optional[RoomScanner] = None


def get_room_scanner(output_dir: Optional[str] = None) -> RoomScanner:
    """Get or create the room scanner singleton."""
    global _room_scanner
    if _room_scanner is None:
        _room_scanner = RoomScanner(Path(output_dir) if output_dir else None)
    return _room_scanner


# ============================================================================
# Room Boundary Detection for Guided Scanning
# ============================================================================

@dataclass
class RoomBoundary:
    """Detected room boundary (floor/wall/ceiling line)."""
    boundary_type: str  # "floor", "wall_left", "wall_right", "ceiling", "corner"
    line: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    angle: float  # degrees from horizontal


@dataclass
class ScanGuidance:
    """Guidance for the user during scanning."""
    coverage: Dict[str, float]  # Area coverage percentages
    missing_views: List[str]  # ["left_wall", "ceiling", etc.]
    suggestion: str  # "Turn left to capture the wall"
    ready_to_scan: bool  # True if enough coverage
    quality_score: float  # 0-1 overall quality


def detect_room_boundaries(image_data: str) -> Dict[str, Any]:
    """
    Detect floor, walls, and ceiling boundaries in an image.

    Uses edge detection and Hough line transform to find
    horizontal lines (floor/ceiling) and vertical lines (walls).

    Args:
        image_data: Base64-encoded image

    Returns:
        Dictionary with detected boundaries and guidance
    """
    if not HAS_OPENCV:
        return {"error": "OpenCV not available", "boundaries": []}

    scanner = get_room_scanner()
    image = scanner.decode_image(image_data)

    if image is None:
        return {"error": "Failed to decode image", "boundaries": []}

    height, width = image.shape[:2]
    boundaries = []

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=width // 6,
        maxLineGap=20
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            abs_angle = abs(angle)

            # Classify line based on angle and position
            line_tuple = (int(x1), int(y1), int(x2), int(y2))
            line_y_avg = (y1 + y2) / 2
            line_x_avg = (x1 + x2) / 2

            # Horizontal lines (floor/ceiling)
            if abs_angle < 15 or abs_angle > 165:
                if line_y_avg > height * 0.6:
                    boundary_type = "floor"
                    confidence = 0.8 + 0.2 * (line_y_avg / height)
                elif line_y_avg < height * 0.3:
                    boundary_type = "ceiling"
                    confidence = 0.8 + 0.2 * (1 - line_y_avg / height)
                else:
                    continue  # Skip middle horizontal lines

                boundaries.append(RoomBoundary(
                    boundary_type=boundary_type,
                    line=line_tuple,
                    confidence=min(confidence, 1.0),
                    angle=angle
                ))

            # Vertical lines (walls)
            elif 75 < abs_angle < 105:
                if line_x_avg < width * 0.3:
                    boundary_type = "wall_left"
                    confidence = 0.7 + 0.3 * (1 - line_x_avg / width)
                elif line_x_avg > width * 0.7:
                    boundary_type = "wall_right"
                    confidence = 0.7 + 0.3 * (line_x_avg / width)
                else:
                    continue  # Skip center vertical lines

                boundaries.append(RoomBoundary(
                    boundary_type=boundary_type,
                    line=line_tuple,
                    confidence=min(confidence, 1.0),
                    angle=angle
                ))

            # Diagonal lines (corners)
            elif 30 < abs_angle < 60 or 120 < abs_angle < 150:
                boundaries.append(RoomBoundary(
                    boundary_type="corner",
                    line=line_tuple,
                    confidence=0.6,
                    angle=angle
                ))

    # Detect dominant colors for floor/wall/ceiling areas
    regions = _analyze_room_regions(image)

    # Calculate coverage and generate guidance
    guidance = _generate_scan_guidance(boundaries, regions, width, height)

    return {
        "boundaries": [
            {
                "type": b.boundary_type,
                "line": b.line,
                "confidence": b.confidence,
                "angle": b.angle
            }
            for b in boundaries
        ],
        "regions": regions,
        "guidance": {
            "coverage": guidance.coverage,
            "missing_views": guidance.missing_views,
            "suggestion": guidance.suggestion,
            "ready_to_scan": guidance.ready_to_scan,
            "quality_score": guidance.quality_score
        },
        "image_size": {"width": width, "height": height}
    }


def _analyze_room_regions(image: np.ndarray) -> Dict[str, Any]:
    """Analyze image regions for floor/wall/ceiling characteristics."""
    height, width = image.shape[:2]

    # Divide image into regions
    top_region = image[0:height//3, :]
    middle_region = image[height//3:2*height//3, :]
    bottom_region = image[2*height//3:, :]

    def get_dominant_color(region):
        avg = region.mean(axis=(0, 1))
        return {"r": int(avg[2]), "g": int(avg[1]), "b": int(avg[0])}

    def get_brightness(region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(gray.mean() / 255)

    def get_edge_density(region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(edges.mean() / 255)

    return {
        "ceiling": {
            "color": get_dominant_color(top_region),
            "brightness": get_brightness(top_region),
            "edge_density": get_edge_density(top_region),
            "detected": get_brightness(top_region) > 0.4  # Ceilings often bright
        },
        "walls": {
            "color": get_dominant_color(middle_region),
            "brightness": get_brightness(middle_region),
            "edge_density": get_edge_density(middle_region),
            "detected": get_edge_density(middle_region) > 0.05
        },
        "floor": {
            "color": get_dominant_color(bottom_region),
            "brightness": get_brightness(bottom_region),
            "edge_density": get_edge_density(bottom_region),
            "detected": get_brightness(bottom_region) < 0.6  # Floors often darker
        }
    }


def _generate_scan_guidance(
    boundaries: List[RoomBoundary],
    regions: Dict[str, Any],
    width: int,
    height: int
) -> ScanGuidance:
    """Generate guidance for the user based on detected boundaries."""

    # Track what we've detected
    detected_types = {b.boundary_type for b in boundaries}

    # Calculate coverage
    coverage = {
        "floor": 1.0 if "floor" in detected_types else (0.5 if regions["floor"]["detected"] else 0.0),
        "ceiling": 1.0 if "ceiling" in detected_types else (0.5 if regions["ceiling"]["detected"] else 0.0),
        "wall_left": 1.0 if "wall_left" in detected_types else 0.0,
        "wall_right": 1.0 if "wall_right" in detected_types else 0.0,
        "corners": min(len([b for b in boundaries if b.boundary_type == "corner"]) / 2, 1.0)
    }

    # Determine missing views
    missing = []
    suggestions = []

    if coverage["floor"] < 0.5:
        missing.append("floor")
        suggestions.append("Tilt camera down to capture the floor")

    if coverage["ceiling"] < 0.5:
        missing.append("ceiling")
        suggestions.append("Tilt camera up to capture the ceiling")

    if coverage["wall_left"] < 0.5:
        missing.append("left_wall")
        suggestions.append("Turn left to capture the left wall")

    if coverage["wall_right"] < 0.5:
        missing.append("right_wall")
        suggestions.append("Turn right to capture the right wall")

    if coverage["corners"] < 0.5:
        missing.append("corners")
        suggestions.append("Point at room corners for better depth")

    # Calculate overall quality
    quality = sum(coverage.values()) / len(coverage)

    # Determine if ready
    ready = quality > 0.6 and len(missing) <= 2

    # Pick best suggestion
    suggestion = suggestions[0] if suggestions else "Good framing! Capture this view."

    return ScanGuidance(
        coverage=coverage,
        missing_views=missing,
        suggestion=suggestion,
        ready_to_scan=ready,
        quality_score=quality
    )


def analyze_frame_for_guided_scan(image_data: str) -> Dict[str, Any]:
    """
    Lightweight frame analysis for real-time guided scanning.

    Returns boundary detection, quality assessment, and user guidance
    quickly enough for real-time feedback (~30fps target).

    Args:
        image_data: Base64-encoded image frame

    Returns:
        Dictionary with boundaries, guidance, and overlay data
    """
    return detect_room_boundaries(image_data)


if __name__ == "__main__":
    # Test with a sample image
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        result = process_room_images([image_data])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python room_scanner.py <image_path>")
        print("\nTesting with synthetic data...")

        # Create a simple test image
        if HAS_OPENCV:
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some colored rectangles
            cv2.rectangle(test_image, (100, 200), (250, 350), (139, 69, 19), -1)  # Brown furniture
            cv2.rectangle(test_image, (400, 150), (550, 280), (107, 142, 35), -1)  # Green plant
            cv2.rectangle(test_image, (250, 350), (450, 450), (169, 169, 169), -1)  # Gray floor area

            # Encode
            _, buffer = cv2.imencode('.jpg', test_image)
            image_data = base64.b64encode(buffer).decode()

            result = process_room_images([image_data])
            print(f"Assets generated: {result['assets_generated']}")
            for asset in result['assets']:
                print(f"  - {asset['asset_type']}: {asset['position']}")
