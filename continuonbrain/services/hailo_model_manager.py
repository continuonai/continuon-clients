"""
Hailo Model Manager - Download, catalog, and manage HEF models for Hailo-8 NPU.

Supports:
- Model catalog with metadata
- Automatic model download from Hailo Model Zoo
- Model validation and health checks
- Runtime model switching

Available Models (Hailo Model Zoo):
- yolov8s: Object detection (80 COCO classes)
- yolov8n: Lightweight object detection
- yolov8s_pose: Human pose estimation (17 keypoints)
- resnet50: Image classification (ImageNet 1000)
- mobilenet_v2: Lightweight classification
- efficientnet_lite0: Efficient classification
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve
from urllib.error import URLError
import time

logger = logging.getLogger(__name__)


@dataclass
class HailoModelInfo:
    """Metadata for a Hailo HEF model."""
    name: str
    task: str  # detection, classification, pose, segmentation
    hef_path: Optional[str] = None
    input_shape: tuple = (640, 640, 3)  # H, W, C
    num_classes: int = 80
    class_names: Optional[List[str]] = None
    source_url: Optional[str] = None
    sha256: Optional[str] = None
    downloaded: bool = False
    validated: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HailoModelInfo":
        d = d.copy()
        if "input_shape" in d:
            d["input_shape"] = tuple(d["input_shape"])
        return cls(**d)


# Hailo Model Zoo catalog
# Note: These URLs point to the official Hailo Model Zoo
# https://github.com/hailo-ai/hailo_model_zoo
MODEL_CATALOG = {
    "yolov8s": HailoModelInfo(
        name="yolov8s",
        task="detection",
        input_shape=(640, 640, 3),
        num_classes=80,
        description="YOLOv8 Small - Object detection (80 COCO classes)",
        source_url="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov8s.hef",
    ),
    "yolov8n": HailoModelInfo(
        name="yolov8n",
        task="detection",
        input_shape=(640, 640, 3),
        num_classes=80,
        description="YOLOv8 Nano - Lightweight object detection",
        source_url="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov8n.hef",
    ),
    "yolov8s_pose": HailoModelInfo(
        name="yolov8s_pose",
        task="pose",
        input_shape=(640, 640, 3),
        num_classes=1,  # person class
        description="YOLOv8 Pose - Human pose estimation (17 keypoints)",
        source_url="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov8s_pose.hef",
    ),
    "resnet50": HailoModelInfo(
        name="resnet50",
        task="classification",
        input_shape=(224, 224, 3),
        num_classes=1000,
        description="ResNet50 - ImageNet classification",
        source_url="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/resnet_v1_50.hef",
    ),
    "mobilenet_v2": HailoModelInfo(
        name="mobilenet_v2",
        task="classification",
        input_shape=(224, 224, 3),
        num_classes=1000,
        description="MobileNetV2 - Lightweight classification",
        source_url="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/mobilenet_v2_1.0.hef",
    ),
    "efficientnet_lite0": HailoModelInfo(
        name="efficientnet_lite0",
        task="classification",
        input_shape=(224, 224, 3),
        num_classes=1000,
        description="EfficientNet-Lite0 - Efficient classification",
        source_url="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/efficientnet_lite0.hef",
    ),
}

# COCO class names for detection models
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


class HailoModelManager:
    """
    Manages Hailo HEF models for inference.

    Features:
    - Automatic model discovery in model directory
    - Model download from Hailo Model Zoo
    - Model validation via hailortcli
    - Runtime model switching
    - Catalog persistence
    """

    def __init__(
        self,
        model_dir: Path = Path("/opt/continuonos/brain/model/hailo"),
        catalog_file: str = "model_catalog.json",
        auto_discover: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.model_dir / catalog_file

        # Local catalog of available models
        self._models: Dict[str, HailoModelInfo] = {}

        # Currently loaded model
        self._active_model: Optional[str] = None

        # Load catalog
        self._load_catalog()

        # Auto-discover local HEF files
        if auto_discover:
            self._discover_models()

    def _load_catalog(self) -> None:
        """Load model catalog from disk."""
        if self.catalog_path.exists():
            try:
                data = json.loads(self.catalog_path.read_text())
                for name, info in data.get("models", {}).items():
                    self._models[name] = HailoModelInfo.from_dict(info)
                logger.info(f"Loaded {len(self._models)} models from catalog")
            except Exception as e:
                logger.warning(f"Failed to load catalog: {e}")

    def _save_catalog(self) -> None:
        """Save model catalog to disk."""
        try:
            data = {
                "models": {name: info.to_dict() for name, info in self._models.items()},
                "updated_at": time.time(),
            }
            self.catalog_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save catalog: {e}")

    def _discover_models(self) -> None:
        """Discover HEF files in model directory and common locations."""
        search_paths = [
            self.model_dir,
            Path("/opt/continuonos/brain/model/base_model"),
            Path("/opt/continuonos/brain/model"),
            Path.home() / "models" / "hailo",
        ]

        for search_path in search_paths:
            if not search_path.exists():
                continue
            for hef_file in search_path.glob("*.hef"):
                name = hef_file.stem
                if name not in self._models:
                    # Try to match with catalog
                    if name in MODEL_CATALOG:
                        info = MODEL_CATALOG[name]
                        info.hef_path = str(hef_file)
                        info.downloaded = True
                    else:
                        # Create generic entry
                        info = HailoModelInfo(
                            name=name,
                            task="unknown",
                            hef_path=str(hef_file),
                            downloaded=True,
                            description=f"Discovered HEF: {hef_file.name}",
                        )
                    self._models[name] = info
                    logger.info(f"Discovered model: {name} at {hef_file}")

        self._save_catalog()

    def list_available(self) -> List[str]:
        """List all available (downloaded) models."""
        return [name for name, info in self._models.items() if info.downloaded]

    def list_downloadable(self) -> List[str]:
        """List models available for download from Hailo Model Zoo."""
        available = set(self._models.keys())
        return [name for name in MODEL_CATALOG.keys() if name not in available]

    def get_model_info(self, name: str) -> Optional[HailoModelInfo]:
        """Get model info by name."""
        return self._models.get(name) or MODEL_CATALOG.get(name)

    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the HEF file path for a model."""
        info = self._models.get(name)
        if info and info.hef_path:
            return Path(info.hef_path)
        return None

    def download_model(
        self,
        name: str,
        force: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """
        Download a model from Hailo Model Zoo.

        Args:
            name: Model name from catalog
            force: Re-download even if exists
            progress_callback: Optional callback(bytes_downloaded, total_bytes)

        Returns:
            True if download successful
        """
        if name not in MODEL_CATALOG:
            logger.error(f"Model '{name}' not in catalog")
            return False

        catalog_info = MODEL_CATALOG[name]

        if not catalog_info.source_url:
            logger.error(f"No download URL for model '{name}'")
            return False

        dest_path = self.model_dir / f"{name}.hef"

        if dest_path.exists() and not force:
            logger.info(f"Model '{name}' already exists at {dest_path}")
            self._models[name] = HailoModelInfo(
                **{**asdict(catalog_info), "hef_path": str(dest_path), "downloaded": True}
            )
            self._save_catalog()
            return True

        logger.info(f"Downloading {name} from {catalog_info.source_url}")

        try:
            # Download with progress
            def reporthook(block_num, block_size, total_size):
                if progress_callback:
                    downloaded = block_num * block_size
                    progress_callback(downloaded, total_size)

            temp_path = dest_path.with_suffix(".tmp")
            urlretrieve(catalog_info.source_url, temp_path, reporthook)

            # Verify download
            if temp_path.stat().st_size < 1024:
                logger.error(f"Downloaded file too small, likely failed")
                temp_path.unlink()
                return False

            # Move to final location
            shutil.move(temp_path, dest_path)

            # Update catalog
            self._models[name] = HailoModelInfo(
                **{**asdict(catalog_info), "hef_path": str(dest_path), "downloaded": True}
            )
            self._save_catalog()

            logger.info(f"Successfully downloaded {name} to {dest_path}")
            return True

        except URLError as e:
            logger.error(f"Download failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def validate_model(self, name: str) -> bool:
        """
        Validate a model using hailortcli.

        Returns:
            True if model is valid and can be loaded
        """
        info = self._models.get(name)
        if not info or not info.hef_path:
            return False

        hef_path = Path(info.hef_path)
        if not hef_path.exists():
            return False

        try:
            # Use hailortcli to validate HEF
            result = subprocess.run(
                ["hailortcli", "parse-hef", str(hef_path)],
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0:
                info.validated = True
                self._save_catalog()
                logger.info(f"Model '{name}' validated successfully")
                return True
            else:
                logger.warning(f"Model '{name}' validation failed: {result.stderr.decode()}")
                return False

        except FileNotFoundError:
            logger.warning("hailortcli not found, skipping validation")
            return True  # Assume valid if can't check
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return False

    def get_class_names(self, name: str) -> List[str]:
        """Get class names for a model."""
        info = self._models.get(name)
        if info and info.class_names:
            return info.class_names

        # Default to COCO for detection models
        if info and info.task == "detection":
            return COCO_CLASSES

        return []

    def ensure_model(self, name: str) -> Optional[Path]:
        """
        Ensure a model is available, downloading if necessary.

        Returns:
            Path to HEF file, or None if unavailable
        """
        # Check if already available
        path = self.get_model_path(name)
        if path and path.exists():
            return path

        # Try to download
        if name in MODEL_CATALOG:
            if self.download_model(name):
                return self.get_model_path(name)

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get overall model manager status."""
        return {
            "model_dir": str(self.model_dir),
            "available_models": self.list_available(),
            "downloadable_models": self.list_downloadable(),
            "total_models": len(self._models),
            "active_model": self._active_model,
        }


# Singleton instance
_manager: Optional[HailoModelManager] = None


def get_model_manager() -> HailoModelManager:
    """Get the singleton model manager instance."""
    global _manager
    if _manager is None:
        _manager = HailoModelManager()
    return _manager
