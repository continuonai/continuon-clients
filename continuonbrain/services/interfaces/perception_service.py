"""
Perception Service Interface

Protocol definition for perception/vision services.
"""
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class IPerceptionService(Protocol):
    """
    Protocol for perception/vision services.

    Implementations handle:
    - Object detection
    - Scene understanding
    - Depth processing
    - Visual segmentation
    """

    def detect_objects(
        self,
        frame: Optional[np.ndarray] = None,
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in current or provided frame.

        Args:
            frame: Optional RGB frame. If None, captures from camera.
            conf_threshold: Confidence threshold for detections

        Returns:
            List of detection dictionaries:
                - label: str - Object class label
                - confidence: float - Detection confidence
                - bbox: tuple - (x1, y1, x2, y2) bounding box
                - class_id: int - Class ID
                - depth_mm: float - Estimated depth (if available)
        """
        ...

    def describe_scene(
        self,
        frame: Optional[np.ndarray] = None,
    ) -> str:
        """
        Get natural language description of current scene.

        Args:
            frame: Optional RGB frame. If None, captures from camera.

        Returns:
            Natural language description of the scene
        """
        ...

    def get_scene_representation(
        self,
        frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Get full scene representation with depth, objects, etc.

        Args:
            frame: Optional RGB frame. If None, captures from camera.

        Returns:
            Dictionary containing:
                - objects: list - Detected objects
                - depth_map: ndarray - Depth map (if available)
                - description: str - Scene description
                - statistics: dict - Scene statistics
                - timestamp: float - Capture timestamp
        """
        ...

    def segment(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Segment objects in frame using text or point prompts.

        Args:
            frame: RGB frame
            prompt: Text prompt for segmentation
            points: List of (x, y) points to segment around

        Returns:
            Dictionary containing:
                - masks: list - List of segmentation masks
                - scores: list - Confidence scores
                - boxes: list - Bounding boxes
        """
        ...

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get perception capability flags.

        Returns:
            Dictionary of capability name -> available boolean:
                - detection: bool
                - segmentation: bool
                - depth: bool
                - scene_description: bool
        """
        ...

    def is_available(self) -> bool:
        """Check if perception service is available."""
        ...
