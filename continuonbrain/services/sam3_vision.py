"""
SAM Vision Service - Real-time Promptable Segmentation for Robotics

Provides:
- Point/box-prompted object segmentation using SAM2 (facebook/sam2-hiera-large)
- Automatic mask generation for scene understanding
- Hardware-aware inference (CUDA GPU > CPU)
- Integration with RLDS episode enrichment
- Real-time object detection for manipulation tasks

Supported Models:
- SAM2 (facebook/sam2-hiera-large) - Video-capable, excellent for robotics
- SAM (facebook/sam-vit-huge) - Original SAM model
- SAM-HQ - Higher quality masks
- SAM3 (facebook/sam3) - When available in transformers

Note: SAM2 uses point/box prompts. For text prompts, combine with GroundingDINO.
"""

from __future__ import annotations

import os
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

logger = logging.getLogger(__name__)


@dataclass
class SAMConfig:
    """Configuration for SAM Vision Service."""
    model_id: str = "facebook/sam3"  # Default to SAM3 (Segment Anything 3)
    model_type: str = "auto"  # auto, sam3, sam2, sam, sam_hq
    device: str = "auto"  # auto, cuda, cpu
    threshold: float = 0.5
    mask_threshold: float = 0.5
    cache_dir: Optional[str] = None
    max_instances: int = 10
    # For automatic mask generation
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95


# Alias for backward compatibility
SAM3Config = SAMConfig


@dataclass
class SegmentationResult:
    """Result from SAM3 segmentation."""
    prompt: str
    num_instances: int
    masks: List[Any]  # numpy arrays or torch tensors
    boxes_xyxy: List[List[float]]  # [[x1,y1,x2,y2], ...]
    scores: List[float]
    inference_time_ms: float
    device_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "prompt": self.prompt,
            "num_instances": self.num_instances,
            "boxes_xyxy": self.boxes_xyxy,
            "scores": self.scores,
            "inference_time_ms": self.inference_time_ms,
            "device_used": self.device_used,
        }


class SAMVisionService:
    """
    Hardware-aware SAM segmentation service.
    
    Supports SAM2, SAM, and SAM3 (when available).
    
    Usage:
        service = SAMVisionService()
        if service.is_available():
            # Point-based segmentation
            result = service.segment_points(image, points=[[100, 200]], labels=[1])
            
            # Automatic mask generation (find all objects)
            masks = service.generate_masks(image)
            print(f"Found {len(masks)} objects")
    """
    
    # Model registry with fallback order
    MODEL_REGISTRY = {
        "sam2": {
            "model_class": "Sam2Model",
            "processor_class": "Sam2Processor",
            "default_id": "facebook/sam2-hiera-large",
        },
        "sam": {
            "model_class": "SamModel", 
            "processor_class": "SamProcessor",
            "default_id": "facebook/sam-vit-huge",
        },
        "sam3": {
            "model_class": "Sam3Model",
            "processor_class": "Sam3Processor", 
            "default_id": "facebook/sam3",
        },
        "sam_hq": {
            "model_class": "SamHQModel",
            "processor_class": "SamHQProcessor",
            "default_id": "lkeab/hq-sam",
        },
    }
    
    def __init__(self, config: Optional[SAMConfig] = None):
        self.config = config or SAMConfig()
        self.model = None
        self.processor = None
        self.device = None
        self.model_type = None
        self._initialized = False
        self._available = False
        self._available_models: List[str] = []
        self._init_error: Optional[str] = None
        
        # Check which SAM models are available
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check which SAM model variants are available."""
        try:
            import torch
        except ImportError:
            self._init_error = "torch not installed"
            return False
        
        try:
            import transformers
        except ImportError:
            self._init_error = "transformers not installed"
            return False
        
        # Check each model type
        for model_type, info in self.MODEL_REGISTRY.items():
            try:
                model_cls = getattr(transformers, info["model_class"], None)
                proc_cls = getattr(transformers, info["processor_class"], None)
                if model_cls is not None and proc_cls is not None:
                    self._available_models.append(model_type)
            except Exception:
                pass
        
        self._available = len(self._available_models) > 0
        if not self._available:
            self._init_error = "No SAM models available in transformers"
        else:
            logger.info(f"Available SAM models: {self._available_models}")
        
        return self._available
    
    def is_available(self) -> bool:
        """Check if any SAM model is available for use."""
        return self._available
    
    def get_available_models(self) -> List[str]:
        """Get list of available SAM model types."""
        return self._available_models.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "available": self._available,
            "initialized": self._initialized,
            "model_id": self.config.model_id,
            "model_type": self.model_type,
            "available_models": self._available_models,
            "device": str(self.device) if self.device else "not_initialized",
            "error": self._init_error,
        }
    
    def initialize(self, force: bool = False, model_type: Optional[str] = None) -> bool:
        """
        Load SAM model into memory.
        
        Args:
            force: Re-initialize even if already loaded
            model_type: Specific model type to load ("sam2", "sam", "sam3", "sam_hq")
                       If None, uses config.model_type or auto-selects best available.
        
        Call this explicitly to control when model loading happens,
        or let segment methods auto-initialize on first call.
        """
        if self._initialized and not force:
            return True
        
        if not self._available:
            logger.error("Cannot initialize: No SAM models available")
            return False
        
        try:
            import torch
            import transformers
            
            # Determine which model to load
            if model_type:
                selected_type = model_type
            elif self.config.model_type != "auto":
                selected_type = self.config.model_type
            else:
                # Auto-select: prefer sam3 > sam2 > sam_hq > sam (best to older)
                priority = ["sam3", "sam2", "sam_hq", "sam"]
                selected_type = None
                for mt in priority:
                    if mt in self._available_models:
                        selected_type = mt
                        break
                if not selected_type:
                    selected_type = self._available_models[0]
            
            if selected_type not in self._available_models:
                self._init_error = f"Model type {selected_type} not available"
                return False
            
            self.model_type = selected_type
            model_info = self.MODEL_REGISTRY[selected_type]
            
            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info(f"SAM ({selected_type}): Using CUDA GPU")
                else:
                    self.device = torch.device("cpu")
                    logger.info(f"SAM ({selected_type}): Using CPU")
            else:
                self.device = torch.device(self.config.device)
            
            # Get model ID
            model_id = self.config.model_id
            if model_id == "facebook/sam2-hiera-large" and selected_type != "sam2":
                # Use default for selected type
                model_id = model_info["default_id"]
            
            logger.info(f"Loading {selected_type} model: {model_id}")
            start = time.time()
            
            # Get classes
            model_cls = getattr(transformers, model_info["model_class"])
            proc_cls = getattr(transformers, model_info["processor_class"])
            
            # Load model
            cache_kwargs = {}
            if self.config.cache_dir:
                cache_kwargs["cache_dir"] = self.config.cache_dir
            
            self.processor = proc_cls.from_pretrained(model_id, **cache_kwargs)
            self.model = model_cls.from_pretrained(model_id, **cache_kwargs).to(self.device)
            
            load_time = time.time() - start
            logger.info(f"{selected_type.upper()} loaded in {load_time:.1f}s on {self.device}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self._init_error = f"SAM initialization failed: {e}"
            logger.error(self._init_error)
            import traceback
            traceback.print_exc()
            return False
    
    def _load_image(self, image: Union[str, Path, Any]) -> "PIL.Image.Image":
        """Load image from various sources."""
        import numpy as np
        from PIL import Image as PILImage
        
        if isinstance(image, (str, Path)):
            return PILImage.open(str(image)).convert("RGB")
        elif hasattr(image, "convert"):  # PIL Image
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return PILImage.fromarray(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def segment_text(
        self,
        image: Union[str, Path, "PIL.Image.Image", "np.ndarray"],
        prompt: str,
        threshold: float = 0.5,
    ) -> Optional[SegmentationResult]:
        """
        Segment objects using text prompt (SAM3 feature).
        
        Args:
            image: Path to image, PIL Image, or numpy array (RGB)
            prompt: Text description like "cup", "hand", "robot gripper"
            threshold: Detection threshold
            
        Returns:
            SegmentationResult with masks for matching objects
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        # SAM3 supports text prompts
        if self.model_type != "sam3":
            logger.warning(f"Text prompts only supported with SAM3, current model: {self.model_type}")
            # Fall back to center point for non-SAM3
            pil_image = self._load_image(image)
            w, h = pil_image.size
            return self.segment_points(image, [[w//2, h//2]], [1])
        
        try:
            import torch
            import numpy as np
            
            start = time.time()
            pil_image = self._load_image(image)
            
            # SAM3 text prompt API
            inputs = self.processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process for SAM3
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            
            masks = results.get("masks")
            boxes = results.get("boxes")
            scores = results.get("scores")
            
            masks_list = []
            boxes_list = []
            scores_list = []
            
            if masks is not None and len(masks) > 0:
                num_masks = min(masks.shape[0], self.config.max_instances)
                for i in range(num_masks):
                    masks_list.append(masks[i].cpu().numpy())
                    if boxes is not None and i < len(boxes):
                        boxes_list.append(boxes[i].cpu().tolist())
                    if scores is not None and i < len(scores):
                        scores_list.append(float(scores[i].cpu()))
            
            inference_time = (time.time() - start) * 1000
            
            return SegmentationResult(
                prompt=prompt,
                num_instances=len(masks_list),
                masks=masks_list,
                boxes_xyxy=boxes_list,
                scores=scores_list,
                inference_time_ms=inference_time,
                device_used=str(self.device),
            )
            
        except Exception as e:
            logger.error(f"SAM3 text segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def segment_points(
        self,
        image: Union[str, Path, "PIL.Image.Image", "np.ndarray"],
        points: List[List[int]],
        labels: Optional[List[int]] = None,
    ) -> Optional[SegmentationResult]:
        """
        Segment objects at specific point locations.
        
        Args:
            image: Path to image, PIL Image, or numpy array (RGB)
            points: List of [x, y] coordinates to segment
            labels: List of labels (1=foreground, 0=background). Defaults to all 1s.
            
        Returns:
            SegmentationResult with masks for pointed objects
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        # SAM3 uses text prompts, not points - use text fallback
        if self.model_type == "sam3":
            logger.info("SAM3 uses text prompts. Using 'object' as default.")
            return self.segment_text(image, "object")
        
        if labels is None:
            labels = [1] * len(points)
        
        try:
            import torch
            import numpy as np
            
            start = time.time()
            pil_image = self._load_image(image)
            
            # SAM2/SAM format with points
            inputs = self.processor(
                images=pil_image,
                input_points=[points],
                input_labels=[labels],
                return_tensors="pt"
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.cpu().numpy().flatten()
            
            # Extract best masks
            masks_list = []
            scores_list = []
            
            if masks is not None and len(masks) > 0:
                # Get the best mask for each point (highest IoU score)
                for i in range(min(len(masks), self.config.max_instances)):
                    mask = masks[i]
                    if len(mask.shape) == 3:
                        # Take best mask (highest score)
                        best_idx = scores[i].argmax() if len(scores[i].shape) > 0 else 0
                        mask = mask[best_idx]
                    masks_list.append(mask.numpy() > 0.5)
                    if i < len(scores):
                        score = scores[i]
                        if hasattr(score, '__len__') and len(score) > 0:
                            scores_list.append(float(score.max()))
                        else:
                            scores_list.append(float(score))
            
            inference_time = (time.time() - start) * 1000
            
            return SegmentationResult(
                prompt=f"points:{points}",
                num_instances=len(masks_list),
                masks=masks_list,
                boxes_xyxy=[],  # Could compute from masks
                scores=scores_list,
                inference_time_ms=inference_time,
                device_used=str(self.device),
            )
            
        except Exception as e:
            logger.error(f"SAM point segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def segment_box(
        self,
        image: Union[str, Path, "PIL.Image.Image", "np.ndarray"],
        box: List[float],
    ) -> Optional[SegmentationResult]:
        """
        Segment object within a bounding box.
        
        Args:
            image: Path to image, PIL Image, or numpy array (RGB)
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            SegmentationResult with mask for boxed region
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            import torch
            
            start = time.time()
            pil_image = self._load_image(image)
            
            # Prepare inputs
            inputs = self.processor(
                images=pil_image,
                input_boxes=[[box]],
                return_tensors="pt"
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.cpu().numpy().flatten()
            
            masks_list = []
            scores_list = []
            
            if masks is not None and len(masks) > 0:
                mask = masks[0]
                if len(mask.shape) == 3:
                    best_idx = scores.argmax() if len(scores) > 0 else 0
                    mask = mask[best_idx]
                masks_list.append(mask.numpy() > 0.5)
                if len(scores) > 0:
                    scores_list.append(float(scores.max()))
            
            inference_time = (time.time() - start) * 1000
            
            return SegmentationResult(
                prompt=f"box:{box}",
                num_instances=len(masks_list),
                masks=masks_list,
                boxes_xyxy=[box],
                scores=scores_list,
                inference_time_ms=inference_time,
                device_used=str(self.device),
            )
            
        except Exception as e:
            logger.error(f"SAM box segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_masks(
        self,
        image: Union[str, Path, "PIL.Image.Image", "np.ndarray"],
        points_per_side: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate masks for all objects in image.
        
        This uses a grid of points to find all objects automatically.
        
        Args:
            image: Path to image, PIL Image, or numpy array (RGB)
            points_per_side: Grid density for automatic mask generation
            
        Returns:
            List of mask dictionaries with 'mask', 'score', 'area'
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        try:
            import numpy as np
            
            start = time.time()
            pil_image = self._load_image(image)
            w, h = pil_image.size
            
            # Generate grid of points
            points_per_side = points_per_side or self.config.points_per_side
            step_x = w // points_per_side
            step_y = h // points_per_side
            
            all_masks = []
            
            # Sample points and segment
            for y in range(step_y // 2, h, step_y):
                for x in range(step_x // 2, w, step_x):
                    result = self.segment_points(pil_image, [[x, y]], [1])
                    if result and result.num_instances > 0:
                        for i, mask in enumerate(result.masks):
                            area = float(np.sum(mask))
                            score = result.scores[i] if i < len(result.scores) else 0.0
                            
                            # Filter by thresholds
                            if score >= self.config.pred_iou_thresh:
                                all_masks.append({
                                    "mask": mask,
                                    "score": score,
                                    "area": area,
                                    "point": [x, y],
                                })
            
            # Remove duplicate/overlapping masks
            all_masks = self._filter_duplicate_masks(all_masks)
            
            inference_time = (time.time() - start) * 1000
            logger.info(f"Generated {len(all_masks)} masks in {inference_time:.0f}ms")
            
            return all_masks
            
        except Exception as e:
            logger.error(f"SAM mask generation failed: {e}")
            return []
    
    def _filter_duplicate_masks(self, masks: List[Dict]) -> List[Dict]:
        """Remove overlapping/duplicate masks, keeping highest scoring ones."""
        if len(masks) <= 1:
            return masks
        
        import numpy as np
        
        # Sort by score descending
        masks = sorted(masks, key=lambda x: x["score"], reverse=True)
        
        kept = []
        for mask_info in masks:
            mask = mask_info["mask"]
            is_duplicate = False
            
            for kept_info in kept:
                kept_mask = kept_info["mask"]
                # Check IoU
                intersection = np.logical_and(mask, kept_mask).sum()
                union = np.logical_or(mask, kept_mask).sum()
                iou = intersection / (union + 1e-6)
                
                if iou > 0.8:  # High overlap = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(mask_info)
        
        return kept
    
    def find_objects(
        self,
        image: Union[str, Path, "PIL.Image.Image", "np.ndarray"],
        min_area: int = 100,
        max_objects: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find all distinct objects in image.
        
        Uses automatic mask generation to discover objects.
        
        Args:
            image: Image to analyze
            min_area: Minimum mask area in pixels
            max_objects: Maximum objects to return
            
        Returns:
            List of object dictionaries with mask, score, bbox, center
        """
        import numpy as np
        
        masks = self.generate_masks(image)
        
        objects = []
        for mask_info in masks:
            mask = mask_info["mask"]
            area = mask_info["area"]
            
            if area < min_area:
                continue
            
            # Compute bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue
            
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            objects.append({
                "mask": mask,
                "score": mask_info["score"],
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(cx), float(cy)],
                "area": float(area),
            })
        
        # Sort by score and limit
        objects.sort(key=lambda x: x["score"], reverse=True)
        return objects[:max_objects]
    
    def get_manipulation_targets(
        self,
        image: Union[str, Path, "PIL.Image.Image", "np.ndarray"],
        min_area: int = 500,
        max_area: int = 500000,
    ) -> List[Dict[str, Any]]:
        """
        Find manipulation targets in image for robotics.
        
        Returns sorted list of objects suitable for grasping/manipulation.
        Filters by size appropriate for robot gripper.
        
        Args:
            image: Image to analyze
            min_area: Minimum object area (filter tiny objects)
            max_area: Maximum object area (filter background/large surfaces)
            
        Returns:
            List of graspable objects with mask, bbox, center
        """
        objects = self.find_objects(image, min_area=min_area)
        
        # Filter by manipulation-appropriate size
        targets = []
        for obj in objects:
            area = obj["area"]
            if area <= max_area:
                box = obj["box_xyxy"]
                width = box[2] - box[0]
                height = box[3] - box[1]
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                
                # Add graspability heuristics
                obj["width"] = float(width)
                obj["height"] = float(height)
                obj["aspect_ratio"] = float(aspect_ratio)
                obj["graspable"] = aspect_ratio < 5  # Not too elongated
                
                targets.append(obj)
        
        # Sort by score
        targets.sort(key=lambda x: x["score"], reverse=True)
        return targets
    
    def unload(self):
        """Free model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._initialized = False
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        logger.info("SAM3 model unloaded")


# Alias for backward compatibility  
SAM3VisionService = SAMVisionService


def create_sam_service(
    device: str = "auto",
    model_id: str = "facebook/sam3",
    model_type: str = "auto",
    auto_init: bool = False,
) -> SAMVisionService:
    """
    Factory function to create SAM vision service.
    
    Args:
        device: "auto", "cuda", or "cpu"
        model_id: HuggingFace model ID
        model_type: "auto", "sam2", "sam", "sam3", "sam_hq"
        auto_init: Load model immediately if True
        
    Returns:
        SAMVisionService instance
    """
    config = SAMConfig(
        model_id=model_id,
        model_type=model_type,
        device=device,
    )
    service = SAMVisionService(config)
    
    if auto_init and service.is_available():
        service.initialize()
    
    return service


# Alias for backward compatibility
create_sam3_service = create_sam_service


def main():
    """CLI for testing SAM vision service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM Vision Service (SAM2/SAM/SAM3)")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--point", type=str, help="Point to segment as 'x,y'")
    parser.add_argument("--box", type=str, help="Box to segment as 'x1,y1,x2,y2'")
    parser.add_argument("--auto", action="store_true", help="Auto-generate all masks")
    parser.add_argument("--model", type=str, default="auto", 
                       choices=["auto", "sam2", "sam", "sam3", "sam_hq"],
                       help="Model type to use")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--status", action="store_true", help="Show service status only")
    args = parser.parse_args()
    
    print("\n🎯 SAM Vision Service")
    print("=" * 60)
    
    service = create_sam_service(device=args.device, model_type=args.model)
    status = service.get_status()
    
    print(f"Available: {status['available']}")
    print(f"Available Models: {status['available_models']}")
    
    if args.status:
        if status['error']:
            print(f"Error: {status['error']}")
        return
    
    if not service.is_available():
        print(f"\n❌ No SAM models available: {status['error']}")
        print("\nTo install:")
        print("  pip install torch transformers")
        return
    
    if not args.image:
        print("\nUsage examples:")
        print("  # Point segmentation:")
        print("  python -m continuonbrain.services.sam3_vision --image photo.jpg --point 100,200")
        print()
        print("  # Box segmentation:")
        print("  python -m continuonbrain.services.sam3_vision --image photo.jpg --box 50,50,200,200")
        print()
        print("  # Auto-detect all objects:")
        print("  python -m continuonbrain.services.sam3_vision --image photo.jpg --auto")
        return
    
    print(f"\nLoading model (type: {args.model})...")
    if not service.initialize():
        print(f"❌ Failed to initialize: {service._init_error}")
        return
    
    print(f"Model: {service.model_type}")
    print(f"Device: {service.device}")
    
    if args.auto:
        print(f"\n🔍 Auto-detecting objects in {args.image}...")
        objects = service.find_objects(args.image)
        print(f"\n✅ Found {len(objects)} objects")
        for i, obj in enumerate(objects[:10]):
            box = obj['box_xyxy']
            print(f"   #{i}: score={obj['score']:.2f}, "
                  f"bbox=[{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}], "
                  f"area={int(obj['area'])}")
    
    elif args.point:
        x, y = map(int, args.point.split(','))
        print(f"\n📍 Segmenting at point ({x}, {y})...")
        result = service.segment_points(args.image, [[x, y]], [1])
        if result:
            print(f"\n✅ Found {result.num_instances} masks")
            print(f"   Inference: {result.inference_time_ms:.0f}ms")
            for i, score in enumerate(result.scores):
                print(f"   Mask #{i}: IoU score={score:.3f}")
        else:
            print("❌ Segmentation failed")
    
    elif args.box:
        coords = list(map(float, args.box.split(',')))
        print(f"\n📦 Segmenting box {coords}...")
        result = service.segment_box(args.image, coords)
        if result:
            print(f"\n✅ Segmented box region")
            print(f"   Inference: {result.inference_time_ms:.0f}ms")
            if result.scores:
                print(f"   IoU score: {result.scores[0]:.3f}")
        else:
            print("❌ Segmentation failed")
    
    else:
        print("\nSpecify --point, --box, or --auto to segment")


if __name__ == "__main__":
    main()

