#!/usr/bin/env python3
"""
Autonomous Inference Test - Real Hardware Pipeline

Tests the full inference stack:
1. OAK-D camera capture (RGB + Depth)
2. Hailo-8 accelerated detection (YOLOv8)
3. SAM3 segmentation for detailed masks
4. Autonomous mode: continuous detection loop

Usage:
    python -m continuonbrain.tests.autonomous_inference_test
    python -m continuonbrain.tests.autonomous_inference_test --duration 30
    python -m continuonbrain.tests.autonomous_inference_test --no-sam3  # Hailo only
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import io

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def run_autonomous_test(
    duration_seconds: float = 10.0,
    use_sam3: bool = True,
    use_hailo: bool = True,
    save_frames: bool = False,
):
    """
    Run autonomous inference test with real hardware.
    
    Args:
        duration_seconds: How long to run the test
        use_sam3: Whether to use SAM3 for segmentation
        use_hailo: Whether to use Hailo for detection
        save_frames: Whether to save captured frames
    """
    print_header("🤖 AUTONOMOUS INFERENCE TEST")
    print(f"Duration: {duration_seconds}s")
    print(f"SAM3: {'enabled' if use_sam3 else 'disabled'}")
    print(f"Hailo: {'enabled' if use_hailo else 'disabled'}")
    
    # ========================================
    # 1. Initialize Hardware
    # ========================================
    print_header("1. Initializing Hardware")
    
    # Initialize OAK-D camera
    camera = None
    try:
        from continuonbrain.sensors.oak_depth import OAKDepthCapture
        camera = OAKDepthCapture()
        if camera.initialize() and camera.start():
            print("✅ OAK-D camera initialized")
            meta = camera.get_camera_metadata()
            print(f"   Resolution: {meta.get('rgb_resolution', 'unknown')}")
        else:
            print("⚠️  OAK-D not available, using mock frames")
            camera = None
    except Exception as e:
        print(f"⚠️  OAK-D init failed: {e}")
        camera = None
    
    # Initialize Hailo
    hailo_vision = None
    if use_hailo:
        try:
            from continuonbrain.services.hailo_vision import HailoVision
            hailo_vision = HailoVision()
            state = hailo_vision.get_state()
            if state.available:
                print(f"✅ Hailo-8 initialized ({state.hef_path})")
            else:
                print("⚠️  Hailo not available")
                hailo_vision = None
        except Exception as e:
            print(f"⚠️  Hailo init failed: {e}")
            hailo_vision = None
    
    # Initialize SAM3
    sam_service = None
    if use_sam3:
        try:
            from continuonbrain.services.sam3_vision import create_sam_service
            sam_service = create_sam_service(device="cpu")
            if sam_service.is_available():
                print("✅ SAM3 service available")
                print(f"   Models: {sam_service.get_available_models()}")
                # Don't initialize yet - do on first use to save memory
            else:
                print("⚠️  SAM3 not available")
                sam_service = None
        except Exception as e:
            print(f"⚠️  SAM3 init failed: {e}")
            sam_service = None
    
    # ========================================
    # 2. Run Autonomous Loop
    # ========================================
    print_header("2. Starting Autonomous Mode")
    print("Press Ctrl+C to stop early\n")
    
    stats = {
        "frames_captured": 0,
        "hailo_detections": 0,
        "sam3_segmentations": 0,
        "objects_found": [],
        "errors": 0,
    }
    
    start_time = time.time()
    frame_count = 0
    sam_initialized = False
    
    try:
        while (time.time() - start_time) < duration_seconds:
            elapsed = time.time() - start_time
            frame_count += 1
            
            # Status line
            print(f"\r[{elapsed:5.1f}s] Frame {frame_count}", end="", flush=True)
            
            # ----------------------------------------
            # Capture frame
            # ----------------------------------------
            frame_data = None
            rgb_image = None
            depth_image = None
            
            if camera:
                frame_data = camera.capture_frame()
                if frame_data:
                    rgb_image = frame_data.get("rgb")
                    depth_image = frame_data.get("depth")
                    stats["frames_captured"] += 1
            
            if rgb_image is None:
                # Mock frame for testing without camera
                import numpy as np
                rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                stats["frames_captured"] += 1
            
            # ----------------------------------------
            # Hailo Detection
            # ----------------------------------------
            hailo_result = None
            if hailo_vision and rgb_image is not None:
                try:
                    # Convert to JPEG for Hailo
                    try:
                        import cv2
                        _, jpeg_bytes = cv2.imencode(".jpg", rgb_image)
                        jpeg_bytes = jpeg_bytes.tobytes()
                    except ImportError:
                        from PIL import Image
                        pil_img = Image.fromarray(rgb_image)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=85)
                        jpeg_bytes = buf.getvalue()
                    
                    hailo_result = hailo_vision.infer_jpeg(jpeg_bytes)
                    
                    if hailo_result.get("ok"):
                        stats["hailo_detections"] += 1
                        topk = hailo_result.get("topk", [])
                        if topk:
                            top_score = topk[0].get("score", 0)
                            top_idx = topk[0].get("index", -1)
                            print(f" | Hailo: idx={top_idx} score={top_score:.2f}", end="")
                except Exception as e:
                    stats["errors"] += 1
            
            # ----------------------------------------
            # SAM3 Segmentation (every 10th frame to save resources)
            # ----------------------------------------
            if sam_service and frame_count % 10 == 0 and rgb_image is not None:
                try:
                    # Initialize on first use
                    if not sam_initialized:
                        print("\n   Initializing SAM3 (first use)...", end="", flush=True)
                        if sam_service.initialize():
                            sam_initialized = True
                            print(f" done! (model: {sam_service.model_type})")
                        else:
                            print(" failed!")
                            sam_service = None
                            continue
                    
                    # Find objects using text prompt (SAM3's key feature)
                    from PIL import Image
                    pil_img = Image.fromarray(rgb_image)
                    
                    # Use text prompt for SAM3
                    result = sam_service.segment_text(pil_img, "object")
                    
                    if result and result.num_instances > 0:
                        stats["sam3_segmentations"] += 1
                        print(f" | SAM3: {result.num_instances} '{result.prompt}' ({result.inference_time_ms:.0f}ms)", end="")
                        
                except Exception as e:
                    stats["errors"] += 1
                    print(f" | SAM3 error: {e}", end="")
            
            # ----------------------------------------
            # Depth info
            # ----------------------------------------
            if depth_image is not None:
                import numpy as np
                valid_depth = depth_image[depth_image > 0]
                if len(valid_depth) > 0:
                    min_d = valid_depth.min()
                    print(f" | Depth: {min_d}mm", end="")
            
            print("", flush=True)  # Newline
            
            # Small delay to not overwhelm
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    
    # ========================================
    # 3. Cleanup
    # ========================================
    print_header("3. Cleanup")
    
    if camera:
        camera.stop()
        print("✅ Camera stopped")
    
    if sam_service and sam_initialized:
        sam_service.unload()
        print("✅ SAM3 unloaded")
    
    # ========================================
    # 4. Results
    # ========================================
    total_time = time.time() - start_time
    fps = stats["frames_captured"] / total_time if total_time > 0 else 0
    
    print_header("4. Results")
    print(f"Duration:           {total_time:.1f}s")
    print(f"Frames captured:    {stats['frames_captured']}")
    print(f"Frame rate:         {fps:.1f} FPS")
    print(f"Hailo detections:   {stats['hailo_detections']}")
    print(f"SAM3 segmentations: {stats['sam3_segmentations']}")
    print(f"Errors:             {stats['errors']}")
    
    print("\n✅ Autonomous inference test complete!")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Autonomous Inference Test")
    parser.add_argument("--duration", type=float, default=10.0, 
                       help="Test duration in seconds (default: 10)")
    parser.add_argument("--no-sam3", action="store_true",
                       help="Disable SAM3 segmentation")
    parser.add_argument("--no-hailo", action="store_true",
                       help="Disable Hailo detection")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save captured frames to disk")
    args = parser.parse_args()
    
    run_autonomous_test(
        duration_seconds=args.duration,
        use_sam3=not args.no_sam3,
        use_hailo=not args.no_hailo,
        save_frames=args.save_frames,
    )


if __name__ == "__main__":
    main()

