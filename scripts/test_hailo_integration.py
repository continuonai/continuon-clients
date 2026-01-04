#!/usr/bin/env python3
"""
Test script for Hailo AI Accelerator Integration.

Tests:
1. HailoModelManager - Model discovery and management
2. HailoPipeline - Async inference pipeline
3. YOLOv8 Worker - Object detection
4. VisionCore - Integrated perception with fallback

Usage:
    python scripts/test_hailo_integration.py
    python scripts/test_hailo_integration.py --test model_manager
    python scripts/test_hailo_integration.py --test pipeline
    python scripts/test_hailo_integration.py --test vision_core
    python scripts/test_hailo_integration.py --benchmark
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_model_manager():
    """Test HailoModelManager functionality."""
    print("\n" + "=" * 60)
    print("Testing HailoModelManager")
    print("=" * 60)

    try:
        from continuonbrain.services.hailo_model_manager import (
            HailoModelManager,
            get_model_manager,
        )

        manager = get_model_manager()

        print(f"\nModel Directory: {manager.model_dir}")
        print(f"Available Models: {manager.list_available()}")
        print(f"Downloadable Models: {manager.list_downloadable()}")

        # Check each available model
        for name in manager.list_available():
            info = manager.get_model_info(name)
            path = manager.get_model_path(name)
            print(f"\n  Model: {name}")
            print(f"    Task: {info.task if info else 'unknown'}")
            print(f"    Path: {path}")
            print(f"    Exists: {path.exists() if path else False}")

        # Test model validation
        for name in manager.list_available()[:1]:  # Just first model
            print(f"\n  Validating {name}...")
            valid = manager.validate_model(name)
            print(f"    Valid: {valid}")

        print("\n  Status:")
        status = manager.get_status()
        for k, v in status.items():
            print(f"    {k}: {v}")

        print("\n[PASS] HailoModelManager test completed")
        return True

    except Exception as e:
        print(f"\n[FAIL] HailoModelManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline():
    """Test HailoPipeline functionality."""
    print("\n" + "=" * 60)
    print("Testing HailoPipeline")
    print("=" * 60)

    try:
        from continuonbrain.services.hailo_pipeline import HailoPipeline

        pipeline = HailoPipeline()

        print(f"\nAvailable: {pipeline.is_available()}")
        print(f"Models: {pipeline.get_available_models()}")

        if not pipeline.is_available():
            print("\n[SKIP] No models available for pipeline test")
            return True

        # Create test image
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some structure
        test_frame[100:200, 100:200] = [255, 0, 0]  # Red square
        test_frame[300:400, 400:500] = [0, 255, 0]  # Green square

        print("\nStarting pipeline...")
        await pipeline.start()

        # Test detection
        print("\nRunning detection...")
        start = time.time()
        result = await pipeline.detect(test_frame)
        elapsed = (time.time() - start) * 1000

        print(f"  OK: {result.ok}")
        print(f"  Error: {result.error}")
        print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"  Total Time: {elapsed:.1f}ms")
        print(f"  Detections: {len(result.data.get('detections', []))}")

        # Print detections
        for det in result.data.get("detections", [])[:5]:
            print(f"    - {det['label']}: {det['confidence']:.2f}")

        # Test batch
        print("\nRunning batch detection (3 frames)...")
        frames = [test_frame.copy() for _ in range(3)]
        start = time.time()
        results = await pipeline.detect_batch(frames)
        elapsed = (time.time() - start) * 1000

        print(f"  Batch Time: {elapsed:.1f}ms ({elapsed/3:.1f}ms per frame)")
        print(f"  Results: {[r.ok for r in results]}")

        # Stats
        print("\nPipeline Stats:")
        stats = pipeline.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        await pipeline.stop()

        print("\n[PASS] HailoPipeline test completed")
        return True

    except Exception as e:
        print(f"\n[FAIL] HailoPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_core():
    """Test VisionCore with Hailo integration."""
    print("\n" + "=" * 60)
    print("Testing VisionCore (Hailo Integration)")
    print("=" * 60)

    try:
        from continuonbrain.services.vision_core import VisionCore, create_vision_core

        # Create VisionCore with all features
        print("\nInitializing VisionCore...")
        core = create_vision_core(
            enable_hailo=True,
            enable_sam3=False,  # Skip SAM3 for this test
            enable_depth=False,  # Skip OAK-D for this test
            use_hailo_pipeline=True,
            enable_cpu_fallback=True,
        )

        print("\nCapabilities:")
        caps = core.get_capabilities()
        for k, v in caps.items():
            print(f"  {k}: {v}")

        # Create test image
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add colored regions for potential detection
        test_frame[100:200, 100:200] = [255, 128, 0]  # Orange
        test_frame[300:400, 400:500] = [0, 128, 255]  # Light blue

        # Test perception
        print("\nRunning perception...")
        start = time.time()
        scene = core.perceive(rgb_frame=test_frame, run_detection=True)
        elapsed = (time.time() - start) * 1000

        print(f"  Total Time: {elapsed:.1f}ms")
        print(f"  Hailo Inference: {scene.hailo_inference_ms:.1f}ms")
        print(f"  Has Detection: {scene.has_detection}")
        print(f"  Objects Found: {scene.object_count}")

        for obj in scene.objects[:5]:
            print(f"    - {obj.label} ({obj.source}): {obj.confidence:.2%}")

        # Test stats
        print("\nPipeline Stats:")
        stats = core.get_pipeline_stats()
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {k}: {v}")

        core.close()

        print("\n[PASS] VisionCore test completed")
        return True

    except Exception as e:
        print(f"\n[FAIL] VisionCore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_benchmark():
    """Run performance benchmark."""
    print("\n" + "=" * 60)
    print("Running Hailo Benchmark")
    print("=" * 60)

    try:
        from continuonbrain.services.hailo_pipeline import HailoPipeline

        pipeline = HailoPipeline()

        if not pipeline.is_available():
            print("\n[SKIP] No models available for benchmark")
            return

        # Create test frames
        num_frames = 30
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]

        print(f"\nBenchmarking {num_frames} frames...")

        # Without pipeline (sequential)
        print("\n1. Sequential (no pipeline):")
        times = []
        for frame in frames:
            start = time.time()
            result = pipeline._run_inference_sync(frame, "detection")
            times.append((time.time() - start) * 1000)

        avg = sum(times) / len(times)
        fps = 1000 / avg if avg > 0 else 0
        print(f"   Avg: {avg:.1f}ms | FPS: {fps:.1f}")

        # With pipeline
        print("\n2. With Pipeline (async):")
        await pipeline.start()

        start = time.time()
        results = await pipeline.detect_batch(frames)
        total = (time.time() - start) * 1000
        per_frame = total / num_frames

        print(f"   Total: {total:.1f}ms | Per Frame: {per_frame:.1f}ms | FPS: {1000/per_frame:.1f}")
        print(f"   Success: {sum(1 for r in results if r.ok)}/{len(results)}")

        await pipeline.stop()

        print("\n[PASS] Benchmark completed")

    except Exception as e:
        print(f"\n[FAIL] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


def check_hailo_device():
    """Check if Hailo device is available."""
    print("\n" + "=" * 60)
    print("Checking Hailo Device")
    print("=" * 60)

    import subprocess
    from pathlib import Path

    # Check /dev/hailo*
    hailo_devs = list(Path("/dev").glob("hailo*"))
    print(f"\nHailo devices in /dev: {[str(d) for d in hailo_devs]}")

    # Check hailortcli
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            timeout=5,
        )
        print(f"\nhailortcli identify:\n{result.stdout.decode()}")
    except FileNotFoundError:
        print("\nhailortcli not found")
    except Exception as e:
        print(f"\nhailortcli error: {e}")

    # Check hailo_platform import
    try:
        import hailo_platform
        print(f"\nhailo_platform version: {getattr(hailo_platform, '__version__', 'unknown')}")
    except ImportError:
        print("\nhailo_platform not importable")


def main():
    parser = argparse.ArgumentParser(description="Test Hailo AI Integration")
    parser.add_argument(
        "--test",
        choices=["all", "device", "model_manager", "pipeline", "vision_core"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Hailo AI Accelerator Integration Test")
    print("=" * 60)

    results = []

    if args.test in ("all", "device"):
        check_hailo_device()

    if args.test in ("all", "model_manager"):
        results.append(("ModelManager", test_model_manager()))

    if args.test in ("all", "pipeline"):
        results.append(("Pipeline", asyncio.run(test_pipeline())))

    if args.test in ("all", "vision_core"):
        results.append(("VisionCore", test_vision_core()))

    if args.benchmark:
        asyncio.run(run_benchmark())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")

    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
