#!/usr/bin/env python3
"""
Fresh Brain Training & Benchmark Script

Runs on a freshly reset brain with Hailo acceleration:
1. Initial WaveCore training cycles
2. Hailo inference tests
3. Progressive benchmark

Usage:
    python scripts/fresh_brain_training.py
    python scripts/fresh_brain_training.py --cycles 5
    python scripts/fresh_brain_training.py --skip-training
"""

import argparse
import asyncio
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_BASE = "http://127.0.0.1:8081"
LOG_FILE = Path("/home/craigm26/Downloads/ContinuonXR/fresh_brain_training.log")


def log(msg: str):
    """Log with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def api_call(method: str, endpoint: str, data: dict = None, timeout: int = 180):
    """Make API call."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=data or {}, timeout=timeout)
        return resp.json() if resp.text else {}
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def check_server():
    """Check if server is running."""
    try:
        resp = requests.get(f"{API_BASE}/api/status", timeout=5)
        return resp.status_code == 200
    except:
        return False


def get_hailo_status():
    """Get Hailo accelerator status."""
    try:
        status = api_call("GET", "/api/status", timeout=5)
        hw = status.get("detected_hardware", {}).get("devices", {})
        accelerators = hw.get("ai_accelerator", [])
        for acc in accelerators:
            if "Hailo" in acc.get("name", ""):
                cfg = acc.get("config", {})
                return {
                    "available": True,
                    "model": cfg.get("model"),
                    "tops": cfg.get("tops"),
                    "status": cfg.get("runtime_status"),
                    "firmware": cfg.get("firmware_version"),
                }
        return {"available": False}
    except:
        return {"available": False}


def run_wavecore_training(use_hailo: bool = True):
    """Run WaveCore training loop."""
    log("Running WaveCore training...")
    result = api_call("POST", "/api/training/wavecore_loops", {
        "preset": "pi5",
        "fast_loops": 1,
        "mid_loops": 1,
        "slow_loops": 1,
        "use_jit": False,
        "use_hailo": use_hailo,
    }, timeout=300)

    if "error" in result:
        log(f"  Error: {result['error']}")
        return None

    # Extract metrics
    fast_loss = result.get("fast", {}).get("result", {}).get("final_loss", 0)
    mid_loss = result.get("mid", {}).get("result", {}).get("final_loss", 0)
    slow_loss = result.get("slow", {}).get("result", {}).get("final_loss", 0)

    log(f"  Fast loss: {fast_loss:.4f}")
    log(f"  Mid loss: {mid_loss:.4f}")
    log(f"  Slow loss: {slow_loss:.4f}")

    return {
        "fast_loss": fast_loss,
        "mid_loss": mid_loss,
        "slow_loss": slow_loss,
    }


def run_cms_compact():
    """Run CMS memory compaction."""
    log("Running CMS compaction...")
    result = api_call("POST", "/api/cms/compact", {}, timeout=30)
    gc.collect()
    return result


def test_hailo_inference():
    """Test Hailo inference via VisionCore."""
    log("Testing Hailo inference...")

    try:
        import numpy as np
        from continuonbrain.services.vision_core import create_vision_core

        # Create VisionCore with Hailo
        core = create_vision_core(
            enable_hailo=True,
            use_hailo_pipeline=True,
            enable_sam3=False,
            enable_depth=False,
        )

        caps = core.get_capabilities()
        log(f"  Hailo available: {caps.get('hailo_detection')}")
        log(f"  Pipeline available: {caps.get('hailo_pipeline')}")

        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run detection
        times = []
        detections_count = []

        for i in range(5):
            start = time.time()
            scene = core.perceive(rgb_frame=test_frame, run_detection=True)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            detections_count.append(scene.object_count)

        avg_time = sum(times) / len(times)
        fps = 1000 / avg_time if avg_time > 0 else 0

        log(f"  Avg inference: {avg_time:.1f}ms")
        log(f"  FPS: {fps:.1f}")
        log(f"  Detections: {detections_count}")

        stats = core.get_pipeline_stats()
        log(f"  Hailo success: {stats.get('hailo_success', 0)}")
        log(f"  Fallback used: {stats.get('fallback_used', 0)}")

        core.close()

        return {
            "avg_ms": avg_time,
            "fps": fps,
            "hailo_success": stats.get("hailo_success", 0),
        }

    except Exception as e:
        log(f"  Error: {e}")
        return {"error": str(e)}


async def test_hailo_pipeline_async():
    """Test async Hailo pipeline."""
    log("Testing async Hailo pipeline...")

    try:
        import numpy as np
        from continuonbrain.services.hailo_pipeline import HailoPipeline

        pipeline = HailoPipeline()

        if not pipeline.is_available():
            log("  No models available")
            return {"error": "no models"}

        log(f"  Models: {pipeline.get_available_models()}")

        await pipeline.start()

        # Create test frames
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]

        # Batch detection
        start = time.time()
        results = await pipeline.detect_batch(frames)
        elapsed = (time.time() - start) * 1000

        success = sum(1 for r in results if r.ok)
        per_frame = elapsed / len(frames)
        fps = 1000 / per_frame if per_frame > 0 else 0

        log(f"  Batch of {len(frames)}: {elapsed:.1f}ms total")
        log(f"  Per frame: {per_frame:.1f}ms")
        log(f"  FPS: {fps:.1f}")
        log(f"  Success: {success}/{len(frames)}")

        stats = pipeline.get_stats()
        log(f"  Pipeline stats: {stats}")

        await pipeline.stop()

        return {
            "batch_ms": elapsed,
            "per_frame_ms": per_frame,
            "fps": fps,
            "success_rate": success / len(frames),
        }

    except Exception as e:
        log(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_progressive_benchmark():
    """Run progressive benchmark."""
    log("Running progressive benchmark...")

    results = []

    # Training iterations with increasing complexity
    for i in range(3):
        log(f"\n--- Benchmark Iteration {i+1}/3 ---")

        # Training
        train_start = time.time()
        train_result = run_wavecore_training(use_hailo=True)
        train_time = time.time() - train_start

        # Memory cleanup
        run_cms_compact()
        gc.collect()

        # Inference test
        infer_start = time.time()
        infer_result = test_hailo_inference()
        infer_time = time.time() - infer_start

        results.append({
            "iteration": i + 1,
            "train_time_s": train_time,
            "train_result": train_result,
            "infer_time_s": infer_time,
            "infer_result": infer_result,
        })

        log(f"  Train time: {train_time:.1f}s")
        log(f"  Infer time: {infer_time:.1f}s")

        # Brief pause
        time.sleep(5)

    return results


def main():
    parser = argparse.ArgumentParser(description="Fresh Brain Training & Benchmark")
    parser.add_argument("--cycles", type=int, default=3, help="Training cycles")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference tests")
    args = parser.parse_args()

    # Clear log
    LOG_FILE.write_text("")

    print("=" * 60)
    print("  FRESH BRAIN TRAINING & BENCHMARK")
    print("  Hailo-8 NPU Accelerated")
    print("=" * 60)

    # Check server
    log("Checking server...")
    if not check_server():
        log("ERROR: Server not running at " + API_BASE)
        return 1
    log("  Server OK")

    # Check Hailo
    log("Checking Hailo accelerator...")
    hailo = get_hailo_status()
    if hailo.get("available"):
        log(f"  Model: {hailo.get('model')}")
        log(f"  TOPS: {hailo.get('tops')}")
        log(f"  Status: {hailo.get('status')}")
    else:
        log("  WARNING: Hailo not available")

    results = {
        "started_at": datetime.now().isoformat(),
        "hailo": hailo,
        "training": [],
        "inference": None,
        "pipeline": None,
        "benchmark": None,
    }

    # Training cycles
    if not args.skip_training:
        log(f"\n=== TRAINING ({args.cycles} cycles) ===")
        for i in range(args.cycles):
            log(f"\n--- Training Cycle {i+1}/{args.cycles} ---")
            train_result = run_wavecore_training(use_hailo=True)
            results["training"].append(train_result)
            run_cms_compact()
            gc.collect()
            time.sleep(3)

    # Inference tests
    if not args.skip_inference:
        log("\n=== INFERENCE TESTS ===")

        # Sync VisionCore test
        results["inference"] = test_hailo_inference()

        # Async pipeline test
        log("\n--- Async Pipeline Test ---")
        results["pipeline"] = asyncio.run(test_hailo_pipeline_async())

    # Progressive benchmark
    log("\n=== PROGRESSIVE BENCHMARK ===")
    results["benchmark"] = run_progressive_benchmark()

    # Summary
    results["completed_at"] = datetime.now().isoformat()

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    if results["training"]:
        losses = [t.get("slow_loss", 0) for t in results["training"] if t]
        if losses:
            print(f"  Training: {len(losses)} cycles, final loss: {losses[-1]:.4f}")

    if results["inference"]:
        infer = results["inference"]
        if "fps" in infer:
            print(f"  Inference: {infer['fps']:.1f} FPS ({infer['avg_ms']:.1f}ms)")

    if results["pipeline"]:
        pipe = results["pipeline"]
        if "fps" in pipe:
            print(f"  Pipeline: {pipe['fps']:.1f} FPS (batch)")

    # Save results
    results_file = Path("/home/craigm26/Downloads/ContinuonXR/fresh_brain_results.json")
    results_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to: {results_file}")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
