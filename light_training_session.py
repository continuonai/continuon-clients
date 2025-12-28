#!/usr/bin/env python3
"""
Lighter training session with memory management.
- Shorter cycles
- Memory cleanup between runs
- No chat validation (to save memory)
"""

import json
import time
import requests
import gc
from datetime import datetime, timedelta
from pathlib import Path

API_BASE = "http://127.0.0.1:8081"
LOG_FILE = Path("/home/craigm26/Downloads/ContinuonXR/training_light_log.jsonl")
DURATION_HOURS = 2  # Reduced from 4

def log_event(event_type: str, data: dict = None):
    """Append event to JSONL log"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "data": data or {}
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    print(f"[{event['timestamp'][:19]}] {event_type}")

def api_call(method: str, endpoint: str, data: dict = None, timeout: int = 120):
    """Make API call with error handling"""
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

def run_wavecore_light():
    """Run WaveCore with reduced settings"""
    result = api_call("POST", "/api/training/wavecore_loops", {
        "preset": "pi5",
        "fast_loops": 1,  # Reduced
        "mid_loops": 1,
        "slow_loops": 1,
        "use_jit": False
    }, timeout=180)
    return result

def run_cms_compact():
    """Run memory compaction"""
    result = api_call("POST", "/api/cms/compact", {})
    gc.collect()  # Force Python GC
    return result

def check_memory():
    """Check if memory is okay to continue"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "mem_percent": mem.percent,
            "swap_percent": swap.percent,
            "ok": mem.percent < 85 and swap.percent < 80
        }
    except:
        return {"ok": True}

def main():
    print("=" * 60)
    print("  LIGHT TRAINING SESSION")
    print("  Duration: 2 hours, with memory management")
    print("=" * 60)
    
    log_event("session_start", {"duration_hours": DURATION_HOURS, "mode": "light"})
    
    end_time = datetime.now() + timedelta(hours=DURATION_HOURS)
    cycle = 0
    
    while datetime.now() < end_time:
        cycle += 1
        print(f"\n{'='*40}")
        print(f"  CYCLE {cycle}")
        print(f"  Time remaining: {(end_time - datetime.now()).seconds // 60} min")
        print(f"{'='*40}")
        
        # Check memory before proceeding
        mem = check_memory()
        log_event("memory_check", mem)
        
        if not mem.get("ok", True):
            print("⚠️  Memory pressure detected, running cleanup...")
            gc.collect()
            time.sleep(30)  # Let system recover
            continue
        
        # WaveCore training
        print("\n🌊 Running WaveCore loops...")
        wc_result = run_wavecore_light()
        log_event("wavecore", wc_result)
        
        # CMS compaction
        print("📦 Running CMS compaction...")
        cms_result = run_cms_compact()
        log_event("cms_compact", cms_result)
        
        # Cleanup
        gc.collect()
        
        # Wait between cycles (10 min)
        print(f"\n⏳ Waiting 10 minutes until next cycle...")
        time.sleep(600)
    
    log_event("session_complete", {"cycles": cycle})
    print(f"\n✅ Training complete! {cycle} cycles finished.")

if __name__ == "__main__":
    main()
