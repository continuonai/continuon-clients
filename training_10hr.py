#!/usr/bin/env python3
"""
16-Hour Extended Training Session
- Robust memory management
- Automatic recovery
- Periodic checkpointing
"""

import json
import time
import requests
import gc
import os
from datetime import datetime, timedelta
from pathlib import Path

API_BASE = "http://127.0.0.1:8081"
LOG_FILE = Path("/home/craigm26/Downloads/ContinuonXR/training_10hr_log.jsonl")
DURATION_HOURS = 10
CYCLE_INTERVAL_MIN = 15  # Run every 15 minutes

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

def api_call(method: str, endpoint: str, data: dict = None, timeout: int = 180):
    """Make API call with error handling and retries"""
    url = f"{API_BASE}{endpoint}"
    for attempt in range(3):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=timeout)
            else:
                resp = requests.post(url, json=data or {}, timeout=timeout)
            return resp.json() if resp.text else {}
        except requests.exceptions.Timeout:
            print(f"  Timeout attempt {attempt+1}/3")
            time.sleep(10)
        except Exception as e:
            print(f"  Error attempt {attempt+1}/3: {e}")
            time.sleep(5)
    return {"error": "max retries exceeded"}

def check_memory():
    """Check memory status"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "mem_percent": mem.percent,
            "swap_percent": swap.percent,
            "available_mb": mem.available // (1024*1024),
            "ok": mem.percent < 85 and swap.percent < 90
        }
    except:
        return {"ok": True, "mem_percent": 0}

def run_wavecore():
    """Run WaveCore training"""
    result = api_call("POST", "/api/training/wavecore_loops", {
        "preset": "pi5",
        "fast_loops": 1,
        "mid_loops": 1,
        "slow_loops": 1,
        "use_jit": False
    }, timeout=300)
    return result

def run_cms_compact():
    """Run CMS memory compaction"""
    result = api_call("POST", "/api/cms/compact", {})
    gc.collect()
    return result

def run_chat_learn(topic: str = None):
    """Run a multi-turn curiosity cycle"""
    payload = {
        "turns": 6,
        "delegate_model_hint": "consult:google/gemma-3-4b-it"
    }
    if topic:
        payload["topic"] = topic
    
    result = api_call("POST", "/api/learning/chat_learn", payload, timeout=600)
    return result

def check_server_health():
    """Check if server is responsive"""
    try:
        resp = requests.get(f"{API_BASE}/api/status", timeout=10)
        return resp.status_code == 200
    except:
        return False

def main():
    print("=" * 60)
    print("  10-HOUR EXTENDED TRAINING SESSION")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End:   {(datetime.now() + timedelta(hours=DURATION_HOURS)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    log_event("session_start", {
        "duration_hours": DURATION_HOURS,
        "cycle_interval_min": CYCLE_INTERVAL_MIN,
        "mode": "extended"
    })
    
    end_time = datetime.now() + timedelta(hours=DURATION_HOURS)
    cycle = 0
    successful_cycles = 0
    failed_cycles = 0
    
    while datetime.now() < end_time:
        cycle += 1
        remaining = (end_time - datetime.now())
        remaining_hours = remaining.total_seconds() / 3600
        
        print(f"\n{'='*50}")
        print(f"  CYCLE {cycle} | {remaining_hours:.1f} hours remaining")
        print(f"  Success: {successful_cycles} | Failed: {failed_cycles}")
        print(f"{'='*50}")
        
        # Memory check
        mem = check_memory()
        log_event("memory_check", mem)
        print(f"  Memory: {mem.get('mem_percent', 0):.1f}% | Available: {mem.get('available_mb', 0)}MB")
        
        if not mem.get("ok", True):
            print("  ⚠️  Memory pressure - waiting 60s...")
            gc.collect()
            time.sleep(60)
            continue
        
        # Server health check
        if not check_server_health():
            print("  ⚠️  Server not responding - waiting 30s...")
            log_event("server_error", {"message": "Server unresponsive"})
            time.sleep(30)
            continue
        
        # WaveCore training
        print("  🌊 Running WaveCore...")
        wc_result = run_wavecore()
        
        if "error" in wc_result:
            print(f"  ❌ WaveCore error: {wc_result.get('error')}")
            log_event("wavecore_error", wc_result)
            failed_cycles += 1
        else:
            # Extract losses
            fast_loss = wc_result.get("fast", {}).get("result", {}).get("final_loss", 0)
            mid_loss = wc_result.get("mid", {}).get("result", {}).get("final_loss", 0)
            slow_loss = wc_result.get("slow", {}).get("result", {}).get("final_loss", 0)
            print(f"  ✅ WaveCore: Fast={fast_loss:.4f} Mid={mid_loss:.4f} Slow={slow_loss:.4f}")
            log_event("wavecore", wc_result)
            successful_cycles += 1
        
        # CMS compaction
        print("  📦 Running CMS compaction...")
        cms_result = run_cms_compact()
        log_event("cms_compact", cms_result)
        
        # Chat Learn (Discovery & Self-Improvement)
        if cycle % 2 == 1:
            print("  🤔 Running Curiosity/Discovery loop (Self-Improvement focus)...")
            topic = "Self-improvement across system objectives: World Model (Mamba), CMS loops, and Contextual Search."
            cl_result = run_chat_learn(topic=topic)
            log_event("chat_learn", cl_result)
            if "error" in cl_result:
                print(f"  ❌ ChatLearn error: {cl_result.get('error')}")
            else:
                print(f"  ✅ ChatLearn discovery complete. Teacher: Google Gemma-3-4B-IT")
        
        # Cleanup
        gc.collect()
        
        # Checkpoint every 10 cycles
        if cycle % 10 == 0:
            log_event("checkpoint", {
                "cycle": cycle,
                "successful": successful_cycles,
                "failed": failed_cycles
            })
            print(f"  💾 Checkpoint saved at cycle {cycle}")
        
        # Wait for next cycle
        print(f"\n  ⏳ Next cycle in {CYCLE_INTERVAL_MIN} minutes...")
        time.sleep(CYCLE_INTERVAL_MIN * 60)
    
    # Session complete
    log_event("session_complete", {
        "total_cycles": cycle,
        "successful_cycles": successful_cycles,
        "failed_cycles": failed_cycles
    })
    
    print(f"\n{'='*60}")
    print("  10-HOUR TRAINING SESSION COMPLETE!")
    print(f"  Total Cycles: {cycle}")
    print(f"  Successful: {successful_cycles}")
    print(f"  Failed: {failed_cycles}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

