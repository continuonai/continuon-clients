#!/usr/bin/env python3
"""
10-Hour Robust Autonomous Training Session (Hailo Accelerated)
- Duration: 10 Hours
- Acceleration: Hailo-8 NPU (VisionCore)
- Intelligence: FunctionGemma-270m (Fast) / Gemma-3-4B (Deep)
- Resilience: Auto-swap management, Memory compaction
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
    print(f"[{event['timestamp'][11:19]}] {event_type}")

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
            print(f"  ⚠️  Timeout attempt {attempt+1}/3 ({timeout}s)")
            time.sleep(10)
        except Exception as e:
            print(f"  ⚠️  Error attempt {attempt+1}/3: {e}")
            time.sleep(5)
    return {"error": "max retries exceeded"}

def check_memory():
    """Check memory status with STRICT swap limits and AUTO-REMEDIATION"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 10-hour run requires strict hygiene
        is_ok = mem.percent < 90 and swap.percent < 85
        
        if not is_ok:
             print(f"  ⚠️  Memory Pressure: RAM={mem.percent}% Swap={swap.percent}%")
             print("  🧹 Triggering emergency cleanup...")
             try:
                import gc
                gc.collect()
                
                # Check if we should recommend expansion
                if swap.percent > 95:
                    print("  🚨 CRITICAL SWAP! Run 'sudo ./scripts/expand_swap.sh 4096' immediately.")
             except:
                pass

        return {
            "mem_percent": mem.percent,
            "swap_percent": swap.percent,
            "available_mb": mem.available // (1024*1024),
            "ok": is_ok
        }
    except:
        return {"ok": True, "mem_percent": 0}

def check_hailo_status():
    """Verify Hailo NPU availability details via detected hardware"""
    try:
        status = api_call("GET", "/api/status", timeout=5)
        hardware = status.get("detected_hardware", {})
        devices = hardware.get("devices", {})
        accelerators = devices.get("ai_accelerator", [])
        
        is_active = False
        if accelerators:
            # Check if any accelerator is active
            for acc in accelerators:
                if acc.get("config", {}).get("runtime_status") == "active":
                    is_active = True
                    break
        
        return {"available": is_active, "raw": accelerators}
    except:
        return {"available": False, "raw": []}

def run_wavecore():
    """Run WaveCore training"""
    result = api_call("POST", "/api/training/wavecore_loops", {
        "preset": "pi5",
        "fast_loops": 1,
        "mid_loops": 1,
        "slow_loops": 1, 
        "use_jit": False,
        "use_hailo": True # Hint to use NPU if applicable
    }, timeout=300)
    return result

def run_cms_compact():
    """Run CMS memory compaction"""
    result = api_call("POST", "/api/cms/compact", {})
    gc.collect()
    return result

def run_chat_learn(topic: str = None):
    """Run a multi-turn curiosity cycle with Local Model Priority"""
    
    # 1. Preferred: FunctionGemma (Fastest on CPU)
    # 2. Secondary: Gemma-3-4B (Deepest, matches user intent)
    candidate_paths = [
        Path.home() / "models/functiongemma-270m-it",
        Path.home() / "models/gemma-3-4b-it",
        Path("/opt/continuonos/models/functiongemma-270m-it"),
        Path("/opt/continuonos/models/gemma-3-4b-it"),
    ]
    
    model_hint = "google/gemma-3-4b-it" # Fallback default
    
    # Select first available local model
    for path in candidate_paths:
        if path.exists():
            model_hint = str(path)
            break
            
    payload = {
        "turns": 2, 
        "delegate_model_hint": f"consult:{model_hint}"
    }
    if topic:
        payload["topic"] = topic
    
    print(f"  ... Requesting ChatLearn (Model: {Path(model_hint).name}) ...")
    # 10 min timeout for inference safety
    result = api_call("POST", "/api/learning/chat_learn", payload, timeout=600)
    return result

def run_curriculum_lesson(lesson_id: str):
    """Trigger a specific curriculum lesson."""
    return api_call("POST", "/api/curriculum/run", {"lesson_id": lesson_id}, timeout=30)

def main():
    print("=" * 60)
    print("  10-HOUR ROBUST TRAINING SESSION (HAILO ENABLED)")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End:   {(datetime.now() + timedelta(hours=DURATION_HOURS)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Mode:  Autonomous | Self-Healing | NPU-Accelerated")
    print("=" * 60)
    
    log_event("session_start", {
        "duration_hours": DURATION_HOURS,
        "mode": "hailo_10hr"
    })
    
    end_time = datetime.now() + timedelta(hours=DURATION_HOURS)
    cycle = 0
    successful_cycles = 0
    failed_cycles = 0
    
    while datetime.now() < end_time:
        cycle += 1
        remaining_hrs = (end_time - datetime.now()).total_seconds() / 3600
        
        print(f"\n[{datetime.now().strftime('%H:%M')}] CYCLE {cycle} | {remaining_hrs:.1f}h remaining")
        
        # 1. Health Check (Memory & Hailo)
        mem = check_memory()
        hailo = check_hailo_status()
        
        log_event("health_check", {"memory": mem, "hailo": hailo})
        print(f"  Health: RAM={mem.get('mem_percent')}% Swap={mem.get('swap_percent')}% Hailo={hailo.get('available')}")
        
        if not mem.get("ok", True):
            print("  ⚠️  High Memory - Performing aggressive compaction and skipping cycle...")
            run_cms_compact()
            gc.collect()
            time.sleep(60)
            continue

        # 2. WaveCore (The Brain)
        print("  🌊 Running WaveCore...")
        wc_result = run_wavecore()
        if "error" in wc_result:
            print(f"  ❌ WaveCore Error: {wc_result.get('error')}")
            failed_cycles += 1
        else:
            loss = wc_result.get("fast", {}).get("result", {}).get("final_loss", 0)
            print(f"  ✅ WaveCore Success (Loss: {loss:.4f})")
            successful_cycles += 1
            
        # 3. Memory Compaction (Routine)
        run_cms_compact()
        
        # 4. Curiosity & Discovery (Odd Cycles)
        if cycle % 2 == 1:
            print("  🤔 Running Discovery (ChatLearn)...")
            cl_result = run_chat_learn()
            if "error" in cl_result:
                 print(f"  ❌ ChatLearn Error: {cl_result.get('error')}")
            else:
                 print(f"  ✅ ChatLearn Complete")
                 
        # 5. Physics Lab (Every 4th Cycle)
        if cycle % 4 == 0:
            print("  🧪 Running Physics Lab...")
            run_curriculum_lesson("physics-lab")
            
        # 6. Agentic Self-Improvement (Every 6th Cycle)
        if cycle % 6 == 0:
            print("  🛠️ Running Agentic Lessons...")
            run_curriculum_lesson("coding-basics")
            run_curriculum_lesson("visual-monitor")
            
        # 7. Checkpoint
        if cycle % 4 == 0:
            log_event("checkpoint", {"cycle": cycle, "success": successful_cycles})
            print("  💾 Checkpoint Saved")
            
        # Sleep
        print(f"  ⏳ Sleeping {CYCLE_INTERVAL_MIN} min...")
        time.sleep(CYCLE_INTERVAL_MIN * 60)

    print("\n" + "="*60)
    print(f"  SESSION COMPLETE | Cycles: {cycle} | Success: {successful_cycles}")
    print("="*60)

if __name__ == "__main__":
    main()
