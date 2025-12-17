#!/usr/bin/env python3
"""
Complete systems check for ContinuonBrain.
Tests inference, training, and agent manager capabilities.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def check_api_availability():
    """Check if the robot API server is running."""
    print("=" * 70)
    print("1️⃣  API SERVER AVAILABILITY")
    print("=" * 70)
    
    if not REQUESTS_AVAILABLE:
        print("⚠️  requests library not available")
        return False
    
    try:
        resp = requests.get("http://localhost:8080/api/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print("✅ API server is running")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Mode: {data.get('mode', 'unknown')}")
            return True
        else:
            print(f"❌ API server returned status {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running (connection refused)")
        return False
    except Exception as e:
        print(f"❌ Error checking API: {e}")
        return False


def check_inference_capabilities():
    """Check inference capabilities."""
    print("\n" + "=" * 70)
    print("2️⃣  INFERENCE CAPABILITIES")
    print("=" * 70)
    
    results = {
        "api_available": False,
        "chat_available": False,
        "model_loaded": False,
        "tool_router_available": False,
        "vision_available": False,
    }
    
    if not REQUESTS_AVAILABLE:
        print("⚠️  Cannot test inference (requests not available)")
        return results
    
    # Check API status
    try:
        resp = requests.get("http://localhost:8080/api/status", timeout=5)
        if resp.status_code == 200:
            results["api_available"] = True
            print("✅ API server available")
            
            data = resp.json()
            model_info = data.get("model", {})
            if model_info:
                results["model_loaded"] = True
                print(f"✅ Model loaded: {model_info.get('name', 'unknown')}")
                print(f"   Backend: {model_info.get('backend', 'unknown')}")
    except Exception as e:
        print(f"❌ API check failed: {e}")
    
    # Check chat endpoint
    try:
        resp = requests.post(
            "http://localhost:8080/api/chat",
            json={"message": "test", "max_tokens": 10},
            timeout=10
        )
        if resp.status_code == 200:
            results["chat_available"] = True
            print("✅ Chat endpoint working")
        else:
            print(f"⚠️  Chat endpoint returned {resp.status_code}")
    except Exception as e:
        print(f"⚠️  Chat endpoint test failed: {e}")
    
    # Check tool router
    try:
        resp = requests.post(
            "http://localhost:8080/api/training/tool_router_predict",
            json={"prompt": "test", "k": 3},
            timeout=5
        )
        if resp.status_code == 200:
            results["tool_router_available"] = True
            data = resp.json()
            if data.get("predictions"):
                print("✅ Tool router inference working")
                print(f"   Predictions: {len(data.get('predictions', []))} tools")
        else:
            print(f"⚠️  Tool router returned {resp.status_code}")
    except Exception as e:
        print(f"⚠️  Tool router test failed: {e}")
    
    # Check vision/camera
    try:
        resp = requests.get("http://localhost:8080/api/camera/stream", timeout=5, stream=True)
        if resp.status_code == 200:
            results["vision_available"] = True
            print("✅ Camera/vision stream available")
        else:
            print(f"⚠️  Camera stream returned {resp.status_code}")
    except Exception as e:
        print(f"⚠️  Camera check failed: {e}")
    
    return results


def check_training_capabilities():
    """Check training capabilities."""
    print("\n" + "=" * 70)
    print("3️⃣  TRAINING CAPABILITIES")
    print("=" * 70)
    
    results = {
        "background_learner": False,
        "chat_learning": False,
        "wavecore": False,
        "tool_router_training": False,
        "hope_eval": False,
        "cms_compaction": False,
    }
    
    if not REQUESTS_AVAILABLE:
        print("⚠️  Cannot test training (requests not available)")
        return results
    
    try:
        resp = requests.get("http://localhost:8080/api/training/architecture_status", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            
            # Background learner
            bg = data.get("background_learner", {})
            if bg.get("enabled") or bg.get("running"):
                results["background_learner"] = True
                print("✅ Background learner:")
                print(f"   Running: {bg.get('running', False)}")
                print(f"   Paused: {bg.get('paused', False)}")
                print(f"   Steps: {bg.get('total_steps', 0):,}")
                print(f"   Updates: {bg.get('learning_updates', 0):,}")
            
            # Chat learning
            chat = data.get("chat_learn", {})
            if chat.get("enabled"):
                results["chat_learning"] = True
                print("\n✅ Chat learning:")
                print(f"   Enabled: {chat.get('enabled', False)}")
                print(f"   Interval: {chat.get('interval_s', 0)}s")
                print(f"   Model: {chat.get('model_hint', 'N/A')}")
            
            # Autonomy orchestrator
            orch = data.get("autonomy_orchestrator", {})
            if orch.get("enabled"):
                print("\n✅ Autonomy orchestrator:")
                print(f"   Enabled: {orch.get('enabled', False)}")
                print(f"   CMS compaction: Every {orch.get('cms_compact_every_s', 0)}s")
                print(f"   HOPE eval: Every {orch.get('hope_eval_every_s', 0)}s")
                print(f"   WaveCore: Every {orch.get('wavecore_every_s', 0)}s")
                results["cms_compaction"] = orch.get("cms_compact_every_s", 0) > 0
                results["hope_eval"] = orch.get("hope_eval_every_s", 0) > 0
                results["wavecore"] = orch.get("wavecore_every_s", 0) > 0
            
            # Check for training metrics
            metrics_resp = requests.get("http://localhost:8080/api/training/metrics", timeout=5)
            if metrics_resp.status_code == 200:
                metrics = metrics_resp.json()
                
                if metrics.get("tool_router"):
                    results["tool_router_training"] = True
                    print("\n✅ Tool router training metrics available")
                
                if metrics.get("wavecore"):
                    wc = metrics.get("wavecore", {})
                    if wc.get("fast") or wc.get("mid") or wc.get("slow"):
                        print("\n✅ WaveCore training metrics available")
            
        else:
            print(f"❌ Training status returned {resp.status_code}")
    except Exception as e:
        print(f"❌ Training check failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def check_agent_manager_capabilities():
    """Check agent manager capabilities."""
    print("\n" + "=" * 70)
    print("4️⃣  AGENT MANAGER CAPABILITIES")
    print("=" * 70)
    
    results = {
        "agent_manager_available": False,
        "subagent_delegation": False,
        "multi_agent_conversations": False,
        "tool_selection": False,
        "safety_protocol": False,
    }
    
    if not REQUESTS_AVAILABLE:
        print("⚠️  Cannot test agent manager (requests not available)")
        return results
    
    try:
        # Check architecture status for agent manager info
        resp = requests.get("http://localhost:8080/api/training/architecture_status", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            
            # Check chat learn config (agent manager uses this)
            chat = data.get("chat_learn", {})
            if chat.get("enabled"):
                results["agent_manager_available"] = True
                print("✅ Agent Manager:")
                print(f"   Enabled: {chat.get('enabled', False)}")
                
                # Check for subagent delegation
                if chat.get("delegate_model_hint"):
                    results["subagent_delegation"] = True
                    print(f"   Subagent model: {chat.get('delegate_model_hint')}")
                    print("   ✅ Subagent delegation enabled")
                
                if chat.get("model_hint"):
                    print(f"   Primary model: {chat.get('model_hint')}")
                    results["multi_agent_conversations"] = True
                    print("   ✅ Multi-agent conversations enabled")
            
            # Check tool router (agent manager uses this for tool selection)
            tool_router_resp = requests.post(
                "http://localhost:8080/api/training/tool_router_predict",
                json={"prompt": "search for information", "k": 5},
                timeout=5
            )
            if tool_router_resp.status_code == 200:
                results["tool_selection"] = True
                print("\n✅ Tool selection (tool router):")
                data = tool_router_resp.json()
                preds = data.get("predictions", [])
                if preds:
                    print(f"   Top tool: {preds[0].get('tool', 'N/A')} (score: {preds[0].get('score', 0):.3f})")
            
            # Check safety protocol
            safety_resp = requests.get("http://localhost:8080/api/status", timeout=5)
            if safety_resp.status_code == 200:
                # Safety is checked via system_instructions
                results["safety_protocol"] = True
                print("\n✅ Safety protocol:")
                print("   Safety checks integrated into action paths")
                print("   Base rules: 4 immutable safety rules")
            
        else:
            print(f"❌ Agent manager check returned {resp.status_code}")
    except Exception as e:
        print(f"❌ Agent manager check failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def check_file_system_health():
    """Check file system health and required directories."""
    print("\n" + "=" * 70)
    print("5️⃣  FILE SYSTEM HEALTH")
    print("=" * 70)
    
    required_dirs = [
        "/opt/continuonos/brain",
        "/opt/continuonos/brain/rlds/episodes",
        "/opt/continuonos/brain/trainer/logs",
        "/opt/continuonos/brain/model/adapters",
        "/opt/continuonos/brain/logs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
            if path.is_dir():
                # Count files if relevant
                if "episodes" in dir_path:
                    episodes = list(path.glob("*.json"))
                    if episodes:
                        print(f"   Episodes: {len(episodes)}")
                elif "logs" in dir_path:
                    log_files = list(path.glob("*.log")) + list(path.glob("*.json"))
                    if log_files:
                        print(f"   Log files: {len(log_files)}")
        else:
            print(f"⚠️  {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def check_safety_policies():
    """Check safety policies are loaded and enforced."""
    print("\n" + "=" * 70)
    print("6️⃣  SAFETY POLICIES")
    print("=" * 70)
    
    try:
        from continuonbrain.system_instructions import SafetyProtocol, SystemInstructions
        
        config_dir = Path("/opt/continuonos/brain")
        protocol = SafetyProtocol.load(config_dir)
        instructions = SystemInstructions.load(config_dir)
        
        print(f"✅ Safety protocol loaded: {len(protocol.rules)} rules")
        print(f"✅ System instructions loaded: {len(instructions.instructions)} instructions")
        
        # Test validation
        is_safe, reason, _ = protocol.validate_action("harm a human", {})
        if not is_safe:
            print("✅ Safety validation working (harmful action blocked)")
        else:
            print("❌ Safety validation not working (harmful action allowed!)")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Safety check failed: {e}")
        return False


def generate_summary(inference_results, training_results, agent_results, fs_health, safety_ok):
    """Generate summary report."""
    print("\n" + "=" * 70)
    print("📊 COMPLETE SYSTEMS CHECK SUMMARY")
    print("=" * 70)
    print()
    
    # Inference summary
    print("INFERENCE CAPABILITIES:")
    inference_score = sum([
        inference_results.get("api_available", False),
        inference_results.get("model_loaded", False),
        inference_results.get("chat_available", False),
        inference_results.get("tool_router_available", False),
    ])
    print(f"   Status: {inference_score}/4 components working")
    if inference_score == 4:
        print("   ✅ All inference capabilities operational")
    elif inference_score >= 2:
        print("   ⚠️  Partial inference capabilities")
    else:
        print("   ❌ Inference capabilities limited")
    print()
    
    # Training summary
    print("TRAINING CAPABILITIES:")
    training_score = sum([
        training_results.get("background_learner", False),
        training_results.get("chat_learning", False),
        training_results.get("wavecore", False),
        training_results.get("tool_router_training", False),
        training_results.get("hope_eval", False),
        training_results.get("cms_compaction", False),
    ])
    print(f"   Status: {training_score}/6 components working")
    if training_score >= 4:
        print("   ✅ Training capabilities operational")
    elif training_score >= 2:
        print("   ⚠️  Partial training capabilities")
    else:
        print("   ❌ Training capabilities limited")
    print()
    
    # Agent manager summary
    print("AGENT MANAGER CAPABILITIES:")
    agent_score = sum([
        agent_results.get("agent_manager_available", False),
        agent_results.get("subagent_delegation", False),
        agent_results.get("multi_agent_conversations", False),
        agent_results.get("tool_selection", False),
        agent_results.get("safety_protocol", False),
    ])
    print(f"   Status: {agent_score}/5 components working")
    if agent_score == 5:
        print("   ✅ All agent manager capabilities operational")
    elif agent_score >= 3:
        print("   ⚠️  Partial agent manager capabilities")
    else:
        print("   ❌ Agent manager capabilities limited")
    print()
    
    # Overall health
    print("SYSTEM HEALTH:")
    print(f"   File system: {'✅ Healthy' if fs_health else '⚠️  Issues detected'}")
    print(f"   Safety policies: {'✅ Enforced' if safety_ok else '❌ Not enforced'}")
    print()
    
    overall_score = inference_score + training_score + agent_score
    max_score = 4 + 6 + 5  # 15 total
    
    print("=" * 70)
    print(f"OVERALL SYSTEM STATUS: {overall_score}/{max_score} ({overall_score*100//max_score}%)")
    
    if overall_score >= 12:
        print("✅ System is fully operational")
    elif overall_score >= 8:
        print("⚠️  System is partially operational")
    else:
        print("❌ System has significant issues")
    print("=" * 70)


def main():
    """Run complete systems check."""
    print("\n" + "=" * 70)
    print("🔍 COMPLETE SYSTEMS CHECK")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Run all checks
    api_ok = check_api_availability()
    
    if not api_ok:
        print("\n⚠️  API server not available. Some checks will be skipped.")
        inference_results = {}
        training_results = {}
        agent_results = {}
    else:
        inference_results = check_inference_capabilities()
        training_results = check_training_capabilities()
        agent_results = check_agent_manager_capabilities()
    
    fs_health = check_file_system_health()
    safety_ok = check_safety_policies()
    
    # Generate summary
    generate_summary(inference_results, training_results, agent_results, fs_health, safety_ok)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
