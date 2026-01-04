#!/usr/bin/env python3
"""
Test script for Hope agent chat and learning capabilities.
Tests both direct BrainService access and API endpoints.
"""
import sys
from pathlib import Path

# Add both repo and parent directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PARENT_ROOT = REPO_ROOT.parent

for p in [str(REPO_ROOT), str(PARENT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import time


def test_hope_chat():
    """Test Hope agent chat via BrainService."""
    print("=" * 60)
    print("Testing HOPE Agent Chat")
    print("=" * 60)

    from continuonbrain.services.brain_service import BrainService
    import uuid

    print("\n1. Initializing BrainService...")
    brain = BrainService(config_dir="/tmp/continuonbrain_test")

    print("\n2. Checking chat agent status...")
    if hasattr(brain, 'gemma_chat') and brain.gemma_chat:
        print("   ✅ Chat agent available")
        info = brain.gemma_chat.get_model_info() if hasattr(brain.gemma_chat, 'get_model_info') else {}
        print(f"   Model: {info.get('model_name', 'unknown')}")
        print(f"   Device: {info.get('device', 'unknown')}")
    else:
        print("   ⚠️  Chat agent not available, will use fallback")

    # Generate a session ID for multi-turn context
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    print(f"\n3. Testing MULTI-TURN chat (Session: {session_id})...")

    test_messages = [
        "Hello, who are you?",
        "What can you do?",
        "Can you remember what I just asked you?",
        "What was my first question to you?",
    ]

    # Track conversation history for non-session fallback
    history = []

    for i, msg in enumerate(test_messages, 1):
        print(f"\n   [Turn {i}] User: {msg}")
        try:
            # Use session_id for persistent multi-turn context
            response = brain.ChatWithGemma(msg, history=history, session_id=session_id)
            answer = response.get("response", response.get("reply", "No response"))
            confidence = response.get("confidence")
            agent = response.get("agent", "unknown")

            # Truncate long responses for display
            display_answer = answer[:300] + "..." if len(answer) > 300 else answer
            print(f"   Agent: {display_answer}")
            print(f"   [Agent: {agent}, Confidence: {confidence}]")

            # Update history for fallback
            history.append({"role": "user", "content": msg})
            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    return brain


def test_hope_learning(brain):
    """Test Hope agent learning capabilities."""
    print("\n" + "=" * 60)
    print("Testing HOPE Agent Learning")
    print("=" * 60)

    print("\n1. Checking learning status...")
    try:
        if hasattr(brain, 'get_learning_status'):
            status = brain.get_learning_status()
            print(f"   Learning enabled: {status.get('enabled', 'unknown')}")
            print(f"   Episodes logged: {status.get('episodes_logged', 0)}")
    except Exception as e:
        print(f"   ⚠️  Could not get learning status: {e}")

    print("\n2. Testing multi-turn learning conversation...")
    try:
        import asyncio

        async def run_chat_learn():
            result = await brain.RunChatLearn({
                "turns": 3,
                "topic": "robot capabilities",
                "model_hint": "hope-v1",
            })
            return result

        result = asyncio.run(run_chat_learn())
        print(f"   Turns completed: {result.get('turns_completed', 0)}")
        print(f"   Learning active: {result.get('learning_active', False)}")

        if result.get('conversation'):
            print("\n   Conversation snippet:")
            for turn in result['conversation'][:2]:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')[:100]
                print(f"   {role}: {content}...")
    except Exception as e:
        print(f"   ⚠️  Chat learn not available: {e}")

    print("\n3. Checking HOPE brain metrics...")
    try:
        if hasattr(brain, 'hope_brain') and brain.hope_brain:
            metrics = brain.get_hope_metrics()
            print(f"   CMS utilization: {metrics.get('cms_utilization', 'N/A')}")
            print(f"   Stability: {metrics.get('stability', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Could not get HOPE metrics: {e}")


def test_api_endpoints():
    """Test REST API endpoints."""
    print("\n" + "=" * 60)
    print("Testing REST API Endpoints")
    print("=" * 60)

    import requests
    import uuid

    base_url = "http://localhost:8080"

    # Test status endpoint
    print("\n1. Testing /api/status...")
    try:
        resp = requests.get(f"{base_url}/api/status", timeout=5)
        if resp.ok:
            status = resp.json()
            print(f"   Server mode: {status.get('mode', 'unknown')}")
            print(f"   Chat agent: {status.get('chat_agent', 'unknown')}")
        else:
            print(f"   ⚠️  Status code: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ API not available: {e}")
        print("   (Is the server running? Start with: python -m continuonbrain --port 8080)")
        return

    # Test multi-turn chat via API
    print("\n2. Testing /api/chat (Multi-turn)...")
    session_id = f"api-test-{uuid.uuid4().hex[:8]}"
    chat_messages = [
        "Hello! My name is TestUser.",
        "What is my name?",
    ]
    history = []

    for i, msg in enumerate(chat_messages, 1):
        print(f"\n   [Turn {i}] User: {msg}")
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "message": msg,
                    "history": history,
                    "session_id": session_id,
                },
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                answer = data.get("response", data.get("reply", "No response"))
                display = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"   Agent: {display}")
                print(f"   [Confidence: {data.get('confidence')}]")
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": answer})
            else:
                print(f"   ⚠️  Status code: {resp.status_code}")
                print(f"   Response: {resp.text[:200]}")
        except Exception as e:
            print(f"   ❌ Chat failed: {e}")

    # Test HOPE metrics endpoint
    print("\n3. Testing /api/hope/metrics...")
    try:
        resp = requests.get(f"{base_url}/api/hope/metrics", timeout=5)
        if resp.ok:
            metrics = resp.json()
            if "error" in metrics:
                print(f"   ⚠️  HOPE not initialized: {metrics['error']}")
            else:
                print(f"   Lyapunov total: {metrics.get('lyapunov', {}).get('total', 'N/A')}")
                print(f"   Steps: {metrics.get('steps', 0)}")
        else:
            print(f"   ⚠️  Status code: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Endpoint not available: {e}")

    # Test learning status
    print("\n4. Testing /api/learning/status...")
    try:
        resp = requests.get(f"{base_url}/api/learning/status", timeout=5)
        if resp.ok:
            status = resp.json()
            print(f"   Learning enabled: {status.get('enabled', 'unknown')}")
            print(f"   Episodes: {status.get('episodes_logged', 0)}")
        else:
            print(f"   ⚠️  Status code: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Endpoint not available: {e}")

    # Test chat/events SSE endpoint (quick check)
    print("\n5. Testing /api/chat/events (SSE)...")
    try:
        resp = requests.get(f"{base_url}/api/chat/events", timeout=2, stream=True)
        if resp.status_code == 200:
            print("   ✅ SSE endpoint available")
        else:
            print(f"   ⚠️  Status code: {resp.status_code}")
    except requests.exceptions.Timeout:
        print("   ✅ SSE endpoint available (timeout expected for streaming)")
    except Exception as e:
        print(f"   ❌ SSE not available: {e}")


if __name__ == "__main__":
    print("\n🤖 ContinuonBrain Hope Agent Test Suite\n")

    # Test direct BrainService
    brain = test_hope_chat()

    # Test learning
    test_hope_learning(brain)

    # Test API (if server is running)
    test_api_endpoints()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
