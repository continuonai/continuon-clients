#!/usr/bin/env python3
"""
Quick test script for agent chat manager multi-turn learning.
This is a simplified version for quick verification.
"""

import requests
import json
import time
import os

BASE_URL = os.environ.get("CONTINUON_API_URL", "http://localhost:8082")
ENDPOINT = f"{BASE_URL}/api/training/chat_learn"

def quick_test():
    """Run a quick test of the chat learning endpoint."""
    print("="*60)
    print("Quick Test: Agent Chat Manager Multi-Turn Learning")
    print("="*60)
    print(f"Server: {BASE_URL}")
    print()
    
    # Test 1: Basic HOPE with Gemma subagent
    print("Test 1: HOPE Agent Manager with Gemma Subagent")
    print("-" * 60)
    
    payload = {
        "turns": 4,
        "model_hint": "hope-v1",
        "delegate_model_hint": "google/gemma-370m",
        "topic": "JAX training pipeline and multi-turn learning"
    }
    
    try:
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print("\nSending request...")
        
        t0 = time.time()
        resp = requests.post(
            ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        dt = time.time() - t0
        
        print(f"Response time: {dt:.2f}s")
        print(f"Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            # Handle wrapped response
            if "result" in data:
                result = data["result"]
            else:
                result = data
            
            print(f"\n✓ Success!")
            print(f"  Status: {result.get('status', 'unknown')}")
            
            history = result.get("history", [])
            outputs = result.get("outputs", [])
            
            print(f"  History turns: {len(history)}")
            print(f"  Outputs: {len(outputs)}")
            
            # Show conversation
            if history:
                print(f"\n  Conversation:")
                for i, turn in enumerate(history):
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    # Truncate long content
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"    [{i+1}] {role}: {preview}")
            
            # Check for training data indicators
            all_text = json.dumps(result)
            if "mock" in all_text.lower() or "fallback" in all_text.lower():
                print(f"\n  ⚠ Warning: Mock/fallback responses detected")
            else:
                print(f"\n  ✓ Dynamic conversation detected")
            
            return True
        else:
            print(f"\n✗ Failed: HTTP {resp.status_code}")
            print(f"  Response: {resp.text[:500]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to server at {BASE_URL}")
        print(f"  Make sure the server is running.")
        return False
    except requests.exceptions.Timeout:
        print(f"\n✗ Request timed out")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = quick_test()
    sys.exit(0 if success else 1)
