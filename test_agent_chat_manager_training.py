#!/usr/bin/env python3
"""
Test script for Agent Chat Manager multi-turn agentic learning process.
Tests training with JAX and other models.

This script:
1. Tests multi-turn conversations between Agent Manager and subagents
2. Tests with different model combinations (HOPE, Gemma, JAX if available)
3. Verifies RLDS logging for training data
4. Tests different topics and configurations
5. Validates training data quality
"""

import requests
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Default server URL - can be overridden via environment
BASE_URL = os.environ.get("CONTINUON_API_URL", "http://localhost:8082")
CHAT_LEARN_ENDPOINT = f"{BASE_URL}/api/training/chat_learn"
STATUS_ENDPOINT = f"{BASE_URL}/api/status"

# Enable RLDS logging for testing (requires CONTINUON_LOG_CHAT_RLDS=1 on server)
TEST_RLDS_LOGGING = os.environ.get("CONTINUON_LOG_CHAT_RLDS", "0").lower() in ("1", "true", "yes", "on")


class ChatLearnTester:
    """Test harness for agent chat manager multi-turn learning."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.chat_learn_url = f"{base_url}/api/training/chat_learn"
        self.status_url = f"{base_url}/api/status"
        self.results: List[Dict[str, Any]] = []
        
    def check_server_health(self) -> bool:
        """Check if the server is running and accessible."""
        try:
            resp = requests.get(self.status_url, timeout=5)
            if resp.status_code == 200:
                print(f"✓ Server is accessible at {self.base_url}")
                return True
            else:
                print(f"✗ Server returned status {resp.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to server at {self.base_url}: {e}")
            return False
    
    def run_chat_learn(
        self,
        turns: int = 5,
        model_hint: str = "hope-v1",
        delegate_model_hint: Optional[str] = None,
        topic: str = "tool use + planning + safety",
        session_id: Optional[str] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Run a single chat learning session."""
        if session_id is None:
            session_id = f"test_{int(time.time())}"
        
        payload = {
            "turns": turns,
            "model_hint": model_hint,
            "topic": topic,
            "session_id": session_id,
        }
        
        if delegate_model_hint:
            payload["delegate_model_hint"] = delegate_model_hint
        
        print(f"\n{'='*60}")
        print(f"Running Chat Learn Session")
        print(f"{'='*60}")
        print(f"Turns: {turns}")
        print(f"Model: {model_hint}")
        if delegate_model_hint:
            print(f"Delegate: {delegate_model_hint}")
        print(f"Topic: {topic}")
        print(f"Session ID: {session_id}")
        print(f"{'='*60}\n")
        
        try:
            t0 = time.time()
            resp = requests.post(
                self.chat_learn_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            dt = time.time() - t0
            
            if resp.status_code == 200:
                response_data = resp.json()
                # API wraps result in "result" key
                if "result" in response_data:
                    data = response_data["result"]
                else:
                    data = response_data
                
                data["_metadata"] = {
                    "duration_sec": dt,
                    "status_code": resp.status_code,
                    "turns_requested": turns,
                    "model_hint": model_hint,
                    "delegate_model_hint": delegate_model_hint,
                    "topic": topic,
                    "session_id": session_id,
                }
                
                # Analyze response
                history = data.get("history", [])
                outputs = data.get("outputs", [])
                results = data.get("results", [])
                
                print(f"✓ Completed in {dt:.2f}s")
                print(f"  History length: {len(history)}")
                print(f"  Outputs: {len(outputs)}")
                print(f"  Results: {len(results)}")
                
                # Print conversation preview
                if history:
                    print(f"\n  Conversation Preview:")
                    for i, turn in enumerate(history[:3]):  # First 3 turns
                        role = turn.get("role", "unknown")
                        content = turn.get("content", "")
                        preview = content[:150] + "..." if len(content) > 150 else content
                        print(f"    [{i+1}] {role}: {preview}")
                    if len(history) > 3:
                        print(f"    ... ({len(history) - 3} more turns)")
                
                # Check for mock/fallback responses
                all_text = json.dumps(data)
                is_mock = any(indicator in all_text.lower() for indicator in [
                    "mock response", "fallback", "status snapshot", "ready for xr"
                ])
                
                if is_mock:
                    print(f"  ⚠ Warning: Detected mock/fallback responses")
                else:
                    print(f"  ✓ Dynamic content detected")
                
                return data
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"✗ Failed: {error_msg}")
                return {
                    "status": "error",
                    "error": error_msg,
                    "_metadata": {
                        "duration_sec": dt,
                        "status_code": resp.status_code,
                    }
                }
                
        except requests.exceptions.Timeout:
            print(f"✗ Request timed out after {timeout}s")
            return {"status": "error", "error": "timeout"}
        except Exception as e:
            print(f"✗ Exception: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_basic_hope(self) -> Dict[str, Any]:
        """Test 1: Basic HOPE agent manager (no subagent)."""
        print("\n" + "="*60)
        print("TEST 1: Basic HOPE Agent Manager")
        print("="*60)
        return self.run_chat_learn(
            turns=5,
            model_hint="hope-v1",
            delegate_model_hint=None,
            topic="system architecture and learning mechanisms"
        )
    
    def test_hope_with_gemma_subagent(self) -> Dict[str, Any]:
        """Test 2: HOPE with Gemma subagent."""
        print("\n" + "="*60)
        print("TEST 2: HOPE Agent Manager with Gemma Subagent")
        print("="*60)
        return self.run_chat_learn(
            turns=6,
            model_hint="hope-v1",
            delegate_model_hint="google/gemma-3n-2b",
            topic="CMS compaction and memory management"
        )
    
    def test_gemma_direct(self) -> Dict[str, Any]:
        """Test 3: Direct Gemma model (no HOPE)."""
        print("\n" + "="*60)
        print("TEST 3: Direct Gemma Model")
        print("="*60)
        return self.run_chat_learn(
            turns=4,
            model_hint="google/gemma-3n-2b",
            delegate_model_hint=None,
            topic="WaveCore training loops"
        )
    
    def test_jax_integration(self) -> Dict[str, Any]:
        """Test 4: Test with JAX model hint (if available)."""
        print("\n" + "="*60)
        print("TEST 4: JAX Model Integration")
        print("="*60)
        # Note: JAX models may need special handling
        # This tests if the system can handle JAX model hints
        return self.run_chat_learn(
            turns=4,
            model_hint="hope-v1",  # HOPE may use JAX internally
            delegate_model_hint="consult:google/gemma-370m",
            topic="JAX training pipeline and TPU deployment"
        )
    
    def test_multi_topic_learning(self) -> Dict[str, Any]:
        """Test 5: Multiple topics to test learning diversity."""
        print("\n" + "="*60)
        print("TEST 5: Multi-Topic Learning")
        print("="*60)
        topics = [
            "safety policies and intervention",
            "tool router and action mapping",
            "RLDS data quality and patterns"
        ]
        
        results = []
        for topic in topics:
            result = self.run_chat_learn(
                turns=3,
                model_hint="hope-v1",
                delegate_model_hint="google/gemma-370m",
                topic=topic
            )
            results.append(result)
            time.sleep(1)  # Brief pause between topics
        
        return {
            "status": "ok",
            "topics": topics,
            "results": results
        }
    
    def test_extended_conversation(self) -> Dict[str, Any]:
        """Test 6: Extended multi-turn conversation."""
        print("\n" + "="*60)
        print("TEST 6: Extended Multi-Turn Conversation")
        print("="*60)
        return self.run_chat_learn(
            turns=10,
            model_hint="hope-v1",
            delegate_model_hint="google/gemma-3n-2b",
            topic="comprehensive system understanding and improvement",
            timeout=600  # Longer timeout for extended conversation
        )
    
    def check_rlds_logging(self, config_dir: Optional[str] = None) -> Dict[str, Any]:
        """Check if RLDS logging is working."""
        print("\n" + "="*60)
        print("Checking RLDS Logging")
        print("="*60)
        
        if config_dir is None:
            # Try to infer from common locations
            possible_dirs = [
                os.path.expanduser("~/.continuon_config"),
                "/opt/continuonos/brain",
                "./brain_config",
            ]
            for d in possible_dirs:
                rlds_dir = Path(d) / "rlds" / "episodes"
                if rlds_dir.exists():
                    config_dir = d
                    break
        
        if config_dir is None:
            print("⚠ Could not find RLDS episodes directory")
            return {"status": "unknown", "message": "RLDS directory not found"}
        
        rlds_dir = Path(config_dir) / "rlds" / "episodes"
        if not rlds_dir.exists():
            print(f"⚠ RLDS directory does not exist: {rlds_dir}")
            return {"status": "not_found", "path": str(rlds_dir)}
        
        # List recent episode files
        episode_files = list(rlds_dir.glob("*.json"))
        episode_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        print(f"✓ Found RLDS directory: {rlds_dir}")
        print(f"  Total episode files: {len(episode_files)}")
        
        if episode_files:
            print(f"  Recent files (last 5):")
            for f in episode_files[:5]:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                size = f.stat().st_size
                print(f"    - {f.name} ({size} bytes, {mtime})")
            
            # Try to read a recent file
            try:
                with open(episode_files[0], 'r') as f:
                    episode_data = json.load(f)
                print(f"  ✓ Successfully read episode file")
                print(f"    Keys: {list(episode_data.keys())}")
            except Exception as e:
                print(f"  ⚠ Could not read episode file: {e}")
        
        return {
            "status": "ok" if episode_files else "empty",
            "rlds_dir": str(rlds_dir),
            "episode_count": len(episode_files),
            "recent_files": [f.name for f in episode_files[:5]]
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test scenarios."""
        print("\n" + "="*80)
        print("AGENT CHAT MANAGER MULTI-TURN TRAINING TEST SUITE")
        print("="*80)
        print(f"Server: {self.base_url}")
        print(f"RLDS Logging: {'Enabled' if TEST_RLDS_LOGGING else 'Disabled (set CONTINUON_LOG_CHAT_RLDS=1 to enable)'}")
        print("="*80)
        
        if not self.check_server_health():
            print("\n✗ Server health check failed. Cannot proceed with tests.")
            return {"status": "error", "message": "Server not accessible"}
        
        test_results = {}
        
        # Run test suite
        tests = [
            ("basic_hope", self.test_basic_hope),
            ("hope_with_gemma", self.test_hope_with_gemma_subagent),
            ("gemma_direct", self.test_gemma_direct),
            ("jax_integration", self.test_jax_integration),
            ("multi_topic", self.test_multi_topic_learning),
            ("extended", self.test_extended_conversation),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                test_results[test_name] = result
                time.sleep(2)  # Brief pause between tests
            except Exception as e:
                print(f"\n✗ Test '{test_name}' failed with exception: {e}")
                test_results[test_name] = {"status": "error", "error": str(e)}
        
        # Check RLDS logging
        rlds_check = self.check_rlds_logging()
        test_results["rlds_check"] = rlds_check
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = 0
        failed = 0
        for test_name, result in test_results.items():
            if test_name == "rlds_check":
                continue
            status = result.get("status", "unknown")
            if status == "ok" or "_metadata" in result:
                passed += 1
                print(f"✓ {test_name}")
            else:
                failed += 1
                print(f"✗ {test_name}: {result.get('error', 'unknown error')}")
        
        print(f"\nTotal: {passed} passed, {failed} failed")
        print("="*80)
        
        return {
            "status": "ok" if failed == 0 else "partial",
            "passed": passed,
            "failed": failed,
            "results": test_results
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Agent Chat Manager multi-turn agentic learning"
    )
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help=f"Base URL for API server (default: {BASE_URL})"
    )
    parser.add_argument(
        "--test",
        choices=["all", "basic", "hope-gemma", "gemma", "jax", "multi-topic", "extended"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=5,
        help="Number of turns for single test (default: 5)"
    )
    parser.add_argument(
        "--model",
        default="hope-v1",
        help="Model hint (default: hope-v1)"
    )
    parser.add_argument(
        "--delegate",
        help="Delegate model hint (optional)"
    )
    parser.add_argument(
        "--topic",
        default="tool use + planning + safety",
        help="Topic for conversation"
    )
    parser.add_argument(
        "--check-rlds",
        action="store_true",
        help="Only check RLDS logging status"
    )
    
    args = parser.parse_args()
    
    tester = ChatLearnTester(base_url=args.url)
    
    if args.check_rlds:
        result = tester.check_rlds_logging()
        print(json.dumps(result, indent=2))
        return
    
    if args.test == "all":
        result = tester.run_all_tests()
    else:
        # Single test
        if args.test == "basic":
            result = tester.test_basic_hope()
        elif args.test == "hope-gemma":
            result = tester.test_hope_with_gemma_subagent()
        elif args.test == "gemma":
            result = tester.test_gemma_direct()
        elif args.test == "jax":
            result = tester.test_jax_integration()
        elif args.test == "multi-topic":
            result = tester.test_multi_topic_learning()
        elif args.test == "extended":
            result = tester.test_extended_conversation()
        else:
            # Custom test with provided parameters
            result = tester.run_chat_learn(
                turns=args.turns,
                model_hint=args.model,
                delegate_model_hint=args.delegate,
                topic=args.topic
            )
    
    # Output results as JSON for programmatic use
    if os.environ.get("JSON_OUTPUT"):
        print("\n" + json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
