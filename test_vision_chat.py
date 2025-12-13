#!/usr/bin/env python3
"""
Test Chat Agent Manager with Camera Image and Compare Models
"""
import requests
import base64
import json
import time
from pathlib import Path

def test_vision_chat():
    base_url = "http://localhost:8080"
    
    print("=" * 70)
    print("TESTING CHAT AGENT MANAGER WITH CAMERA IMAGE")
    print("=" * 70)
    print()
    
    # Wait for service
    print("Waiting for service to be ready...")
    for i in range(30):
        try:
            resp = requests.get(f"{base_url}/api/status", timeout=2)
            if resp.status_code == 200:
                print("✅ Service is ready")
                break
        except:
            if i < 29:
                time.sleep(1)
            else:
                print("❌ Service not available after 30 seconds")
                return
    else:
        print("❌ Service not available")
        return
    
    # Step 1: Get camera frame
    print("\n1. Capturing image from camera...")
    try:
        resp = requests.get(f"{base_url}/api/camera/frame", timeout=10)
        if resp.status_code == 200:
            img_path = Path("/tmp/camera_frame.jpg")
            img_path.write_bytes(resp.content)
            print(f"   ✅ Image captured: {img_path} ({len(resp.content)} bytes)")
            
            # Step 2: Test with different models
            print("\n2. Testing with different models...")
            
            img_b64 = base64.b64encode(resp.content).decode('utf-8')
            models_to_test = [
                ("google/gemma-3n-2b", "Gemma 3n 2B"),
                ("google/gemma-370m", "Gemma 370M"),
                ("google/gemma-3n-E2B-it", "Gemma 3n E2B (VLM)"),
                (None, "Default Model"),
            ]
            
            results = []
            for model_hint, model_name in models_to_test:
                print(f"\n   Testing: {model_name}...")
                
                payload = {
                    "message": "What do you see in this image? Describe it in detail and explain what the robot should understand about this scene.",
                    "image_jpeg": img_b64,
                    "image_source": "camera",
                    "vision_requested": True,
                    "session_id": f"vision_test_{model_name.replace(' ', '_')}",
                }
                
                if model_hint:
                    payload["model_hint"] = model_hint
                
                try:
                    chat_resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
                    if chat_resp.status_code == 200:
                        result = chat_resp.json()
                        response = result.get("response", "")
                        is_fallback = "Status snapshot" in response or "Robot status:" in response
                        
                        results.append({
                            "model": model_name,
                            "response": response,
                            "is_fallback": is_fallback,
                            "length": len(response),
                            "structured": result.get("structured", {})
                        })
                        
                        print(f"      ✅ Response: {len(response)} chars")
                        print(f"      Type: {'FALLBACK ⚠️' if is_fallback else 'REAL VISION ✅'}")
                        if not is_fallback:
                            print(f"      Preview: {response[:200]}...")
                    else:
                        print(f"      ❌ Failed: {chat_resp.status_code}")
                        print(f"      {chat_resp.text[:200]}")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            # Compare results
            print("\n" + "=" * 70)
            print("COMPARISON RESULTS")
            print("=" * 70)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['model']}:")
                print(f"   Length: {result['length']} chars")
                print(f"   Type: {'FALLBACK ⚠️' if result['is_fallback'] else 'REAL VISION ✅'}")
                print(f"   Preview: {result['response'][:400]}...")
                
                # Check for Hailo vision metadata
                structured = result.get('structured', {})
                vision_info = structured.get('vision', {})
                if vision_info:
                    hailo_info = vision_info.get('hailo', {})
                    if hailo_info:
                        print(f"   Hailo Vision: {'✅ Enabled' if hailo_info.get('enabled') else '❌ Disabled'}")
                        if hailo_info.get('available'):
                            print(f"   Hailo Available: ✅")
            
            # Determine best result
            real_results = [r for r in results if not r['is_fallback']]
            if real_results:
                best = max(real_results, key=lambda x: x['length'])
                print(f"\n🏆 Best Result: {best['model']} ({best['length']} chars, REAL VISION)")
                print(f"\n   Full response:")
                print(f"   {best['response']}")
            else:
                print("\n⚠️  All results were fallback responses")
                print("   This indicates models are not loading properly")
        else:
            print(f"   ❌ Camera failed: {resp.status_code}")
            print(f"   Response: {resp.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vision_chat()
