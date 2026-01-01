import requests
import time
import json
import sys

API_BASE = "http://localhost:8081"

def check_status():
    try:
        resp = requests.get(f"{API_BASE}/api/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error checking status: {e}")
    return None

def test_chat_knowledge(prompt, context=""):
    print(f"\n🧠 Testing Knowledge: '{prompt}'")
    payload = {
        "message": prompt,
        "context": context,
        "mode": "inference" 
    }
    try:
        start = time.time()
        resp = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=90)
        end = time.time()
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ Response ({end-start:.2f}s): {data.get('response', 'No response')}")
            return data
        else:
            print(f"  ❌ Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
    return None

def test_curriculum_recall(lesson_id):
    print(f"\n📚 Testing Curriculum Recall: '{lesson_id}'")
    try:
        resp = requests.post(f"{API_BASE}/api/curriculum/run", json={"lesson_id": lesson_id}, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            score = 0
            passed = 0
            total = len(data.get('results', []))
            for r in data.get('results', []):
                if r.get('passed'): passed += 1
            
            print(f"  ✅ Lesson Result: {passed}/{total} challenges passed.")
            return data
        else:
            print(f"  ❌ Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
    return None

def main():
    print("========================================")
    print("HOPE Inference Verification")
    print("========================================")
    
    # 1. Wait for API
    print("Waiting for API...")
    for i in range(10):
        status = check_status()
        if status:
            print("  ✅ API is Online")
            # Verify inference mode
            print(f"  System Mode: {status.get('mode', 'unknown')}")
            print(f"  Surprise Metric: {status.get('surprise', 'unknown')}")
            break
        time.sleep(2)
    else:
        print("  ❌ API failed to start.")
        sys.exit(1)

    # 2. Test Self-Awareness (VQ-VAE Knowledge)
    test_chat_knowledge("What is the compression ratio of your VQ-VAE vision encoder?")
    test_curriculum_recall("self-awareness")

    # 3. Test Physics Grounding
    test_chat_knowledge("If I drop a ball from 10m, how long to hit the ground? (Use g=9.8)")
    test_curriculum_recall("physics-lab")

    # 4. Test General "Surprise" / Curiosity State
    # Note: Surprise should be low in a static inference test unless we feed weird inputs.
    status = check_status()
    print(f"\nCurrent Surprise Level: {status.get('surprise')}")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    main()
