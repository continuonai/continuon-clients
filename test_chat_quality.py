import requests
import json
import time

BASE_URL = "http://localhost:8082/api/training/chat_learn"
ITERATIONS = 10

def run_test():
    print(f"Starting {ITERATIONS} autonomous chat iterations...")
    
    headers = {"Content-Type": "application/json"}
    # forcing consult:mock to ensure it runs without OOM, but we want to see the CONTENT.
    # If the user wants to see if they "converse", we'll see the mock echoes.
    payload = {
        "turns": 3, 
        "model_hint": "hope-v1",  # Requesting primary agent model
        "delegate_model_hint": "google/gemma-370m", 
        "topic": "System Self-Improvement"
    }

    try:
        for i in range(ITERATIONS):
            print(f"\n--- Iteration {i+1}/{ITERATIONS} ---")
            t0 = time.time()
            resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=120)
            dt = time.time() - t0
            
            if resp.status_code == 200:
                data = resp.json()
                history = data.get("history", [])
                print(f"Completed in {dt:.1f}s. History length: {len(history)} turns.")
                
                # Print the conversation
                if history:
                    for turn in history:
                        role = turn.get("role", "Unknown")
                        content = turn.get("content", "")
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"[{role}]: {preview}")
                else:
                    print("History empty. Checking 'results'...")
                    results = data.get("results", [])
                    for i, res in enumerate(results):
                        # res might be a string or dict
                        content = res.get("response", str(res)) if isinstance(res, dict) else str(res)
                        print(f"[Turn {i} (Result)]: {content[:200]}...")
                    
                # Check for "canned" indicators logic
                all_text = json.dumps(history) + json.dumps(data.get("results", []))
                is_canned = "Mock response" in all_text
                if is_canned:
                    print(">> DETECTED CANNED/MOCK RESPONSES")
                else:
                    print(">> DYNAMIC CONTENT DETECTED")

            else:
                print(f"Failed: {resp.status_code} - {resp.text}")
            
            # Short sleep between
            time.sleep(2)

    except Exception as e:
        print(f"Test aborted: {e}")

if __name__ == "__main__":
    run_test()
