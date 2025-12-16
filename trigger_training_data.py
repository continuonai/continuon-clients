import requests
import time
import json

URL = "http://localhost:8080/api/training/chat_learn"
STATUS_URL = "http://localhost:8080/api/training/architecture_status"

def get_stats():
    try:
        r = requests.get(STATUS_URL)
        d = r.json()
        bl = d.get("background_learner", {})
        cms = d.get("loop_metrics", {}).get("cms", {}) # Wait, cms is in loop_metrics?
        # Checked status: cms is in status -> block -> loop_metrics? No.
        # RobotService.GetArchitectureStatus returns 'status' as GetRobotStatus().
        # GetRobotStatus returns {"status": { ... "loop_metrics": { "cms": ... } } }
        # So d["status_data"]["status"]["loop_metrics"]["cms"]? 
        # Actually architecture_status returns flat dict merging GetRobotStatus?
        # Let's check the previous curl output.
        # Output: {"status": "ok", ... "background_learner": {...}, ...}
        # But where is CMS?
        # "loop_metrics" is NOT in the top level of architecture_status response shown in step 662 output.
        # Wait, step 662 output:
        # { ... "background_learner": {...}, "last_autonomous_learner_action": ... }
        # I don't see "loop_metrics" in step 662 output at TOP level.
        # Accessing /api/status might be better for CMS.
        pass
    except:
        pass

def trigger_chat():
    payload = {
        "turns": 2,
        "model_hint": "mock",
        "delegate_model_hint": "consult:mock",
        "topic": "training_verification"
    }
    try:
        r = requests.post(URL, json=payload)
        print(f"Triggered: {r.status_code}")
        return r.json()
    except Exception as e:
        print(f"Error: {e}")

print("Starting training triggers...")
for i in range(5):
    print(f"Iteration {i+1}/5")
    trigger_chat()
    time.sleep(2) # Short sleep, let the server queue it
    # RunChatLearn is async but RobotService.RunChatLearn might be awaited?
    # The route calls await self.service.RunChatLearn. So it blocks the request until turns complete.
    # So we don't need sleep, it will wait.
    print("Cycle complete.")

print("Done.")
