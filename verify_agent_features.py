import requests
import json
import time

BASE_URL = "http://localhost:8080"

def test_endpoint(name, method, url, data=None):
    print(f"\n--- Testing {name} ---")
    try:
        if method == 'GET':
            response = requests.get(f"{BASE_URL}{url}")
        elif method == 'POST':
            response = requests.post(f"{BASE_URL}{url}", json=data)
        
        print(f"Status: {response.status_code}")
        try:
            print("Response:", json.dumps(response.json(), indent=2))
            return response.json()
        except:
            print("Response Text:", response.text)
            return None
    except Exception as e:
        print(f"Failed: {e}")
        return None

# 1. Test Search
test_endpoint("Search Conversations", "POST", "/api/agent/search", {
    "query": "capability"
})

# 2. Test Consolidation
test_endpoint("Consolidate Memories", "POST", "/api/agent/consolidate")

# 3. Test Decay
test_endpoint("Confidence Decay", "POST", "/api/agent/decay")

# 4. Test Chat with Session
print("\n--- Testing Chat with Session ---")
session_id = f"test_session_{int(time.time())}"
print(f"Session ID: {session_id}")

# Turn 1
data1 = {
    "message": "My name is Tester.",
    "session_id": session_id
}
test_endpoint("Chat Turn 1", "POST", "/api/chat", data1)

# Turn 2
data2 = {
    "message": "What is my name?",
    "session_id": session_id
}
test_endpoint("Chat Turn 2", "POST", "/api/chat", data2)

# 5. Learning Stats
test_endpoint("Learning Stats", "GET", "/api/agent/learning_stats")
