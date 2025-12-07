
import requests
import json
import time
import sys

def test_api():
    base_url = "http://localhost:8083"
    
    # Check info first
    print(f"Checking {base_url}/api/agent/info...")
    try:
        resp = requests.get(f"{base_url}/api/agent/info", timeout=5)
        print(f"Agent Info: {json.dumps(resp.json(), indent=2)}")
    except Exception as e:
        print(f"Info check failed: {e}")

    url = f"{base_url}/api/chat"
    payload = {"message": "Hello from API test! Are you Gemma 2B?"}
    headers = {"Content-Type": "application/json"}
    
    max_retries = 5
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}...")
            # Increase timeout significantly for CPU inference/loading
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            if response.status_code == 200:
                print(f"✅ Success! Status Code: {response.status_code}")
                # Pretty print response
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
                return
            else:
                print(f"❌ Failed. Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                return
        except requests.exceptions.ConnectionError:
            print("Connection refused, server might be starting up...")
            time.sleep(2)
        except Exception as e:
            print(f"API request failed: {e}")
            time.sleep(2)
    
    print("❌ Failed to connect after multiple attempts")

if __name__ == "__main__":
    test_api()
