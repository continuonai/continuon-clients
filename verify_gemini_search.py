import os
import sys
import google.generativeai as genai

# Set key (assuming it is exported or I set it here for test)
os.environ["GEMINI_API_KEY"] = "AIzaSyDhXpcXCwhNP6Xl8HGNcAMAWi6TBdBw20A"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model_name = 'gemini-pro-latest'

# Tools config using string alias feature of SDK
tools = 'google_search_retrieval'

try:
    print(f"Testing Search with {model_name}...")
    model = genai.GenerativeModel(model_name, tools=tools)
    response = model.generate_content("What is the latest stock price of Google?")
    print(f"Response: {response.text}")
    
    # Check if grounding metadata exists (implies search was used)
    if response.candidates[0].grounding_metadata.search_entry_point:
        print("✅ VERIFIED: Search Grounding used.")
    else:
        print("⚠️  Warning: No grounding metadata found (might have answered from knowledge).")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)
