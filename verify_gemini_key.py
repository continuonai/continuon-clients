import os
import sys
# Set key for verification process
os.environ["GEMINI_API_KEY"] = "AIzaSyDhXpcXCwhNP6Xl8HGNcAMAWi6TBdBw20A"

try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    print("Listing available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
    # Try a fallback if pro failed
    model_name = 'gemini-2.0-flash'
    print(f"\nTrying generation with {model_name}...")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Say 'Gemini is working'")
    print(f"✅ VERIFIED: {response.text}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)
