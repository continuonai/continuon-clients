#!/usr/bin/env python3
"""
Gemini CLI Wrapper for ContinuonBrain.
Allows the agent (or user) to query Gemini API from the terminal.

Usage:
    python3 continuonbrain/utils/gemini_cli.py "Your prompt here"
"""
import os
import sys
import argparse
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run 'pip install google-generativeai'")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Gemini CLI")
    parser.add_argument("prompt", help="The prompt to send to Gemini")
    parser.add_argument("--image", help="Path to image file for multimodal query")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use (default: gemini-2.0-flash)")
    args = parser.parse_args()

    # Try to get API key from Env or Config
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please export GEMINI_API_KEY='your_key_here' or add it to systemd config.")
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.model)
        
        content = [args.prompt]
        if args.image:
            import PIL.Image
            img_path = args.image
            if not os.path.exists(img_path):
                print(f"Error: Image not found at {img_path}")
                sys.exit(1)
            img = PIL.Image.open(img_path)
            content.append(img)
            
        response = model.generate_content(content)
        print(response.text)
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
