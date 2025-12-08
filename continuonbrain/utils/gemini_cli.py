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
    parser.add_argument("--file", help="Path to general file (PDF, Video, etc) for multimodal query")
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
        
        input_content = [args.prompt]
        
        # Handle Image (Legacy/Direct)
        if args.image:
            import PIL.Image
            if not os.path.exists(args.image):
                print(f"Error: Image not found at {args.image}")
                sys.exit(1)
            img = PIL.Image.open(args.image)
            input_content.append(img)

        # Handle General File (PDF, etc)
        if args.file:
            path = args.file
            if not os.path.exists(path):
                print(f"Error: File not found at {path}")
                sys.exit(1)
            
            print(f"Uploading file: {path}...", file=sys.stderr)
            uploaded_file = genai.upload_file(path)
            
            # Wait for processing if needed (mostly video, but good practice)
            import time
            while uploaded_file.state.name == "PROCESSING":
                print(".", end="", flush=True, file=sys.stderr)
                time.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)
                
            if uploaded_file.state.name == "FAILED":
               print("File processing failed.", file=sys.stderr)
               sys.exit(1)
               
            input_content.append(uploaded_file)
            
        response = model.generate_content(input_content)
        print(response.text)
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
