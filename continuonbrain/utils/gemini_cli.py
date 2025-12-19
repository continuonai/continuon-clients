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
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai not installed. Run 'pip install google-genai'", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Gemini CLI")
    parser.add_argument("prompt", help="The prompt to send to Gemini")
    parser.add_argument("--image", help="Path to image file for multimodal query")
    parser.add_argument("--file", help="Path to general file (PDF, Video, etc) for multimodal query")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use (default: gemini-2.0-flash)")
    args = parser.parse_args()

    # Try to get API key from Env (support both standards)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable not set.", file=sys.stderr)
        print("Please export GOOGLE_API_KEY='your_key_here' or add it to systemd config.", file=sys.stderr)
        sys.exit(1)

    try:
        client = genai.Client(api_key=api_key)
        
        input_content = [args.prompt]
        
        # Handle Image (Using Types for proper multimodal)
        if args.image:
            import PIL.Image
            if not os.path.exists(args.image):
                print(f"Error: Image not found at {args.image}", file=sys.stderr)
                sys.exit(1)
            img = PIL.Image.open(args.image)
            input_content.append(img)

        # Handle General File (PDF, etc) - Note: Simple file upload not fully mirrored in v1 demo yet
        # For 'google-genai' SDK, we often use types.Part.from_uri or binary data
        # Keeping it simple: textual prompt + image for now, as file upload API differs significantly.
        if args.file:
             print("Warning: --file upload support is limited in this CLI version. Ignoring file argument.", file=sys.stderr)
            
        response = client.models.generate_content(
            model=args.model,
            contents=input_content
        )
        print(response.text)
    except Exception as e:
        print(f"Error querying Gemini: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
