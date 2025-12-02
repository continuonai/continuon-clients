#!/usr/bin/env python3
"""
Example script demonstrating Gemma 3 Nano chat integration.

This shows how to use the gemma_chat module for conversational AI
with robot context awareness.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.gemma_chat import create_gemma_chat


def main():
    print("=" * 60)
    print("Gemma 3 Nano Chat Example")
    print("=" * 60)
    print()
    
    # Create chat instance (will use mock if transformers not available)
    print("Initializing Gemma chat...")
    chat = create_gemma_chat(use_mock=False)
    
    # Get model info
    info = chat.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Device: {info['device']}")
    print(f"Loaded: {info['loaded']}")
    print(f"Has HF Token: {info['has_token']}")
    print()
    
    # Example robot context
    robot_context = "Mode: manual_control | Motion: enabled | Hardware: REAL | Joints: J0:0.0, J1:0.5, J2:-0.3"
    
    # Interactive chat loop
    print("Chat started! (Type 'quit' to exit, 'reset' to clear history)")
    print("-" * 60)
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                chat.reset_history()
                print("Chat history cleared.")
                print()
                continue
            
            # Generate response
            print("Gemma: ", end='', flush=True)
            response = chat.chat(user_input, system_context=robot_context)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


if __name__ == "__main__":
    main()
