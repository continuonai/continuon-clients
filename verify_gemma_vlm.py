import os
import sys
import torch
try:
    from transformers import AutoProcessor, AutoModelForMultimodalLM
except ImportError:
    print("⚠️  AutoModelForMultimodalLM not found.")
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
        print("✅ Using AutoModelForImageTextToText")
    except ImportError:
        print("⚠️  AutoModelForImageTextToText not found, trying AutoModelForVision2Seq...")
        from transformers import AutoModelForVision2Seq as AutoModelForMultimodalLM
from PIL import Image
import requests
from io import BytesIO

def verify_vlm():
    print("🚀 Starting Gemma 3N VLM Verification...")
    
    model_id = "google/gemma-3n-E2B-it"
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("⚠️  Warning: HUGGINGFACE_TOKEN not set")

    print(f"📦 Loading Processor: {model_id}")
    try:
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")
        return

    print(f"🧠 Loading Model: {model_id}")
    try:
        # Try bfloat16 for memory efficiency
        dtype = torch.bfloat16
        print(f"   - Using dtype: {dtype}")
        
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, 
            token=hf_token,
            trust_remote_code=True,
            device_map="cpu", 
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Try fallback?
        return

    # Create dummy image (Red Square)
    img = Image.new('RGB', (100, 100), color='red')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img}, # Note: key might be 'image' or 'url' depending on processor logic. 
                # The user snippet used "url" with a string, but the processor might handle PIL images with "image" key?
                # User snippet: {"type": "image", "url": "..."}
                # Let's try passing PIL image directly if supported, or save to bytes/path.
                # Actually, standard chat templates usually take "image" key with PIL object if interacting via processor.apply_chat_template?
                # Wait, the user snippet used processor.apply_chat_template manually.
                # Let's look at user snippet again:
                # inputs = processor.apply_chat_template(messages, ...)
                # If we pass PIL, we probably need to handle it. 
                # Let's stick to the user's snippet pattern but with local image handling.
                # Most processors handle PIL in the 'content' list if key is 'image'.
                # Let's try the user snippet's exact structure but with a data URL or just check if processor handles PIL.
                {"type": "image", "image": img},
                {"type": "text", "text": "What color is this image?"}
            ]
        },
    ]

    print("🖼️  Processing inputs...")
    try:
        # Pre-process image? 
        # The user snippet does: 
        # inputs = processor.apply_chat_template(messages, ...)
        # We need to make sure processor handles the image object in the message.
        # Typically apply_chat_template processes text. 
        # If it's a VLM processor, it might handle images in the list.
        # Let's try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
         
        # We need to process image and text inputs.
        # For Gemini/Gemma VLMs, the processor usually expects (text, images) or handles it via `processor(text=..., images=...)`.
        # The user snippet:
        # inputs = processor.apply_chat_template(..., return_tensors="pt")
        # This implies apply_chat_template handles everything including images?
        # Let's assume the user snippet is correct for this specific model/library version.
        
        # HOWEVER, the user snippet provided keys "url" and "text". It didn't provide PIL objects.
        # If I provide a PIL object, I might need to adapt.
        # Let's try the standard VLM pattern:
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
        
        # User snippet used:
        # inputs = processor.apply_chat_template(messages, ..., padding=True, return_tensors="pt")
        # If that fails, I will fallback to separate calls.
        
    except Exception as e:
        # Reset and try separate:
        print(f"⚠️  Snippet method failed: {e}")
        try:
             # Manual prompt construction if template fails
             prompt = "User: <image>\nWhat color is this?\nAssistant:"
             inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        except Exception as e2:
             print(f"❌ Input processing failed: {e2}")
             return

    print("🔮 Generating...")
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        print("📝 Decoding...")
        response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"🤖 Response: {response}")
        
        if "red" in response.lower():
            print("✅ VERIFICATION PASSED")
        else:
            print("⚠️  Verification inconclusive (response didn't contain 'red')")
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    verify_vlm()
