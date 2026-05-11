import time
import litert_lm as litert
from huggingface_hub import hf_hub_download

# Ensure this points to a vision-capable model!
MODEL_PATH = "/path/to/your/multimodal_model.litertlm" 
MODEL_PATH = hf_hub_download(repo_id="litert-community/gemma-4-E2B-it-litert-lm",filename="gemma-4-E2B-it.litertlm")


print("Loading engine with Vision Backend...")
start = time.time()

try:
    # CRITICAL FIX: You must explicitly enable the vision backend
    with litert.Engine(
        MODEL_PATH, 
        vision_backend=litert.Backend.CPU 
    ) as engine:
        
        print(f"Engine loaded in {time.time() - start:.1f}s")
        
        with engine.create_conversation() as conversation:
            print("\n--- Testing Vision API (Pattern A) ---")
            msg_start = time.time()
            
            # Pattern A is the correct Google API pattern
            user_message = {
                "role": "user",
                "content": [
                    # Ensure "test_image.jpg" is in the same directory as this script
                    {"type": "image", "path": "test_image.jpg"},
                    {"type": "text", "text": (
                        "You are a friendly learning companion for a 6-year-old. "
                        "Describe what you see in this image in simple, excited language. "
                        "Then ask the child a question about it."
                    )}
                ]
            }
            
            response = conversation.send_message(user_message)
            
            # The API returns a dictionary, so we parse out the actual text 
            # to make the terminal output clean.
            output_text = response["content"][0]["text"]
            
            print(f"Response ({time.time() - msg_start:.1f}s):\n{output_text}")
            print("\n✅ Vision test complete!")

except Exception as e:
    print(f"\nVision test failed: {e}")
