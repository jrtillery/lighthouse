"""Test 1: Text-only prompt with Gemma 4 E2B via LiteRT-LM."""
import time
import litert_lm as litert
from huggingface_hub import hf_hub_download

#MODEL_PATH = "/home/jrtillery/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm"
# MODEL_PATH = "/home/jrtillery/lighthouse/venv/lib/python3.11/site-packages/litert_lm"
MODEL_PATH = hf_hub_download(repo_id="litert-community/gemma-4-E2B-it-litert-lm",filename="gemma-4-E2B-it.litertlm")

print("Loading model...")
start = time.time()

with litert.Engine(MODEL_PATH) as engine:
	print(f"Model loaded in {time.time() - start:.1f}s")

	with engine.create_conversation() as conversation:
		#Test 1: Simple greeting
		print("\n--- Test 1: Simple greeting ---")
		start = time.time()
		response = conversation.send_message(
			"You are a friendly learning companion for a 6-year-old child. "
			"Say hello and ask what they'd like to learn about today. "
			"Keep it to 2 sentence. "
		)
		print(f"Response ({time.time() - start:.1f}s): {response}")

		#Test 2: Activity generation
		print("\n--- Test 2: Activity generation ---")
		start = time.time()
		response = conversation.send_message(
			"The child says they want to learn about butterflies. "
			"Suggest one hands-on activity they can do right now with things "
			"found in a typical home. Keep it to 3 sentences, age-appropriate. "
		)
		print(f"Response ({time.time() - start:.1f}s): {response}")

		#Test 3: Encouraging feedback
		print("\n--- Test 3: Feedback on child's answer ---")
		start = time.time()
		response = conversation.send_message(
			"The child says: 'Butterflies have big wings and they fly to flowers!' "
			"Give encouraging, specific feedback and ask a follow-up question. "
			"Keep it to 2 sentences. "
		)
		print(f"Response ({time.time() - start:.1f}s): {response}")

print("\n Test-only test complete!")
