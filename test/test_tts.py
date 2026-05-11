"""Test 4: Text-to-speech with Piper TTS on RPi 5."""
import subprocess
import time

# --- FIXED PATH ---
VOICE_MODEL = "/home/jrtillery/lighthouse/voices/en_US-amy-medium.onnx"

def speak(text):
    """Convert text to speech and play it."""
    start = time.time()

    # Piper outputs raw audio, pipe to aplay
    piper_proc = subprocess.Popen(
        ["piper", "--model", VOICE_MODEL, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    aplay_proc = subprocess.Popen(
        ["aplay", "-D", "plughw:2,0", "r", "22050", "-f", "S16_LE", "-c", "1", "-t", "raw"],
        stdin=piper_proc.stdout,
	stderr=subprocess.DEVNULL
    )

    # Send the text to Piper
    piper_proc.stdin.write(text.encode())
    piper_proc.stdin.close()
    
    # Wait for the audio to finish playing
    aplay_proc.wait()

    elapsed = time.time() - start
    return elapsed

print("=== Lighthouse TTS Test ===\n")

# Test 1: Short greeting
text1 = "Hello! I'm Lighthouse, your learning friend. What would you like to explore today?"
print(f"Speaking: \"{text1}\"")
elapsed = speak(text1)
print(f"⏱️ Time: {elapsed:.2f}s\n")

# Test 2: Activity suggestion
text2 = "Wow, butterflies are amazing! Did you know they start as tiny caterpillars? Can you find something in your house that changes into something completely different?"
print(f"Speaking: \"{text2}\"")
elapsed = speak(text2)
print(f"⏱️ Time: {elapsed:.2f}s\n")

# Test 3: Excited feedback
text3 = "That's such a great example! You're really good at finding connections between things."
print(f"Speaking: \"{text3}\"")
elapsed = speak(text3)
print(f"⏱️ Time: {elapsed:.2f}s\n")

print("✅ TTS test complete!")
