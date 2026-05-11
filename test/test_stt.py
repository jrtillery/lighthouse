"""Test 3: Speech-to-text with whisper.cpp on RPi 5."""
import subprocess
import time
import os

# --- FIXED PATHS ---
WHISPER_PATH = "/home/jrtillery/lighthouse/whisper.cpp/build/bin/whisper-cli" 
MODEL_PATH = "/home/jrtillery/lighthouse/whisper.cpp/models/ggml-tiny.en.bin"
AUDIO_FILE = "/tmp/lighthouse_audio.wav"

def record_audio(duration=5):
    """Record audio from USB microphone using arecord."""
    print(f"🎤 Recording for {duration} seconds... Speak now!")
    
    # We use plughw:2,0 based on your PyAudio output.
    # plughw automatically handles the sample rate conversion to 16000 Hz!
    try:
        subprocess.run(
            [
                "arecord", 
                "-D", "plughw:3,0", 
                "-f", "S16_LE", 
                "-r", "16000", 
                "-c", "1", 
                "-d", str(duration), 
                AUDIO_FILE
            ],
            check=True,
            # This hides the standard arecord terminal output so it looks clean
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        print("  ✅ Recording saved.")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Recording failed: {e}")

def transcribe(audio_path):
    """Transcribe audio using whisper.cpp."""
    print("🧠 Transcribing with Whisper...")
    start = time.time()
    
    if not os.path.exists(WHISPER_PATH):
        return f"Error: Whisper executable not found at {WHISPER_PATH}", 0
        
    result = subprocess.run(
        [WHISPER_PATH, "-m", MODEL_PATH, "-f", audio_path, "--no-timestamps"],
        capture_output=True, text=True
    )
    elapsed = time.time() - start

    text = result.stdout.strip()
    return text, elapsed

# Run test
if __name__ == "__main__":
    print("=== Lighthouse STT Test ===\n")
    record_audio(duration=5)
    text, elapsed = transcribe(AUDIO_FILE)
    print(f"\n📝 Transcription: \"{text}\"")
    print(f"⏱️ Time: {elapsed:.2f}s")
    print("\n✅ STT test complete!")
