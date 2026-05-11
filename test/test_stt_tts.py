import subprocess
import time
import os

# --- PATHS ---
WHISPER_PATH = "/home/jrtillery/lighthouse/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/home/jrtillery/lighthouse/whisper.cpp/models/ggml-tiny.en.bin"
VOICE_MODEL = "/home/jrtillery/lighthouse/voices/en_US-amy-medium.onnx"
AUDIO_FILE = "/tmp/lighthouse_audio.wav"

def record_audio(duration=5):
    print(f"🎤 Recording for {duration} seconds... Speak into the Logitech Brio!")
    # Using Card 3 for the Brio 101
    subprocess.run(
        ["arecord", "-D", "plughw:3,0", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(duration), AUDIO_FILE],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print("  ✅ Recorded.")

def transcribe():
    print("🧠 Transcribing...")
    result = subprocess.run(
        [WHISPER_PATH, "-m", MODEL_PATH, "-f", AUDIO_FILE, "--no-timestamps"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def speak(text):
    print(f"🔊 Speaking: {text}")
    # Piper pipes to aplay (which uses the default JBL speaker)
    piper_proc = subprocess.Popen(
        ["piper", "--model", VOICE_MODEL, "--output-raw"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    aplay_proc = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1", "-t", "raw"],
        stdin=piper_proc.stdout, stderr=subprocess.DEVNULL
    )
    piper_proc.stdin.write(text.encode())
    piper_proc.stdin.close()
    aplay_proc.wait()

if __name__ == "__main__":
    print("=== Lighthouse Full Audio Loop Test ===\n")
    record_audio(5)
    text = transcribe()
    
    if text:
        print(f"📝 You said: \"{text}\"")
        speak(f"I heard you say: {text}")
    else:
        print("⚠️ Whisper didn't catch any words.")
    
    print("\n✅ Test complete!")
