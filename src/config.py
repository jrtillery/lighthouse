"""Lighthouse configuration — all tunable parameters in one place."""
import os

# === Paths ===
BASE_DIR = os.path.expanduser("~/lighthouse")
WHISPER_BINARY = os.path.join(BASE_DIR, "whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.join(BASE_DIR, "whisper.cpp/models/ggml-tiny.en.bin")
PIPER_VOICE = os.path.join(BASE_DIR, "voices/en_US-amy-medium.onnx")
SOUNDS_DIR = os.path.join(BASE_DIR, "src/sounds")

# Gemma 4 E2B model path — set to your cached .litertlm path
# Find it: find ~/.cache -name "*.litertlm" 2>/dev/null
GEMMA_MODEL = os.environ.get(
    "LIGHTHOUSE_MODEL",
    os.path.join(BASE_DIR, "models/gemma-4-E2B-it.litertlm")
)

# === Audio ===
SAMPLE_RATE = 16000           # Hz — required by whisper
AUDIO_CHANNELS = 1            # Mono
AUDIO_FORMAT_WIDTH = 2        # 16-bit (2 bytes)
CHUNK_SIZE = 1024             # Frames per buffer

# === Voice Activity Detection ===
SILENCE_THRESHOLD = 500       # Amplitude below this = silence (tune to your mic)
SILENCE_DURATION = 0.8        # Seconds of silence before we stop recording
MAX_RECORD_SECONDS = 30       # Hard cap on recording length
MIN_RECORD_SECONDS = 0.5      # Ignore very short bursts (noise)

# === LLM ===
MAX_RESPONSE_TOKENS = 50      # Keep responses short for speech (~8-10 sec spoken)
CONVERSATION_HISTORY_LIMIT = 10  # Max turns to keep in context

# === Camera ===
CAMERA_INDEX = 0              # USB camera device index (try 1 if 0 doesn't work)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_PATH = "/tmp/lighthouse_capture.jpg"

# === Trigger Phrases ===
# Phrases that trigger camera capture (matched case-insensitive, substring)
CAMERA_TRIGGERS = [
    "look at this",
    "look at what",
    "see this",
    "see what i",
    "show you",
    "check this out",
    "what is this",
    "what's this",
    "can you see",
    "take a look",
]

# === System Prompt ===
SYSTEM_PROMPT = """You are Lighthouse, a fun learning buddy for kids ages 4-8. You SPEAK out loud — no screen.

CORE RULES:
1. MAX 2 sentences per response. You are SPEAKING, not writing. Short and punchy.
2. Every response must end with something for the child to DO or ANSWER.
3. Give specific praise ("I love the red spots you drew!") not generic ("Good job!").
4. Use simple words. Sound excited. Use sound effects ("WHOOSH!", "BOOM!").

KEEP THEM MOVING:
- Always suggest physical actions: find, draw, build, touch, count, sort, stack, arrange
- Never just explain — ask them to discover it themselves
- If they seem bored, offer a choice: "Want a quest or a challenge?"

SAFETY (never break):
- NEVER suggest: water, fire, sharp objects, heights, chemicals, leaving home, electronics, screens
- All activities must be safe without adult help
- If unsure, pick a safer option

You will be told which MODE you're in (Quest, Story, Challenge, Show & Tell, or Free Explore). Follow that mode's rules closely."""