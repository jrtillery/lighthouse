"""Text-to-speech module using Piper TTS with overlap pipeline.

Eliminates choppy audio by rendering to WAV files instead of raw piping,
and overlaps generation of the next sentence with playback of the current one.
"""
import subprocess
import time
import threading
import tempfile
import re
import os
import config


# Reusable temp file paths to avoid filesystem churn
_WAV_SLOTS = [
    os.path.join(tempfile.gettempdir(), "lighthouse_tts_a.wav"),
    os.path.join(tempfile.gettempdir(), "lighthouse_tts_b.wav"),
]

# Pre-rendered filler cache: phrase -> wav_path
_filler_cache = {}


def prerender_fillers(phrases):
    """Pre-render filler phrases to WAV files at startup for instant playback.

    Call once during startup with all possible filler phrases.
    Subsequent play_filler() calls skip the ~0.9s render step entirely.
    """
    print(f"[TTS] Pre-rendering {len(phrases)} filler phrases...")
    rendered = 0
    for phrase in phrases:
        key = phrase.strip().lower()
        wav_path = os.path.join(
            tempfile.gettempdir(),
            f"lighthouse_filler_{abs(hash(key)) % 100000}.wav"
        )
        if _render_to_wav(phrase, wav_path):
            _filler_cache[key] = wav_path
            rendered += 1
    print(f"[TTS] Pre-rendered {rendered}/{len(phrases)} fillers")


def play_filler(phrase):
    """Play a pre-rendered filler phrase (blocking). Falls back to speak()."""
    key = phrase.strip().lower()
    wav_path = _filler_cache.get(key)
    if wav_path and os.path.exists(wav_path):
        _play_wav(wav_path)
    else:
        speak(phrase)


def _split_sentences(text):
    """Split text into speakable chunks at sentence boundaries.

    Keeps short fragments attached to their preceding sentence
    so we don't get weird isolated words.
    """
    raw = re.split(r'(?<=[.!?]) +', text.strip())
    sentences = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # Merge very short fragments (< 5 words) onto the previous sentence
        if sentences and len(s.split()) < 5 and not s[-1] in '.!?':
            sentences[-1] += ' ' + s
        else:
            sentences.append(s)
    return sentences


def _render_to_wav(text, wav_path):
    """Render text to a WAV file using Piper. Returns True on success."""
    try:
        result = subprocess.run(
            ["piper", "--model", config.PIPER_VOICE, "--output_file", wav_path],
            input=text.encode('utf-8'),
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0 and os.path.exists(wav_path)
    except subprocess.TimeoutExpired:
        print("[TTS] Piper render timed out")
        return False


def _play_wav(wav_path):
    """Play a WAV file through the speaker. Blocks until done."""
    try:
        subprocess.run(
            ["aplay", "-q", wav_path],
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("[TTS] Playback timed out")


def speak(text):
    """Convert text to speech and play through speaker.

    Uses a two-slot overlap pipeline:
    - Slot A renders while Slot B plays (and vice versa)
    - First sentence has no overlap (must render before playing)
    - Subsequent sentences are pre-rendered during playback of the previous one

    Returns elapsed time in seconds.
    """
    if not text or not text.strip():
        return 0

    start = time.time()
    sentences = _split_sentences(text)

    if not sentences:
        return 0

    print(f"[TTS] Speaking ({len(sentences)} segment{'s' if len(sentences) != 1 else ''}): "
          f"\"{text[:60]}{'...' if len(text) > 60 else ''}\"")

    # --- Single sentence: simple render-then-play ---
    if len(sentences) == 1:
        wav = _WAV_SLOTS[0]
        if _render_to_wav(sentences[0], wav):
            _play_wav(wav)
        elapsed = time.time() - start
        print(f"[TTS] Done ({elapsed:.1f}s)")
        return elapsed

    # --- Multiple sentences: overlap pipeline ---
    # Render the first sentence upfront (can't overlap this one)
    current_slot = 0
    if not _render_to_wav(sentences[0], _WAV_SLOTS[current_slot]):
        print("[TTS] Failed to render first sentence")
        return time.time() - start

    for i in range(len(sentences)):
        play_wav = _WAV_SLOTS[current_slot]
        next_slot = 1 - current_slot  # Toggle between 0 and 1

        # If there's a next sentence, start rendering it in the background
        render_thread = None
        if i + 1 < len(sentences):
            render_thread = threading.Thread(
                target=_render_to_wav,
                args=(sentences[i + 1], _WAV_SLOTS[next_slot]),
                daemon=True,
            )
            render_thread.start()

        # Play the current sentence (blocks)
        _play_wav(play_wav)

        # Wait for the background render to finish (if it hasn't already)
        if render_thread is not None:
            render_thread.join()

        current_slot = next_slot

    elapsed = time.time() - start
    print(f"[TTS] Done ({elapsed:.1f}s)")
    return elapsed


def speak_file(text, output_path="/tmp/lighthouse_speech.wav"):
    """Render full text to a single WAV file and play it.

    Simpler than the overlap pipeline — good for short utterances
    like fillers and greetings where latency is less critical.

    Returns (output_path, elapsed_seconds).
    """
    start = time.time()
    if _render_to_wav(text, output_path):
        _play_wav(output_path)
    elapsed = time.time() - start
    return output_path, elapsed


def speak_async(text):
    """Speak in a background thread (non-blocking).

    Returns the thread object.
    """
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()
    return t