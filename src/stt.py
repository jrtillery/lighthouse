"""Speech-to-text module using whisper.cpp."""
import subprocess
import time
import wave
import contextlib
import threading
import sys
import pyaudio
import config
import os
import numpy as np

# Hides messy ALSA noise
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


@contextlib.contextmanager
def _suppress_stderr():
    """Suppress C-library stderr noise (ALSA/Jack errors) at the OS level."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# --- Persistent PyAudio state ---
# Kept alive between turns to avoid the 300-800ms ALSA device probe on every call.
_pyaudio_instance = None
_persistent_stream = None
_stream_mic_index = None
_stream_channels = None
_stream_lock = threading.Lock()


def _close_stream_unsafe():
    """Close the persistent stream and terminate PyAudio. Caller must hold _stream_lock."""
    global _pyaudio_instance, _persistent_stream, _stream_mic_index, _stream_channels
    if _persistent_stream is not None:
        try:
            _persistent_stream.stop_stream()
            _persistent_stream.close()
        except Exception:
            pass
        _persistent_stream = None
    if _pyaudio_instance is not None:
        try:
            _pyaudio_instance.terminate()
        except Exception:
            pass
        _pyaudio_instance = None
    _stream_mic_index = None
    _stream_channels = None


def _get_stream(mic_index):
    """Return (PyAudio, stream, actual_channels), reusing the persistent stream if valid."""
    global _pyaudio_instance, _persistent_stream, _stream_mic_index, _stream_channels

    with _stream_lock:
        # Reuse if already open on the same device
        if (
            _persistent_stream is not None
            and _persistent_stream.is_active()
            and _stream_mic_index == mic_index
        ):
            return _pyaudio_instance, _persistent_stream, _stream_channels

        # Stale or missing — close and reopen
        _close_stream_unsafe()

        with _suppress_stderr():
            _pyaudio_instance = pyaudio.PyAudio()

        try:
            device_info = _pyaudio_instance.get_device_info_by_index(mic_index)
            max_channels = int(device_info.get('maxInputChannels', 1))
        except Exception:
            max_channels = 1

        actual_channels = min(2, max_channels)

        with _suppress_stderr():
            try:
                stream = _pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=actual_channels,
                    rate=config.SAMPLE_RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=config.CHUNK_SIZE,
                )
            except Exception as e:
                print(f"[STT] {actual_channels}-ch open failed ({e}), falling back to mono")
                actual_channels = 1
                stream = _pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=config.SAMPLE_RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=config.CHUNK_SIZE,
                )

        _persistent_stream = stream
        _stream_mic_index = mic_index
        _stream_channels = actual_channels
        print(f"[STT] Opened persistent {actual_channels}-ch stream on device {mic_index}")
        return _pyaudio_instance, _persistent_stream, _stream_channels


def find_usb_mic():
    """Find the USB microphone device index."""
    with _suppress_stderr():
        p = pyaudio.PyAudio()
    mic_index = None

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()
        # Look for Brio or USB
        if info['maxInputChannels'] > 0 and ('usb' in name or 'brio' in name or 'respeaker' in name):
            mic_index = i
            print(f"[STT] Found USB mic: {info['name']} (index {i})")
            break

    if mic_index is None:
        try:
            mic_index = p.get_default_input_device_info()['index']
            print(f"[STT] No USB mic found, using default (index {mic_index})")
        except Exception as e:
            print(f"[STT] Could not get default input device ({e}), using index 0.")
            mic_index = 0
    
    p.terminate()
    return mic_index

def record_until_silence(mic_index=None):
    """Record audio, stopping when silence is detected."""
    if mic_index is None:
        mic_index = find_usb_mic()

    try:
        p, stream, actual_channels = _get_stream(mic_index)
    except Exception as e:
        print(f"[STT] Failed to get audio stream: {e}")
        return None

    frames = []
    silent_chunks = 0
    has_speech = False
    chunks_per_second = config.SAMPLE_RATE / config.CHUNK_SIZE
    silence_chunks_needed = int(config.SILENCE_DURATION * chunks_per_second)
    max_chunks = int(config.MAX_RECORD_SECONDS * chunks_per_second)
    min_chunks = int(config.MIN_RECORD_SECONDS * chunks_per_second)

    print("[STT] Listening...")

    # Drain audio that accumulated in the ALSA buffer while the brain was thinking.
    # Without this, the first recording after a long LLM inference may pick up
    # stale audio from the previous turn.
    drain_chunks = int(0.3 * chunks_per_second)
    for _ in range(drain_chunks):
        stream.read(config.CHUNK_SIZE, exception_on_overflow=False)

    try:
        for i in range(max_chunks):
            raw_data = stream.read(config.CHUNK_SIZE, exception_on_overflow=False)

            # If we are in stereo, convert to mono for RMS calculation and Whisper
            if actual_channels == 2:
                # Convert buffer to numpy array, take every 2nd sample (left channel), back to bytes
                audio_np = np.frombuffer(raw_data, dtype=np.int16)[0::2]
                mono_data = audio_np.tobytes()
            else:
                mono_data = raw_data

            frames.append(mono_data)

            # RMS calculation (always on mono data)
            samples_np = np.frombuffer(mono_data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples_np.astype(np.float32) ** 2)))

            if rms < config.SILENCE_THRESHOLD:
                silent_chunks += 1
                if has_speech and silent_chunks >= silence_chunks_needed:
                    print("[STT] Silence detected, stopping recording.")
                    break
            else:
                silent_chunks = 0
                has_speech = True

    except OSError as e:
        # Stream died (USB disconnect, buffer overrun) — force reopen next turn
        print(f"[STT] Stream error: {e}. Will reopen next turn.")
        with _stream_lock:
            _close_stream_unsafe()
        if not has_speech:
            return None
        # fall through and save whatever was recorded before the error

    # Stream stays open — no stop/close/terminate here

    if not has_speech or len(frames) < min_chunks:
        print("[STT] No speech detected.")
        return None

    # Save to WAV (Always save as MONO 1 channel for Whisper)
    audio_path = "/tmp/lighthouse_audio.wav"
    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(1) 
    wf.setsampwidth(config.AUDIO_FORMAT_WIDTH)
    wf.setframerate(config.SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    duration = len(frames) * config.CHUNK_SIZE / config.SAMPLE_RATE
    print(f"[STT] Recorded {duration:.1f}s of audio.")
    return audio_path

def transcribe(audio_path):
    """Transcribe audio file using whisper.cpp."""
    start = time.time()
    # SPEED OPTIMIZATION: Added "-t 4" to use all 4 CPU cores of the Pi 5
    result = subprocess.run(
        [
            config.WHISPER_BINARY,
            "-m", config.WHISPER_MODEL,
            "-f", audio_path,
            "-t", "4", 
            "--no-timestamps",
            "--language", "en",
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    text = result.stdout.strip()

    # Clean artifacts like [BLANK_AUDIO] or (whistling)
    if text.startswith("[") and "]" in text:
        text = text[text.index("]") + 1:].strip()
    
    print(f"[STT] Transcribed ({elapsed:.1f}s): \"{text}\"")
    return text, elapsed

def listen_and_transcribe(mic_index=None):
    audio_path = record_until_silence(mic_index)
    if audio_path is None:
        return None, 0
    return transcribe(audio_path)