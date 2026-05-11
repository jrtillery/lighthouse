#!/usr/bin/env python3
"""
Lighthouse Hardware Test Suite

Tests each hardware component in isolation before running the full app.
Run this any time you want to verify the Pi setup is healthy.

Usage:
    python test_hardware.py              # Full test (includes Gemma model load)
    python test_hardware.py --no-model   # Skip Gemma (fast peripheral check only)
    python test_hardware.py --component mic|speaker|camera|tts|stt|model
"""
import argparse
import os
import sys
import time
import struct
import subprocess
import wave

# ── Colour helpers ───────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✅ PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}❌ FAIL{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠️  WARN{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}ℹ{RESET}  {msg}")
def header(msg):print(f"\n{BOLD}{CYAN}{'─'*50}\n  {msg}\n{'─'*50}{RESET}")

# ── Result tracker ────────────────────────────────────────────────────────────

results = {}   # component -> "pass" | "fail" | "warn" | "skip"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SPEAKER
# ═══════════════════════════════════════════════════════════════════════════════

def test_speaker():
    header("1 / 6  SPEAKER  (aplay)")

    # Check aplay is available
    r = subprocess.run(["which", "aplay"], capture_output=True)
    if r.returncode != 0:
        fail("aplay not found — install with: sudo apt install alsa-utils")
        results["speaker"] = "fail"
        return

    # Generate a short 440 Hz test tone on the fly
    tone_path = "/tmp/lh_test_tone.wav"
    _write_test_tone(tone_path, frequency=440, duration_ms=600)

    info("Playing 440 Hz test tone for 0.6 s — you should hear a beep...")
    r = subprocess.run(["aplay", "-q", tone_path], capture_output=True, timeout=5)
    if r.returncode == 0:
        ok("Speaker played tone without error")
        results["speaker"] = "pass"
    else:
        fail(f"aplay returned error: {r.stderr.decode().strip()}")
        info("Try: aplay -l   to list audio devices")
        info("Try: amixer    to check volume levels")
        results["speaker"] = "fail"


def _write_test_tone(path, frequency=440, duration_ms=600, volume=0.4):
    """Write a simple sine-wave WAV file."""
    import math
    sample_rate = 22050
    n = int(sample_rate * duration_ms / 1000)
    fade = int(sample_rate * 0.04)
    samples = []
    for i in range(n):
        v = volume * math.sin(2 * math.pi * frequency * i / sample_rate)
        if i < fade:       v *= i / fade
        if i > n - fade:   v *= (n - i) / fade
        samples.append(int(v * 32767))
    wf = wave.open(path, "wb")
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
    wf.writeframes(struct.pack(f"{n}h", *samples))
    wf.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MICROPHONE
# ═══════════════════════════════════════════════════════════════════════════════

def test_microphone():
    header("2 / 6  MICROPHONE  (PyAudio)")

    try:
        import pyaudio
    except ImportError:
        fail("pyaudio not installed — run: pip install pyaudio --break-system-packages")
        results["mic"] = "fail"
        return

    p = pyaudio.PyAudio()

    # List all input devices
    info("Available audio input devices:")
    mic_index = None
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0:
            tag = ""
            if "usb" in d["name"].lower() or "respeaker" in d["name"].lower():
                tag = "  ← USB mic"
                mic_index = i
            print(f"    [{i}] {d['name']}{tag}")

    if mic_index is None:
        try:
            mic_index = p.get_default_input_device_info()["index"]
            warn(f"No USB mic found — using default input device (index {mic_index})")
        except Exception as e:
            fail(f"No input devices found: {e}")
            p.terminate()
            results["mic"] = "fail"
            return
    else:
        ok(f"USB microphone found at index {mic_index}")

    # Record 2 seconds and check levels
    info("Recording 2 seconds — make some noise near the mic...")
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=mic_index,
            frames_per_buffer=1024,
        )
        frames = []
        for _ in range(int(16000 / 1024 * 2)):
            frames.append(stream.read(1024, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()

        # Check RMS level
        all_samples = struct.unpack(f"{len(frames) * 1024}h", b"".join(frames))
        rms = (sum(s * s for s in all_samples) / len(all_samples)) ** 0.5
        info(f"RMS level: {rms:.1f}  (silence threshold in config: 500)")

        if rms < 50:
            fail("RMS is extremely low — mic may not be recording at all")
            results["mic"] = "fail"
        elif rms < 200:
            warn("Low RMS — mic works but is quiet. Try speaking louder or adjusting SILENCE_THRESHOLD in config.py")
            results["mic"] = "warn"
        else:
            ok(f"Microphone recording looks healthy (RMS {rms:.0f})")
            results["mic"] = "pass"

        # Save recording for the STT test
        wav_path = "/tmp/lh_test_recording.wav"
        wf = wave.open(wav_path, "wb")
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b"".join(frames))
        wf.close()
        info(f"Recording saved to {wav_path} (used by STT test)")

    except Exception as e:
        fail(f"Error recording: {e}")
        results["mic"] = "fail"
    finally:
        p.terminate()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CAMERA
# ═══════════════════════════════════════════════════════════════════════════════

def test_camera():
    header("3 / 6  CAMERA  (OpenCV)")

    try:
        import cv2
    except ImportError:
        fail("opencv-python not installed — run: pip install opencv-python --break-system-packages")
        results["camera"] = "fail"
        return

    import config
    capture_path = "/tmp/lh_test_capture.jpg"

    info(f"Trying camera index {config.CAMERA_INDEX}...")
    cam = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cam.isOpened():
        alt = 1 if config.CAMERA_INDEX == 0 else 0
        info(f"Index {config.CAMERA_INDEX} failed, trying {alt}...")
        cam = cv2.VideoCapture(alt)

    if not cam.isOpened():
        fail("No camera found on index 0 or 1")
        info("Check: ls /dev/video*   to see available video devices")
        info(f"Update CAMERA_INDEX in config.py if your camera is on a different index")
        results["camera"] = "fail"
        return

    # Flush stale frames
    for _ in range(3):
        cam.read()

    ret, frame = cam.read()
    cam.release()

    if not ret or frame is None:
        fail("Camera opened but failed to capture a frame")
        results["camera"] = "fail"
        return

    h, w = frame.shape[:2]
    cv2.imwrite(capture_path, frame)
    ok(f"Camera captured {w}×{h} image → {capture_path}")
    info("Open that file in VS Code to visually verify the camera is pointing correctly")
    results["camera"] = "pass"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TTS (Piper)
# ═══════════════════════════════════════════════════════════════════════════════

def test_tts():
    header("4 / 6  TEXT-TO-SPEECH  (Piper)")

    import config

    # Check piper binary
    r = subprocess.run(["which", "piper"], capture_output=True)
    if r.returncode != 0:
        fail("piper not found in PATH")
        info("Install guide: https://github.com/rhasspy/piper#installation")
        info("After install, make sure 'piper' is on your PATH")
        results["tts"] = "fail"
        return
    ok("piper binary found")

    # Check voice model exists
    if not os.path.exists(config.PIPER_VOICE):
        fail(f"Voice model not found: {config.PIPER_VOICE}")
        info("Download with: wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx")
        info(f"Save to: {config.PIPER_VOICE}")
        results["tts"] = "fail"
        return
    ok(f"Voice model found: {os.path.basename(config.PIPER_VOICE)}")

    # Render a test phrase
    test_text = "Hello! I am Lighthouse. Hardware test successful."
    wav_path = "/tmp/lh_test_tts.wav"
    info(f"Rendering: \"{test_text}\"")

    start = time.time()
    r = subprocess.run(
        ["piper", "--model", config.PIPER_VOICE, "--output_file", wav_path],
        input=test_text.encode("utf-8"),
        capture_output=True,
        timeout=30,
    )
    elapsed = time.time() - start

    if r.returncode != 0 or not os.path.exists(wav_path):
        fail(f"Piper failed: {r.stderr.decode().strip()}")
        results["tts"] = "fail"
        return

    ok(f"TTS rendered in {elapsed:.2f}s")

    # Play it
    info("Playing TTS output — you should hear a voice...")
    subprocess.run(["aplay", "-q", wav_path], timeout=10)
    ok("TTS audio played")
    results["tts"] = "pass"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STT (whisper.cpp)
# ═══════════════════════════════════════════════════════════════════════════════

def test_stt():
    header("5 / 6  SPEECH-TO-TEXT  (whisper.cpp)")

    import config

    # Check whisper binary
    if not os.path.exists(config.WHISPER_BINARY):
        fail(f"whisper.cpp binary not found: {config.WHISPER_BINARY}")
        info("Build with: cd ~/lighthouse/whisper.cpp && make")
        results["stt"] = "fail"
        return
    ok(f"whisper.cpp binary found: {config.WHISPER_BINARY}")

    # Check model file
    if not os.path.exists(config.WHISPER_MODEL):
        fail(f"Whisper model not found: {config.WHISPER_MODEL}")
        info("Download with: bash whisper.cpp/models/download-ggml-model.sh tiny.en")
        results["stt"] = "fail"
        return
    model_mb = os.path.getsize(config.WHISPER_MODEL) / 1024 / 1024
    ok(f"Whisper model found ({model_mb:.0f} MB): {os.path.basename(config.WHISPER_MODEL)}")

    # Try transcribing the recording from the mic test (or a generated tone as fallback)
    wav_path = "/tmp/lh_test_recording.wav"
    if not os.path.exists(wav_path):
        info("No mic recording found — generating a silent WAV for a dry run...")
        _write_test_tone(wav_path, frequency=0, duration_ms=2000, volume=0)

    info("Running whisper.cpp transcription (first run may be slow)...")
    start = time.time()
    r = subprocess.run(
        [
            config.WHISPER_BINARY,
            "-m", config.WHISPER_MODEL,
            "-f", wav_path,
            "--no-timestamps",
            "-nt",
            "--language", "en",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.time() - start

    if r.returncode != 0:
        fail(f"whisper.cpp returned error:\n{r.stderr.strip()}")
        results["stt"] = "fail"
        return

    transcript = r.stdout.strip()
    ok(f"whisper.cpp ran successfully in {elapsed:.1f}s")
    info(f"Transcript: \"{transcript if transcript else '(silence/empty)'}\"")

    if elapsed > 10:
        warn(f"Transcription took {elapsed:.1f}s — may feel slow in real use. Consider tuning threads.")
        results["stt"] = "warn"
    else:
        results["stt"] = "pass"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GEMMA MODEL (LiteRT-LM)
# ═══════════════════════════════════════════════════════════════════════════════

def test_model():
    header("6 / 6  GEMMA 4 E2B  (LiteRT-LM)")

    import config

    # Check model file
    if not os.path.exists(config.GEMMA_MODEL):
        fail(f"Model file not found: {config.GEMMA_MODEL}")
        info("Download from HuggingFace:")
        info("  huggingface-cli download google/gemma-4-E2B-it-litert-lm")
        info(f"Then set LIGHTHOUSE_MODEL env var or update GEMMA_MODEL in config.py")
        results["model"] = "fail"
        return

    model_gb = os.path.getsize(config.GEMMA_MODEL) / 1024 / 1024 / 1024
    ok(f"Model file found ({model_gb:.1f} GB): {os.path.basename(config.GEMMA_MODEL)}")

    # Try importing litert_lm
    try:
        import litert_lm
        ok("litert_lm imported successfully")
    except ImportError as e:
        fail(f"Cannot import litert_lm: {e}")
        info("Install: pip install litert-lm --break-system-packages")
        results["model"] = "fail"
        return

    # Load the engine
    info("Loading model (this takes 10–60s on first load)...")
    start = time.time()
    try:
        engine = litert_lm.Engine(config.GEMMA_MODEL)
        load_elapsed = time.time() - start
        ok(f"Model loaded in {load_elapsed:.1f}s")
    except Exception as e:
        fail(f"Engine failed to load: {e}")
        results["model"] = "fail"
        return

    # Send a test prompt
    info("Sending test prompt: \"Say hello in exactly 5 words.\"")
    try:
        conv = engine.create_conversation()
        start = time.time()
        response = conv.send_message("Say hello in exactly 5 words.")
        infer_elapsed = time.time() - start
        # API returns: {'role': 'assistant', 'content': [{'type': 'text', 'text': '...'}]}
        try:
            response_text = response['content'][0]['text'].strip()
        except (KeyError, IndexError, TypeError):
            response_text = str(response).strip()
        ok(f"Inference completed in {infer_elapsed:.1f}s")
        info(f"Response: \"{response_text}\"")

        if infer_elapsed > 8:
            warn(f"Inference is slow ({infer_elapsed:.1f}s). Check CPU governor: "
                 f"echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            results["model"] = "warn"
        else:
            results["model"] = "pass"

    except Exception as e:
        fail(f"Inference error: {e}")
        results["model"] = "fail"


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{BOLD}{'═'*50}")
    print("  HARDWARE TEST SUMMARY")
    print(f"{'═'*50}{RESET}")

    labels = {
        "speaker": "Speaker       (aplay)",
        "mic":     "Microphone    (PyAudio)",
        "camera":  "Camera        (OpenCV)",
        "tts":     "TTS           (Piper)",
        "stt":     "STT           (whisper.cpp)",
        "model":   "Gemma model   (LiteRT-LM)",
    }

    all_passed = True
    for key, label in labels.items():
        status = results.get(key, "skip")
        if status == "pass":
            icon = f"{GREEN}✅ PASS{RESET}"
        elif status == "fail":
            icon = f"{RED}❌ FAIL{RESET}"
            all_passed = False
        elif status == "warn":
            icon = f"{YELLOW}⚠️  WARN{RESET}"
        else:
            icon = f"{CYAN}── SKIP{RESET}"

        print(f"  {icon}  {label}")

    print(f"{BOLD}{'═'*50}{RESET}")

    failures = [k for k, v in results.items() if v == "fail"]
    warnings = [k for k, v in results.items() if v == "warn"]

    if failures:
        print(f"\n{RED}{BOLD}  {len(failures)} component(s) failed — fix before running lighthouse.py{RESET}")
    elif warnings:
        print(f"\n{YELLOW}  All components functional — review warnings above{RESET}")
    else:
        passed = len([v for v in results.values() if v == "pass"])
        print(f"\n{GREEN}{BOLD}  All {passed} tested components passed — ready to run!{RESET}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

COMPONENT_MAP = {
    "speaker": test_speaker,
    "mic":     test_microphone,
    "camera":  test_camera,
    "tts":     test_tts,
    "stt":     test_stt,
    "model":   test_model,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lighthouse hardware test suite")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip the Gemma model load test (faster peripheral check)")
    parser.add_argument("--component", choices=COMPONENT_MAP.keys(),
                        help="Test a single component only")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*50}")
    print("  LIGHTHOUSE HARDWARE TEST SUITE")
    print(f"{'═'*50}{RESET}")

    if args.component:
        COMPONENT_MAP[args.component]()
    else:
        test_speaker()
        test_microphone()
        test_camera()
        test_tts()
        test_stt()
        if not args.no_model:
            test_model()
        else:
            info("Skipping model test (--no-model)")
            results["model"] = "skip"

    print_summary()