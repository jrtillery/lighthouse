"""Audio cue system — generates and plays feedback sounds."""
import subprocess
import struct
import wave
import math
import os
import config


SOUNDS_CACHE = {}


def _generate_tone(filename, frequency, duration_ms, volume=0.3, fade_ms=50):
    """Generate a simple tone WAV file."""
    path = os.path.join(config.SOUNDS_DIR, filename)
    if os.path.exists(path):
        return path

    sample_rate = 22050
    num_samples = int(sample_rate * duration_ms / 1000)
    fade_samples = int(sample_rate * fade_ms / 1000)

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = volume * math.sin(2 * math.pi * frequency * t)

        # Fade in
        if i < fade_samples:
            value *= i / fade_samples
        # Fade out
        if i > num_samples - fade_samples:
            value *= (num_samples - i) / fade_samples

        samples.append(int(value * 32767))

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(struct.pack(f'{len(samples)}h', *samples))
    wf.close()

    return path


def _generate_chime(filename, frequencies, duration_ms=150, gap_ms=80):
    """Generate a multi-note chime."""
    path = os.path.join(config.SOUNDS_DIR, filename)
    if os.path.exists(path):
        return path

    sample_rate = 22050
    all_samples = []

    for freq in frequencies:
        num_samples = int(sample_rate * duration_ms / 1000)
        fade_samples = int(sample_rate * 30 / 1000)

        for i in range(num_samples):
            t = i / sample_rate
            value = 0.25 * math.sin(2 * math.pi * freq * t)
            if i < fade_samples:
                value *= i / fade_samples
            if i > num_samples - fade_samples:
                value *= (num_samples - i) / fade_samples
            all_samples.append(int(value * 32767))

        # Gap between notes
        gap_samples = int(sample_rate * gap_ms / 1000)
        all_samples.extend([0] * gap_samples)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(struct.pack(f'{len(all_samples)}h', *all_samples))
    wf.close()

    return path


def init_sounds():
    """Pre-generate all sound effects."""
    os.makedirs(config.SOUNDS_DIR, exist_ok=True)

    SOUNDS_CACHE['listening'] = _generate_tone("listening.wav", 880, 150, volume=0.2)
    SOUNDS_CACHE['thinking'] = _generate_tone("thinking.wav", 440, 300, volume=0.15)
    SOUNDS_CACHE['startup'] = _generate_chime("startup.wav", [523, 659, 784], 200, 100)
    SOUNDS_CACHE['shutdown'] = _generate_chime("shutdown.wav", [784, 659, 523], 200, 100)
    SOUNDS_CACHE['error'] = _generate_tone("error.wav", 220, 400, volume=0.2)
    SOUNDS_CACHE['camera'] = _generate_tone("camera.wav", 1047, 100, volume=0.2)

    print(f"[SFX] {len(SOUNDS_CACHE)} sounds ready")


def play(sound_name):
    """Play a named sound effect (non-blocking)."""
    path = SOUNDS_CACHE.get(sound_name)
    if path and os.path.exists(path):
        subprocess.Popen(
            ["aplay", "-q", path],
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"[SFX] Sound not found: {sound_name}")