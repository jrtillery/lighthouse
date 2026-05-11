# Lighthouse

**A screen-free AI learning companion for children ages 4–8, running entirely on-device with Gemma 4.**


Lighthouse is a small hardware device that talks with children. A child walks up to it, says something, and Lighthouse responds with voice. There is no app, no account, no display, no feed. It runs entirely offline on a Raspberry Pi 5 — all speech recognition, language generation, and speech synthesis happen locally. No data leaves the room.

Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) — Future of Education + LiteRT tracks.

---

## Hardware

| Component | Model | Est. Cost |
|-----------|-------|-----------|
| Raspberry Pi 5 (8GB) | Standard | $80 |
| USB Camera | Logitech C270 or C920 | $25–70 |
| USB Microphone | ReSpeaker USB Mic Array or lavalier | $10–30 |
| Speaker | 3.5mm powered or USB speaker | $10–20 |
| MicroSD Card (64GB+) | Samsung EVO or SanDisk Extreme | $10 |
| USB-C Power Supply (5V/5A) | Official RPi 5 PSU | $12 |
| Enclosure (optional) | 3D-printed or wooden box | $0–30 |
| **Total** | | **~$150–250** |

---

## Software Stack

| Layer | Technology |
|-------|-----------|
| OS | Raspberry Pi OS (64-bit, Bookworm) |
| LLM Runtime | LiteRT-LM (Python API) |
| Model | gemma-4-E2B-it-litert-lm |
| Speech-to-Text | whisper.cpp (tiny.en) |
| Text-to-Speech | Piper TTS (Amy medium voice) |
| Camera | OpenCV via USB |
| Audio I/O | PyAudio / sounddevice |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  LIGHTHOUSE                       │
│                                                   │
│   ┌──────────┐    ┌──────────────────────────┐   │
│   │   MIC    │───▶│  whisper.cpp (STT)       │   │
│   └──────────┘    └──────────┬───────────────┘   │
│                              │ text               │
│   ┌──────────┐    ┌──────────▼───────────────┐   │
│   │  CAMERA  │───▶│  Gemma 4 E2B (LiteRT-LM) │   │
│   └──────────┘    │  • Multimodal reasoning   │   │
│                   │  • Activity generation    │   │
│                   └──────────┬───────────────┘   │
│                              │ text               │
│   ┌──────────┐    ┌──────────▼───────────────┐   │
│   │ SPEAKER  │◀───│  Piper TTS               │   │
│   └──────────┘    └──────────────────────────┘   │
│                                                   │
│   ┌──────────────────────────────────────────┐   │
│   │  Orchestrator (Python)                    │   │
│   │  • Session state & learning context       │   │
│   │  • Activity templates & safety rules      │   │
│   │  • Adaptive engagement tracking           │   │
│   └──────────────────────────────────────────┘   │
│                                                   │
│              Raspberry Pi 5 (8GB)                 │
│              100% offline — no internet           │
└─────────────────────────────────────────────────┘
```

**Pipeline:** `voice input → VAD → whisper.cpp → mode detection → Gemma 4 → Piper TTS → voice output`

**Round-trip latency:** ~5–10 seconds (STT ~1.5s + LLM ~3–5s + TTS ~1s), masked with pre-rendered filler audio.

---

## Setup

### 1. Flash Raspberry Pi OS

Flash **Raspberry Pi OS 64-bit (Bookworm)** to your MicroSD card using [Raspberry Pi Imager](https://www.raspberrypi.com/software/). Enable SSH and set your hostname/credentials in the imager settings.

### 2. Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git ffmpeg libportaudio2 libopencv-dev
```

### 3. Clone this repo

```bash
git clone https://github.com/jrtillery/lighthouse.git
cd lighthouse
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Install whisper.cpp

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
bash ./models/download-ggml-model.sh tiny.en
cd ..
```

### 5. Install Piper TTS

```bash
pip install piper-tts
python3 -c "from piper import PiperVoice; PiperVoice.load('en_US-amy-medium')"
```

### 6. Download Gemma 4 E2B via LiteRT-LM

```bash
pip install litert-lm
huggingface-cli download google/gemma-4-E2B-it-litert-lm --local-dir ./models/gemma-4-e2b
```

> You'll need a [Hugging Face account](https://huggingface.co) and to accept the Gemma model terms at [huggingface.co/google/gemma-4-E2B-it-litert-lm](https://huggingface.co/google/gemma-4-E2B-it-litert-lm).

### 7. Connect hardware

Plug in your USB microphone, USB camera, and speaker. Verify devices are detected:

```bash
arecord -l    # should list your USB microphone
aplay -l      # should list your speaker
ls /dev/video*  # should list your USB camera
```

### 8. Run Lighthouse

```bash
source venv/bin/activate
python3 main.py
```

Lighthouse will greet the child on startup and wait for voice input.

---

## Learning Modes

| Mode | Description |
|------|-------------|
| **Quest** | Three-step scavenger hunt around the home, ending with an explorer title |
| **Story Time** | Collaborative storytelling — Lighthouse starts, child decides what happens next |
| **Quick Challenge** | Timed physical activity: stacking, drawing, sorting |
| **Show & Tell** | Child describes an object to the camera; Lighthouse asks curious questions |
| **Free Explore** | Open conversation, always ending with a question to keep them moving |

---

## Safety

Lighthouse is designed to be used without adult supervision. Every response runs under a layered safety system:

- **System prompt** prohibiting fire, sharp objects, heights, chemicals, water play, or leaving home — reinforced on every mode switch
- **Mode-level guardrails** with explicit good/bad activity lists
- **Automated adversarial test suite** (`test_safety.py`) — must pass before any live use
- **3-sentence response cap** — shorter responses are harder to get wrong

Run the safety suite:

```bash
python3 test_safety.py
```

All 18 prompts must pass before deploying with children.

---

## License

Licensed under the [Apache License 2.0](LICENSE).

If this project wins the Gemma 4 Good Hackathon, it will be relicensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) per competition requirements.
