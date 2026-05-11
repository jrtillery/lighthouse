"""Week 1 Validation: Check that all components are working."""
import subprocess
import sys
import os

print("=" * 50)
print("  LIGHTHOUSE — Week 1 Validation")
print("=" * 50)

checks = []

# Check 1: Python venv
print("\n[1/6] Python environment...")
try:
    import litert_lm
    print(f"  ✅ litert-lm-api installed")
    checks.append(True)
except ImportError:
    print(f"  ❌ litert-lm-api NOT found")
    checks.append(False)

# Check 2: Piper TTS
print("\n[2/6] Piper TTS...")
result = subprocess.run(["which", "piper"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"  ✅ Piper found at {result.stdout.strip()}")
    checks.append(True)
else:
    print(f"  ❌ Piper NOT found")
    checks.append(False)

# Check 3: Voice model
print("\n[3/6] TTS voice model...")
voice_path = os.path.expanduser("~/lighthouse/voices/en_US-amy-medium.onnx")
if os.path.exists(voice_path):
    size_mb = os.path.getsize(voice_path) / (1024 * 1024)
    print(f"  ✅ Voice model found ({size_mb:.0f} MB)")
    checks.append(True)
else:
    print(f"  ❌ Voice model NOT found at {voice_path}")
    checks.append(False)

# Check 4: whisper.cpp
print("\n[4/6] whisper.cpp...")
#whisper_path = os.path.expanduser("~/lighthouse/whisper.cpp/main")
whisper_path = os.path.expanduser("~/lighthouse/whisper.cpp/build/bin/whisper-cli")
if os.path.exists(whisper_path):
    print(f"  ✅ whisper.cpp binary found")
    checks.append(True)
else:
    print(f"  ❌ whisper.cpp binary NOT found at {whisper_path}")
    checks.append(False)

# Check 5: Whisper model
print("\n[5/6] Whisper model...")
whisper_model = os.path.expanduser("~/lighthouse/whisper.cpp/models/ggml-tiny.en.bin")
if os.path.exists(whisper_model):
    size_mb = os.path.getsize(whisper_model) / (1024 * 1024)
    print(f"  ✅ Whisper tiny.en model found ({size_mb:.0f} MB)")
    checks.append(True)
else:
    print(f"  ❌ Whisper model NOT found")
    checks.append(False)

# Check 6: USB camera
print("\n[6/6] USB camera...")
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret:
        print(f"  ✅ Camera working ({frame.shape[1]}x{frame.shape[0]})")
        checks.append(True)
    else:
        print(f"  ❌ Camera opened but can't capture")
        checks.append(False)
else:
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"  ✅ Camera working on index 1 ({frame.shape[1]}x{frame.shape[0]})")
            checks.append(True)
        else:
            print(f"  ❌ Camera opened but can't capture")
            checks.append(False)
    else:
        print(f"  ❌ No camera found")
        checks.append(False)

# Summary
print("\n" + "=" * 50)
passed = sum(checks)
total = len(checks)
if passed == total:
    print(f"  🎉 ALL {total} CHECKS PASSED — Milestone 1 complete!")
    print("  You're ready for Week 2: Integration")
else:
    print(f"  ⚠️  {passed}/{total} checks passed")
    print(f"  Fix the failing components before moving on.")
print("=" * 50)
