"""Test 5: USB camera capture with OpenCV on RPi 5."""
import cv2
import time

print("=== Lighthouse Camera Test ===\n")

# Open USB camera (usually index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️  Camera 0 not found, trying index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ No camera found! Check USB connection.")
    print("Run: ls /dev/video*")
    exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# Capture a single frame
start = time.time()
ret, frame = cap.read()
elapsed = time.time() - start

if ret:
    cv2.imwrite("test_capture.jpg", frame)
    print(f"📸 Image captured and saved as test_capture.jpg ({elapsed:.3f}s)")
    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"   File size: {len(open('test_capture.jpg','rb').read())/1024:.0f} KB")
else:
    print("❌ Failed to capture frame")

cap.release()
print("\n✅ Camera test complete!")
