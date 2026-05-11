"""Camera module for capturing images from USB camera."""
import cv2
import time
import config

# Lazy-loaded camera instance
_camera = None

def get_camera():
    """Get or create the camera instance."""
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(config.CAMERA_INDEX)
        if not _camera.isOpened():
            alt = 1 if config.CAMERA_INDEX == 0 else 0
            print(f"[CAM] Camera {config.CAMERA_INDEX} failed, trying {alt}...")
            _camera = cv2.VideoCapture(alt)

        if _camera.isOpened():
            # Set hardware capture resolution
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            # Speed up capture by disabling autofocus/autoexposure wait if possible
            _camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[CAM] Camera ready ({config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT})")
        else:
            print("[CAM] WARNING: No camera available!")
            _camera = None

    return _camera

def capture():
    """Capture a single frame, resize, and save to disk."""
    cam = get_camera()
    if cam is None:
        return None

    # Flush the buffer to get a fresh frame
    for _ in range(2):
        cam.read()

    start = time.time()
    ret, frame = cam.read()
    
    if ret:
        # Resize for storage — vision is a text-only fallback on this hardware,
        # so no colour-space conversion is needed before imwrite (expects BGR).
        small_frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
        cv2.imwrite(config.CAPTURE_PATH, small_frame)
        
        elapsed = time.time() - start
        print(f"[CAM] Captured and Resized ({elapsed:.3f}s) -> {config.CAPTURE_PATH}")
        return config.CAPTURE_PATH
    else:
        print("[CAM] Failed to capture frame")
        return None

def release():
    """Release the camera."""
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None
        print("[CAM] Camera released")