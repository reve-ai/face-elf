"""Webcam capture module for face detection application."""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple


class Camera:
    """Handles webcam capture operations."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        """Initialize camera with specified device and resolution.

        Args:
            device_id: Camera device index (default 0 for first webcam)
            width: Desired frame width
            height: Desired frame height
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the camera device.

        Returns:
            True if camera opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.

        Returns:
            Tuple of (success, frame) where frame is BGR numpy array or None
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Release the camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def set_resolution(self, width: int, height: int) -> bool:
        """Change camera resolution.

        Args:
            width: Desired frame width
            height: Desired frame height

        Returns:
            True if resolution was changed successfully
        """
        if self.cap is None:
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width == width and actual_height == height:
            self.width = width
            self.height = height
            print(f"Camera resolution changed to: {actual_width}x{actual_height}")
            return True
        else:
            print(f"Failed to set resolution {width}x{height}, got {actual_width}x{actual_height}")
            return False

    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution.

        Returns:
            Tuple of (width, height)
        """
        if self.cap is None:
            return (self.width, self.height)
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def find_available_camera() -> int:
    """Find the first available camera device.

    Returns:
        Device ID of first working camera, or -1 if none found
    """
    for device_id in range(10):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            cap.release()
            return device_id
    return -1


# Common webcam resolutions to probe
COMMON_RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1280, 1024),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


def get_supported_resolutions(device_id: int) -> list[Tuple[int, int]]:
    """Get list of resolutions supported by a camera.

    Probes the camera with common resolutions to find which ones it supports.

    Args:
        device_id: Camera device index

    Returns:
        List of (width, height) tuples for supported resolutions
    """
    supported = []
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        return supported

    try:
        for width, height in COMMON_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Check if resolution was accepted (camera reports back the requested size)
            if actual_w == width and actual_h == height:
                if (width, height) not in supported:
                    supported.append((width, height))
    finally:
        cap.release()

    return supported


class ThreadedCamera:
    """Camera with asynchronous frame capture in a background thread.

    This decouples the UI frame rate from the camera frame rate, allowing
    smooth UI interaction even when the camera is slow.
    """

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        """Initialize threaded camera.

        Args:
            device_id: Camera device index
            width: Desired frame width
            height: Desired frame height
        """
        self.device_id = device_id
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Frame data protected by lock
        self._frame: Optional[np.ndarray] = None
        self._frame_ready = False
        self._frame_count = 0
        self._capture_fps = 0.0

        # Resolution change request
        self._pending_resolution: Optional[Tuple[int, int]] = None

    def open(self) -> bool:
        """Open camera and start capture thread.

        Returns:
            True if camera opened successfully
        """
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        self.width = actual_width
        self.height = actual_height

        print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def _capture_loop(self):
        """Background thread that continuously captures frames."""
        fps_start = time.time()
        fps_count = 0

        while self._running:
            # Check for pending resolution change
            with self._lock:
                if self._pending_resolution is not None:
                    new_w, new_h = self._pending_resolution
                    self._pending_resolution = None
                    self._apply_resolution(new_w, new_h)

            if self._cap is None:
                time.sleep(0.01)
                continue

            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_ready = True
                    self._frame_count += 1

                fps_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    with self._lock:
                        self._capture_fps = fps_count / elapsed
                    fps_count = 0
                    fps_start = time.time()
            else:
                time.sleep(0.001)

    def _apply_resolution(self, width: int, height: int):
        """Apply resolution change (called from capture thread)."""
        if self._cap is None:
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.width = actual_w
        self.height = actual_h
        print(f"Camera resolution changed to: {actual_w}x{actual_h}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest captured frame (non-blocking).

        Returns:
            Tuple of (success, frame) where frame is BGR numpy array or None
        """
        with self._lock:
            if self._frame is not None:
                # Return a copy to avoid issues with the frame being overwritten
                return True, self._frame.copy()
            return False, None

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], bool]:
        """Get the latest frame and whether it's new since last call.

        Returns:
            Tuple of (success, frame, is_new_frame)
        """
        with self._lock:
            if self._frame is not None:
                is_new = self._frame_ready
                self._frame_ready = False
                return True, self._frame.copy(), is_new
            return False, None, False

    def set_resolution(self, width: int, height: int) -> bool:
        """Request a resolution change (applied in capture thread).

        Args:
            width: Desired frame width
            height: Desired frame height

        Returns:
            True (change will be applied asynchronously)
        """
        with self._lock:
            self._pending_resolution = (width, height)
        return True

    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution."""
        return (self.width, self.height)

    def get_capture_fps(self) -> float:
        """Get the actual camera capture frame rate."""
        with self._lock:
            return self._capture_fps

    def release(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Check if camera is open and thread is running."""
        return self._cap is not None and self._cap.isOpened() and self._running

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
