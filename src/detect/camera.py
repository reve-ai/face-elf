"""Webcam capture module for face detection application."""

import cv2
import numpy as np
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
