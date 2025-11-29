"""Face capture module for Phase 2 functionality."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from enum import Enum
import time

from .detector import Face


class AppState(Enum):
    """Application state modes."""
    DETECTION = "detection"
    CAPTURE = "capture"


@dataclass
class CapturedFace:
    """A captured face image."""
    image: np.ndarray        # Full 384px normalized image
    thumbnail: np.ndarray    # 128x128 display thumbnail
    timestamp: float = field(default_factory=time.time)


class FaceCaptureSession:
    """Manages a face capture session."""

    NORMALIZED_SIZE = 384    # Longest axis size for saved images
    THUMBNAIL_SIZE = 128     # Display thumbnail size

    def __init__(self):
        self.captures: List[CapturedFace] = []
        self.state = AppState.DETECTION

    def enter_capture_mode(self) -> None:
        """Enter capture mode."""
        self.state = AppState.CAPTURE
        self.captures.clear()

    def exit_capture_mode(self, discard: bool = True) -> None:
        """Exit capture mode.

        Args:
            discard: If True, clear all captures
        """
        self.state = AppState.DETECTION
        if discard:
            self.captures.clear()

    def is_capture_mode(self) -> bool:
        """Check if in capture mode."""
        return self.state == AppState.CAPTURE

    def capture_face(self, frame: np.ndarray, face: Face) -> Optional[CapturedFace]:
        """Capture a face from the frame.

        Args:
            frame: Full camera frame (BGR)
            face: Detected face to capture

        Returns:
            CapturedFace if successful, None otherwise
        """
        cropped = crop_face(frame, face)
        if cropped is None:
            return None

        normalized = scale_to_normalized_size(cropped, self.NORMALIZED_SIZE)
        thumbnail = cv2.resize(normalized, (self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE))

        captured = CapturedFace(
            image=normalized,
            thumbnail=thumbnail
        )
        self.captures.append(captured)
        return captured

    def save_captures(self, name: str, base_dir: str = "known-faces") -> bool:
        """Save all captured faces to directory.

        Args:
            name: Name for this person
            base_dir: Base directory for known faces

        Returns:
            True if successful
        """
        if not self.captures:
            return False

        # Create directory
        save_dir = Path(base_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Find next available index
        existing = list(save_dir.glob("*.jpg"))
        start_idx = len(existing) + 1

        # Save each capture
        for i, capture in enumerate(self.captures):
            filename = save_dir / f"{start_idx + i:03d}.jpg"
            cv2.imwrite(str(filename), capture.image)

        print(f"Saved {len(self.captures)} images to {save_dir}")
        return True

    def get_capture_count(self) -> int:
        """Get number of captured faces."""
        return len(self.captures)


def find_central_face(faces: List[Face], frame_width: int, frame_height: int) -> Optional[Face]:
    """Find the face closest to the center of the frame.

    Args:
        faces: List of detected faces
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Face closest to center, or None if no faces
    """
    if not faces:
        return None

    center_x = frame_width // 2
    center_y = frame_height // 2

    def distance_to_center(face: Face) -> float:
        fx, fy = face.center
        return ((fx - center_x) ** 2 + (fy - center_y) ** 2) ** 0.5

    return min(faces, key=distance_to_center)


def crop_face(frame: np.ndarray, face: Face) -> Optional[np.ndarray]:
    """Crop face region from frame.

    Args:
        frame: Full camera frame (BGR)
        face: Face to crop

    Returns:
        Cropped face image, or None if invalid
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox

    # Clamp to frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Check valid region
    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def scale_to_normalized_size(image: np.ndarray, target_size: int = 384) -> np.ndarray:
    """Scale image so longest axis equals target_size.

    Args:
        image: Input image
        target_size: Target size for longest axis

    Returns:
        Scaled image
    """
    h, w = image.shape[:2]

    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
