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


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Normalize lighting using global linear stretching on the L channel.

    Uses percentile-based contrast stretching for a natural look while
    reducing the impact of lighting variations on face recognition.
    Applies adaptive denoising when significant brightness boosting is needed.

    Args:
        image: BGR input image

    Returns:
        Lighting-normalized BGR image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split into channels
    l, a, b = cv2.split(lab)

    # Extract center 50% of image for p95 calculation
    # This avoids bright edges (windows, lights) affecting the gain
    img_h, img_w = l.shape
    y1, y2 = img_h // 4, img_h * 3 // 4
    x1, x2 = img_w // 4, img_w * 3 // 4
    l_center = l[y1:y2, x1:x2]

    # p5 from entire image (catch dark areas anywhere)
    # p95 from center only (ignore bright edges like windows)
    p5 = np.percentile(l, 5)
    p95 = np.percentile(l_center, 95)

    # Avoid division by zero
    if p95 - p5 < 1:
        p5 = l.min()
        p95 = l.max()
        if p95 - p5 < 1:
            # Image is essentially flat, nothing to normalize
            return image

    # Output range: 5% to 95% of 255
    out_low = 0.05 * 255   # 12.75
    out_high = 0.95 * 255  # 242.25
    out_range = out_high - out_low  # 229.5

    # Calculate light gain (how much we're stretching)
    gain = out_range / (p95 - p5)

    # Linear mapping: p5 -> out_low, p95 -> out_high
    l_float = (l.astype(np.float32) - p5) / (p95 - p5)
    l_normalized = np.clip(l_float * out_range + out_low, 0, 255).astype(np.uint8)

    # Merge channels back
    lab_normalized = cv2.merge([l_normalized, a, b])

    # Convert back to BGR
    result = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    # Apply denoising based on gain
    # No denoising for gain <= 1.3, max 80% strength at gain >= 3.0
    if gain > 1.3:
        # Calculate denoising strength (0 to 0.8)
        denoise_factor = min(0.8, 0.8 * (gain - 1.3) / (3.0 - 1.3))

        # fastNlMeansDenoisingColored parameters:
        # h: filter strength for luminance (3-10 typical)
        # hColor: filter strength for color components
        # templateWindowSize: should be odd (7 is default)
        # searchWindowSize: should be odd (21 is default)
        h = denoise_factor * 10.0  # 0 to 8
        if h >= 1.0:  # Only apply if strength is meaningful
            result = cv2.fastNlMeansDenoisingColored(
                result,
                None,
                h=h,
                hColor=h,
                templateWindowSize=7,
                searchWindowSize=21
            )

    return result


def scale_to_normalized_size(image: np.ndarray, target_size: int = 384, normalize_light: bool = True) -> np.ndarray:
    """Scale image so longest axis equals target_size and optionally normalize lighting.

    Args:
        image: Input image
        target_size: Target size for longest axis
        normalize_light: If True, apply lighting normalization

    Returns:
        Scaled (and optionally lighting-normalized) image
    """
    h, w = image.shape[:2]

    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)

    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if normalize_light:
        scaled = normalize_lighting(scaled)

    return scaled
