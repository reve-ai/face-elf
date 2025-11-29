"""Display module for face detection visualization."""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .detector import Face


# Colors (BGR format)
BOX_COLOR = (0, 255, 0)  # Green
LANDMARK_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 255, 0)  # Green


def draw_faces(
    image: np.ndarray,
    faces: List[Face],
    draw_landmarks: bool = True,
    draw_scores: bool = True
) -> np.ndarray:
    """Draw face detections on image.

    Args:
        image: BGR image to draw on (will be modified in place)
        faces: List of detected faces
        draw_landmarks: Whether to draw facial landmarks
        draw_scores: Whether to draw confidence scores

    Returns:
        Image with drawings (same as input, modified in place)
    """
    for face in faces:
        x1, y1, x2, y2 = face.bbox

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Draw score
        if draw_scores:
            label = f"{face.score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w + 4, y1),
                TEXT_BG_COLOR,
                -1
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - baseline - 2),
                font,
                font_scale,
                TEXT_COLOR,
                thickness
            )

        # Draw landmarks
        if draw_landmarks and face.landmarks is not None:
            for i, (x, y) in enumerate(face.landmarks):
                cv2.circle(image, (int(x), int(y)), 2, LANDMARK_COLOR, -1)

    return image


def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on image.

    Args:
        image: BGR image to draw on
        fps: Frames per second value

    Returns:
        Image with FPS drawn
    """
    label = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw background
    cv2.rectangle(image, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)

    # Draw text
    cv2.putText(image, label, (15, 15 + text_h), font, font_scale, (0, 255, 255), thickness)

    return image


def draw_face_count(image: np.ndarray, count: int) -> np.ndarray:
    """Draw face count on image.

    Args:
        image: BGR image to draw on
        count: Number of faces detected

    Returns:
        Image with face count drawn
    """
    label = f"Faces: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    h = image.shape[0]
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw at bottom left
    y = h - 20
    cv2.rectangle(image, (10, y - text_h - 5), (20 + text_w, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(image, label, (15, y), font, font_scale, (0, 255, 255), thickness)

    return image


class DisplayWindow:
    """Manages OpenCV display window."""

    def __init__(self, window_name: str = "Face Detection"):
        """Initialize display window.

        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        self._created = False

    def create(self) -> None:
        """Create the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._created = True

    def show(self, image: np.ndarray) -> None:
        """Display image in window.

        Args:
            image: BGR image to display
        """
        if not self._created:
            self.create()
        cv2.imshow(self.window_name, image)

    def wait_key(self, delay: int = 1) -> int:
        """Wait for key press.

        Args:
            delay: Wait time in milliseconds (0 = forever)

        Returns:
            Key code or -1 if no key pressed
        """
        return cv2.waitKey(delay) & 0xFF

    def close(self) -> None:
        """Close the display window."""
        if self._created:
            cv2.destroyWindow(self.window_name)
            self._created = False

    def __enter__(self):
        """Context manager entry."""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
