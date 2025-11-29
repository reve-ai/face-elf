"""Display module for face detection visualization."""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .detector import Face


# Colors (BGR format)
BOX_COLOR = (0, 255, 0)  # Green
CENTRAL_BOX_COLOR = (0, 255, 255)  # Yellow - for central face in capture mode
LANDMARK_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 255, 0)  # Green
CAPTURE_MODE_COLOR = (0, 0, 255)  # Red for capture mode indicator
SIDEBAR_BG_COLOR = (40, 40, 40)  # Dark gray


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


def draw_central_face(
    image: np.ndarray,
    face: Face,
    draw_landmarks: bool = True
) -> np.ndarray:
    """Draw the central face with highlighted box.

    Args:
        image: BGR image to draw on
        face: Central face to highlight
        draw_landmarks: Whether to draw facial landmarks

    Returns:
        Image with drawings
    """
    x1, y1, x2, y2 = face.bbox

    # Draw thicker yellow bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), CENTRAL_BOX_COLOR, 3)

    # Draw "CAPTURE" label
    label = "CAPTURE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    cv2.rectangle(
        image,
        (x1, y2),
        (x1 + text_w + 4, y2 + text_h + baseline + 4),
        CENTRAL_BOX_COLOR,
        -1
    )
    cv2.putText(
        image,
        label,
        (x1 + 2, y2 + text_h + 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )

    # Draw landmarks
    if draw_landmarks and face.landmarks is not None:
        for x, y in face.landmarks:
            cv2.circle(image, (int(x), int(y)), 3, CENTRAL_BOX_COLOR, -1)

    return image


def draw_capture_mode_indicator(image: np.ndarray, capture_count: int = 0) -> np.ndarray:
    """Draw capture mode indicator.

    Args:
        image: BGR image to draw on
        capture_count: Number of captures so far

    Returns:
        Image with indicator
    """
    h, w = image.shape[:2]

    # Draw "CAPTURE MODE" banner at top
    label = f"CAPTURE MODE - {capture_count} captured"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Center the banner
    x = (w - text_w) // 2
    y = 30

    cv2.rectangle(image, (x - 10, y - text_h - 5), (x + text_w + 10, y + baseline + 5), CAPTURE_MODE_COLOR, -1)
    cv2.putText(image, label, (x, y), font, font_scale, TEXT_COLOR, thickness)

    return image


def create_frame_with_sidebar(
    main_frame: np.ndarray,
    thumbnails: List[np.ndarray],
    sidebar_width: int = 140,
    thumbnail_size: int = 128
) -> np.ndarray:
    """Create a combined frame with main view and thumbnail sidebar.

    Args:
        main_frame: Main camera frame
        thumbnails: List of thumbnail images to show in sidebar
        sidebar_width: Width of sidebar in pixels
        thumbnail_size: Size of each thumbnail

    Returns:
        Combined frame with sidebar
    """
    h, w = main_frame.shape[:2]

    # Create sidebar
    sidebar = np.full((h, sidebar_width, 3), SIDEBAR_BG_COLOR, dtype=np.uint8)

    # Draw thumbnails
    padding = (sidebar_width - thumbnail_size) // 2
    y_offset = padding

    for thumb in thumbnails:
        if y_offset + thumbnail_size > h - padding:
            break  # No more room

        # Ensure thumbnail is correct size
        if thumb.shape[0] != thumbnail_size or thumb.shape[1] != thumbnail_size:
            thumb = cv2.resize(thumb, (thumbnail_size, thumbnail_size))

        # Draw thumbnail with border
        cv2.rectangle(sidebar, (padding - 2, y_offset - 2),
                     (padding + thumbnail_size + 2, y_offset + thumbnail_size + 2),
                     (100, 100, 100), 1)
        sidebar[y_offset:y_offset + thumbnail_size, padding:padding + thumbnail_size] = thumb

        y_offset += thumbnail_size + padding

    # Combine main frame and sidebar
    combined = np.hstack([main_frame, sidebar])

    return combined


def draw_status_bar(
    image: np.ndarray,
    mode: str,
    keys_help: str
) -> np.ndarray:
    """Draw status bar at bottom of image.

    Args:
        image: BGR image to draw on
        mode: Current mode name
        keys_help: Help text for available keys

    Returns:
        Image with status bar
    """
    h, w = image.shape[:2]
    bar_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw black bar at bottom
    cv2.rectangle(image, (0, h - bar_height), (w, h), (0, 0, 0), -1)

    # Draw mode on left
    mode_text = f"[{mode}]"
    cv2.putText(image, mode_text, (10, h - 10), font, 0.5, (0, 255, 255), 1)

    # Draw keys help on right
    (text_w, _), _ = cv2.getTextSize(keys_help, font, 0.5, 1)
    cv2.putText(image, keys_help, (w - text_w - 10, h - 10), font, 0.5, (200, 200, 200), 1)

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
