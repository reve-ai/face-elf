"""Utility to preview and apply lighting normalization to face images."""

import argparse
import sys
import cv2
import numpy as np

from .face_capture import normalize_lighting


def compute_gain_info(image: np.ndarray) -> tuple:
    """Compute gain and denoise info for an image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    # Center 50% for p95
    h, w = l.shape
    y1, y2 = h // 4, h * 3 // 4
    x1, x2 = w // 4, w * 3 // 4
    l_center = l[y1:y2, x1:x2]

    # p5 from entire image, p95 from center
    p5 = np.percentile(l, 5)
    p95 = np.percentile(l_center, 95)

    if p95 - p5 < 1:
        return 1.0, 0.0

    # Output range is 5% to 95% of 255 = 229.5
    out_range = 0.9 * 255  # 229.5
    gain = out_range / (p95 - p5)

    if gain <= 1.3:
        denoise_pct = 0.0
    else:
        # Max 80% at gain 3.0
        denoise_pct = min(80.0, 80.0 * (gain - 1.3) / (3.0 - 1.3))

    return gain, denoise_pct


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preview and apply lighting normalization to face images"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image file"
    )
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        return 1

    # Compute gain info
    gain, denoise_pct = compute_gain_info(image)

    # Normalize lighting
    normalized = normalize_lighting(image)

    # Create side-by-side comparison
    h, w = image.shape[:2]

    # Add labels
    label_height = 50
    before_labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    before_labeled[label_height:, :] = image
    cv2.putText(before_labeled, "Before", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    after_labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    after_labeled[label_height:, :] = normalized
    cv2.putText(after_labeled, "After (normalized)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(after_labeled, f"gain={gain:.2f} denoise={denoise_pct:.0f}%", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Combine side by side
    combined = np.hstack([before_labeled, after_labeled])

    window_name = f"Normalize: {args.image}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)

    print(f"Showing: {args.image}")
    print(f"Light gain: {gain:.2f}, Denoise strength: {denoise_pct:.0f}%")
    print("Press 's' to save normalized image (overwrites original)")
    print("Press ESC to exit without saving")

    while True:
        key = cv2.waitKey(100) & 0xFF

        if key == 27:  # ESC
            print("Exiting without saving")
            break
        elif key == ord('s'):
            cv2.imwrite(args.image, normalized)
            print(f"Saved normalized image to: {args.image}")
            break

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed, exiting without saving")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
