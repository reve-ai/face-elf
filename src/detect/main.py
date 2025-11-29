"""Main entry point for face detection application."""

import argparse
import time
from pathlib import Path

from .camera import Camera, find_available_camera
from .detector import FaceDetector
from .display import DisplayWindow, draw_faces, draw_fps, draw_face_count


def get_model_path() -> Path:
    """Get the default model path."""
    # Look for model relative to this file's location
    src_dir = Path(__file__).parent.parent.parent
    model_path = src_dir / "models" / "det_10g.onnx"
    return model_path


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Real-time face detection")
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=-1,
        help="Camera device ID (-1 for auto-detect)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to SCRFD ONNX model"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera capture width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera capture height"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Don't draw facial landmarks"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (disable TensorRT GPU)"
    )

    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = get_model_path()

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run setup.sh to download the model")
        return 1

    print(f"Loading model from {model_path}...")

    # Initialize face detector
    detector = FaceDetector(
        model_path=str(model_path),
        conf_threshold=args.conf_threshold,
        use_gpu=not args.cpu
    )
    print("Model loaded successfully")

    # Find camera
    if args.camera == -1:
        device_id = find_available_camera()
        if device_id == -1:
            print("Error: No camera found")
            return 1
        print(f"Auto-detected camera: /dev/video{device_id}")
    else:
        device_id = args.camera

    # Initialize camera
    camera = Camera(device_id, args.width, args.height)
    if not camera.open():
        print(f"Error: Could not open camera {device_id}")
        return 1

    # Initialize display
    window = DisplayWindow("Face Detection - Press 'q' to quit")
    window.create()

    print("\nRunning face detection...")
    print("Press 'q' to quit")
    print("-" * 40)

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_update_interval = 10
    start_time = time.time()

    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Detect faces
            faces = detector.detect(frame)

            # Draw results
            draw_faces(frame, faces, draw_landmarks=not args.no_landmarks)
            draw_fps(frame, fps)
            draw_face_count(frame, len(faces))

            # Display
            window.show(frame)

            # Update FPS
            frame_count += 1
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                if frame_count >= 100:
                    # Reset to avoid precision issues over long runs
                    frame_count = 0
                    start_time = time.time()

            # Check for quit
            key = window.wait_key(1)
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        camera.release()
        window.close()

    print("Done")
    return 0


if __name__ == "__main__":
    exit(main())
