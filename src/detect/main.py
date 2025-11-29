"""Main entry point for face detection and recognition application."""

import argparse
import time
from pathlib import Path

from .camera import Camera, find_available_camera
from .detector import FaceDetector
from .display import (
    DisplayWindow, draw_faces, draw_fps, draw_face_count,
    draw_central_face, draw_capture_mode_indicator, create_frame_with_sidebar,
    draw_status_bar, draw_recognized_face
)
from .face_capture import FaceCaptureSession, find_central_face, crop_face, scale_to_normalized_size
from .embedder import FaceEmbedder
from .face_database import FaceDatabase, UnknownFaceTracker


def get_model_path() -> Path:
    """Get the default model path."""
    src_dir = Path(__file__).parent.parent.parent
    model_path = src_dir / "models" / "det_10g.onnx"
    return model_path


def get_embedding_model_path() -> Path:
    """Get the default embedding model path."""
    src_dir = Path(__file__).parent.parent.parent
    model_path = src_dir / "models" / "w600k_r50.onnx"
    return model_path


def get_name_from_user() -> str:
    """Get a name from the user via console input."""
    print("\n" + "=" * 40)
    print("Enter name for this person (or empty to cancel):")
    try:
        name = input("> ").strip()
        name = "".join(c for c in name if c.isalnum() or c in "._- ")
        return name
    except (EOFError, KeyboardInterrupt):
        return ""


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Real-time face detection and recognition")
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
        "--embedding-model",
        type=str,
        default=None,
        help="Path to ArcFace ONNX model"
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
        "--similarity-threshold",
        type=float,
        default=0.4,
        help="Face recognition similarity threshold"
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
    parser.add_argument(
        "--no-recognition",
        action="store_true",
        help="Disable face recognition (detection only)"
    )

    args = parser.parse_args()

    # Find detection model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = get_model_path()

    if not model_path.exists():
        print(f"Error: Detection model not found at {model_path}")
        print("Run setup.sh to download the model")
        return 1

    # Find embedding model
    if args.embedding_model:
        embedding_model_path = Path(args.embedding_model)
    else:
        embedding_model_path = get_embedding_model_path()

    recognition_enabled = not args.no_recognition
    if recognition_enabled and not embedding_model_path.exists():
        print(f"Warning: Embedding model not found at {embedding_model_path}")
        print("Face recognition will be disabled")
        recognition_enabled = False

    print(f"Loading detection model from {model_path}...")

    # Initialize face detector
    detector = FaceDetector(
        model_path=str(model_path),
        conf_threshold=args.conf_threshold,
        use_gpu=not args.cpu
    )
    print("Detection model loaded")

    # Initialize face embedder and database
    embedder = None
    face_db = None
    unknown_tracker = None

    if recognition_enabled:
        print(f"Loading embedding model from {embedding_model_path}...")
        embedder = FaceEmbedder(
            model_path=str(embedding_model_path),
            use_gpu=not args.cpu
        )
        print("Embedding model loaded")

        print("Loading known faces...")
        face_db = FaceDatabase(
            embedder=embedder,
            known_faces_dir="known-faces",
            similarity_threshold=args.similarity_threshold
        )
        face_db.load_known_faces()

        unknown_tracker = UnknownFaceTracker(
            output_dir="new-faces",
            min_save_interval=3.0
        )

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
    window = DisplayWindow("Face Detection & Recognition")
    window.create()

    # Initialize capture session
    capture_session = FaceCaptureSession()

    print("\nRunning face detection" + (" and recognition" if recognition_enabled else "") + "...")
    print("Controls:")
    print("  q     - Quit")
    print("  c     - Enter capture mode")
    if recognition_enabled:
        print("  r     - Reload known faces database")
    print("  SPACE - Capture face (in capture mode)")
    print("  s     - Save captures (in capture mode)")
    print("  ESC   - Exit capture mode / Quit")
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
            frame_h, frame_w = frame.shape[:2]

            # Handle capture mode
            if capture_session.is_capture_mode():
                # Find central face
                central_face = find_central_face(faces, frame_w, frame_h)

                # Draw non-central faces normally
                other_faces = [f for f in faces if f is not central_face]
                draw_faces(frame, other_faces, draw_landmarks=not args.no_landmarks)

                # Draw central face with highlight
                if central_face:
                    draw_central_face(frame, central_face, draw_landmarks=not args.no_landmarks)

                # Draw capture mode indicator
                draw_capture_mode_indicator(frame, capture_session.get_capture_count())

                # Draw status bar
                draw_status_bar(frame, "CAPTURE", "SPACE=capture  S=save  ESC=cancel")

                # Create frame with sidebar
                thumbnails = [c.thumbnail for c in capture_session.captures]
                display_frame = create_frame_with_sidebar(frame, thumbnails)

            elif recognition_enabled:
                # Recognition mode
                for face in faces:
                    # Crop and normalize face
                    cropped = crop_face(frame, face)
                    if cropped is None:
                        continue

                    normalized = scale_to_normalized_size(cropped, 384)

                    # Get embedding
                    embedding = embedder.get_embedding(normalized)

                    # Find match
                    name, similarity = face_db.find_match(embedding)

                    # Draw face with recognition result
                    draw_recognized_face(
                        frame, face, name, similarity,
                        draw_landmarks=not args.no_landmarks
                    )

                    # Save unknown faces
                    if name is None:
                        unknown_tracker.maybe_save_face(normalized)

                draw_fps(frame, fps)

                # Status bar
                known_count = face_db.get_person_count()
                draw_status_bar(
                    frame,
                    f"RECOGNITION ({known_count} known)",
                    "C=capture  R=reload  Q=quit"
                )
                display_frame = frame

            else:
                # Detection only mode
                draw_faces(frame, faces, draw_landmarks=not args.no_landmarks)
                draw_fps(frame, fps)
                draw_face_count(frame, len(faces))
                draw_status_bar(frame, "DETECTION", "C=capture mode  Q=quit")
                display_frame = frame

            # Display
            window.show(display_frame)

            # Update FPS
            frame_count += 1
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                if frame_count >= 100:
                    frame_count = 0
                    start_time = time.time()

            # Handle key presses
            key = window.wait_key(1)

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == 27:  # ESC
                if capture_session.is_capture_mode():
                    print("\nExiting capture mode (discarding captures)")
                    capture_session.exit_capture_mode(discard=True)
                else:
                    print("\nQuitting...")
                    break

            elif key == ord('c'):
                if not capture_session.is_capture_mode():
                    print("\nEntering capture mode")
                    capture_session.enter_capture_mode()

            elif key == ord(' '):  # SPACE
                if capture_session.is_capture_mode():
                    central_face = find_central_face(faces, frame_w, frame_h)
                    if central_face:
                        captured = capture_session.capture_face(frame, central_face)
                        if captured:
                            print(f"  Captured face #{capture_session.get_capture_count()}")
                    else:
                        print("  No face detected to capture")

            elif key == ord('s'):
                if capture_session.is_capture_mode() and capture_session.get_capture_count() > 0:
                    name = get_name_from_user()
                    if name:
                        capture_session.save_captures(name)
                        capture_session.exit_capture_mode(discard=False)
                        print(f"Saved and exited capture mode")
                        # Reload database to include new person
                        if recognition_enabled:
                            print("Reloading known faces database...")
                            face_db.load_known_faces()
                    else:
                        print("Save cancelled")

            elif key == ord('r'):
                if recognition_enabled and not capture_session.is_capture_mode():
                    print("\nReloading known faces database...")
                    face_db.load_known_faces()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        camera.release()
        window.close()

    print("Done")
    return 0


if __name__ == "__main__":
    exit(main())
