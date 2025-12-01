"""GUI entry point for face detection and recognition using imgui."""

import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np

from .camera import Camera, ThreadedCamera, find_available_camera, get_supported_resolutions
from .config import AppConfig
from .detector import FaceDetector
from .face_capture import FaceCaptureSession, find_central_face, crop_face, scale_to_normalized_size
from .embedder import FaceEmbedder
from .face_database import FaceDatabase, UnknownFaceTracker
from .gui.app import ImguiApp, AppState


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


def main():
    """Main GUI application entry point."""
    parser = argparse.ArgumentParser(description="Real-time face detection and recognition (GUI)")
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
        "--cpu",
        action="store_true",
        help="Force CPU inference (disable TensorRT GPU)"
    )
    parser.add_argument(
        "--no-recognition",
        action="store_true",
        help="Disable face recognition (detection only)"
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Initial window width"
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Initial window height"
    )

    args = parser.parse_args()

    # Load configuration
    config = AppConfig()
    config.load()

    # Command-line args override config file
    if args.width != 640 or args.height != 480:
        # User specified custom resolution on command line
        config.camera_resolution = (args.width, args.height)
    if args.conf_threshold != 0.5:
        config.conf_threshold = args.conf_threshold
    if args.similarity_threshold != 0.4:
        config.similarity_threshold = args.similarity_threshold

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
    embedder: Optional[FaceEmbedder] = None
    face_db: Optional[FaceDatabase] = None
    unknown_tracker: Optional[UnknownFaceTracker] = None

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

    # Probe supported resolutions
    print("Probing camera resolutions...")
    available_resolutions = get_supported_resolutions(device_id)
    if available_resolutions:
        print(f"Supported resolutions: {available_resolutions}")
    else:
        # Fallback to common defaults
        available_resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        print(f"Could not probe resolutions, using defaults: {available_resolutions}")

    # Ensure configured resolution is in the list or use a fallback
    if config.camera_resolution not in available_resolutions:
        if available_resolutions:
            config.camera_resolution = available_resolutions[0]
            print(f"Configured resolution not available, using {config.camera_resolution}")

    # Initialize camera with threaded capture
    camera = ThreadedCamera(device_id, config.camera_width, config.camera_height)
    if not camera.open():
        print(f"Error: Could not open camera {device_id}")
        return 1

    # Initialize GUI
    app = ImguiApp("Face Recognition", args.window_width, args.window_height)
    if not app.init():
        print("Error: Could not initialize GUI")
        camera.release()
        return 1

    # Set initial state
    app.state.conf_threshold = config.conf_threshold
    app.state.similarity_threshold = config.similarity_threshold
    app.state.mode = "recognition" if recognition_enabled else "detection"
    app.state.available_resolutions = available_resolutions
    app.state.current_resolution = config.camera_resolution
    if face_db:
        app.state.known_faces_count = face_db.get_person_count()

    # Initialize capture session
    capture_session = FaceCaptureSession()

    print("\nRunning face detection" + (" and recognition" if recognition_enabled else "") + " (GUI)...")
    print("Use the GUI buttons or keyboard shortcuts to control the application.")

    # FPS tracking (UI frame rate)
    ui_fps = 0.0
    ui_frame_count = 0
    ui_fps_update_interval = 30
    ui_start_time = time.time()

    # State for rendering (persists between frames)
    current_frame: Optional[np.ndarray] = None
    face_data_list: List[Dict[str, Any]] = []

    try:
        while not app.should_close():
            # Handle state changes from GUI
            _handle_state_actions(
                app, capture_session, face_db, recognition_enabled,
                camera, config
            )

            # Browse modes - skip camera and detection
            if app.state.mode in ("browse", "unknown_browse"):
                app.begin_frame()
                app.render_ui()
                app.end_frame()
                continue

            # Update detector threshold if changed
            detector.conf_threshold = app.state.conf_threshold
            if face_db:
                face_db.similarity_threshold = app.state.similarity_threshold

            # Get latest frame from camera (non-blocking)
            ret, frame, is_new_frame = camera.get_latest_frame()

            # Process new frames for face detection
            if ret and is_new_frame and frame is not None:
                current_frame = frame
                frame_h, frame_w = frame.shape[:2]

                # Detect faces
                faces = detector.detect(frame)

                # Process faces and build face data for rendering
                face_data_list = []

                if app.state.mode == "capture":
                    # Find central face
                    central_face = find_central_face(faces, frame_w, frame_h)

                    for face in faces:
                        is_central = (face is central_face)
                        face_data_list.append({
                            'face': face,
                            'name': None,
                            'similarity': 0,
                            'is_central': is_central
                        })

                elif recognition_enabled and embedder and face_db:
                    # Recognition mode
                    for face in faces:
                        cropped = crop_face(frame, face)
                        if cropped is None:
                            face_data_list.append({
                                'face': face,
                                'name': None,
                                'similarity': 0,
                                'is_central': False
                            })
                            continue

                        normalized = scale_to_normalized_size(cropped, 384)
                        embedding = embedder.get_embedding(normalized)
                        name, similarity = face_db.find_match(embedding)

                        face_data_list.append({
                            'face': face,
                            'name': name,
                            'similarity': similarity,
                            'is_central': False
                        })

                        # Save unknown faces (only if bounding box area >= 12000 pixels)
                        if name is None and unknown_tracker:
                            bbox_w = face.bbox[2] - face.bbox[0]
                            bbox_h = face.bbox[3] - face.bbox[1]
                            if bbox_w * bbox_h >= 12000:
                                unknown_tracker.maybe_save_face(normalized)

                else:
                    # Detection only
                    for face in faces:
                        face_data_list.append({
                            'face': face,
                            'name': None,
                            'similarity': 0,
                            'is_central': False
                        })

                # Handle capture face action (needs frame and faces)
                if app.state.mode == "capture" and app.state.action_capture_face:
                    central_face = find_central_face(faces, frame_w, frame_h)
                    if central_face:
                        captured = capture_session.capture_face(frame, central_face)
                        if captured:
                            thumbnail = capture_session.captures[-1].thumbnail
                            app.add_capture_thumbnail(thumbnail)
                            print(f"Captured face #{capture_session.get_capture_count()}")
                    else:
                        print("No face detected to capture")
                    app.state.action_capture_face = False

            # Update UI FPS (tracks render rate, not camera rate)
            ui_frame_count += 1
            if ui_frame_count % ui_fps_update_interval == 0:
                elapsed = time.time() - ui_start_time
                ui_fps = ui_frame_count / elapsed
                # Combine UI FPS with camera FPS for display
                camera_fps = camera.get_capture_fps()
                app.state.fps = ui_fps
                app.state.camera_fps = camera_fps
                if ui_frame_count >= 300:
                    ui_frame_count = 0
                    ui_start_time = time.time()

            # Render GUI (always, even without new frame)
            app.begin_frame()
            if current_frame is not None:
                app.update_video(current_frame)
            app.render_ui(face_data_list)
            app.end_frame()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        camera.release()
        app.shutdown()

    print("Done")
    return 0


def _handle_state_actions(
    app: ImguiApp,
    capture_session: FaceCaptureSession,
    face_db: Optional[FaceDatabase],
    recognition_enabled: bool,
    camera: Union[Camera, ThreadedCamera],
    config: AppConfig
):
    """Handle state actions triggered by GUI."""
    state = app.state

    # Toggle fullscreen
    if state.action_toggle_fullscreen:
        app.toggle_fullscreen()
        state.action_toggle_fullscreen = False

    # Change camera resolution
    if state.action_change_resolution is not None:
        new_res = state.action_change_resolution
        state.action_change_resolution = None
        if camera.set_resolution(new_res[0], new_res[1]):
            state.current_resolution = new_res
            config.camera_resolution = new_res
            config.save()

    # Enter capture mode
    if state.action_capture_mode:
        print("Entering capture mode")
        state.mode = "capture"
        capture_session.enter_capture_mode()
        app.clear_captures()
        state.action_capture_mode = False

    # Note: action_capture_face is handled in main loop where frame/faces are available

    # Save captures
    if state.action_save_captures:
        name = state.capture_name.strip()
        if name and capture_session.get_capture_count() > 0:
            # Sanitize name
            name = "".join(c for c in name if c.isalnum() or c in "._- ")
            capture_session.save_captures(name)
            print(f"Saved captures for {name}")

            # Reload database
            if recognition_enabled and face_db:
                print("Reloading known faces database...")
                face_db.load_known_faces()
                app.state.known_faces_count = face_db.get_person_count()

            # Exit capture mode
            capture_session.exit_capture_mode(discard=False)
            app.clear_captures()
            state.mode = "recognition" if recognition_enabled else "detection"

        state.action_save_captures = False

    # Cancel capture
    if state.action_cancel_capture:
        print("Cancelling capture mode")
        capture_session.exit_capture_mode(discard=True)
        app.clear_captures()
        state.mode = "recognition" if recognition_enabled else "detection"
        state.action_cancel_capture = False

    # Reload database
    if state.action_reload_db:
        if recognition_enabled and face_db:
            print("Reloading known faces database...")
            face_db.load_known_faces()
            app.state.known_faces_count = face_db.get_person_count()
        state.action_reload_db = False

    # Enter browse mode
    if state.action_browse_mode:
        print("Entering browse mode")
        app.load_browse_data()
        state.mode = "browse"
        state.action_browse_mode = False

    # Exit browse mode
    if state.action_exit_browse:
        if state.browse_selected_person:
            # Go back to person list first
            state.browse_selected_person = None
            state.browse_person_images.clear()
        else:
            # Exit browse mode entirely
            print("Exiting browse mode")
            app.clear_browse_data()
            state.mode = "recognition" if recognition_enabled else "detection"
            # Reload database in case images were deleted
            if recognition_enabled and face_db:
                print("Reloading known faces database...")
                face_db.load_known_faces()
                app.state.known_faces_count = face_db.get_person_count()
        state.action_exit_browse = False

    # Enter unknown faces browse mode
    if state.action_unknown_browse_mode:
        print("Entering unknown faces browse mode")
        app.load_unknown_faces_data()
        state.mode = "unknown_browse"
        state.action_unknown_browse_mode = False

    # Exit unknown faces browse mode
    if state.action_exit_unknown_browse:
        if state.unknown_selected_day:
            # Go back to day list first
            state.unknown_selected_day = None
            state.unknown_faces.clear()
        else:
            # Exit unknown browse mode entirely
            print("Exiting unknown faces browse mode")
            app.clear_unknown_faces_data()
            state.mode = "recognition" if recognition_enabled else "detection"
            # Reload database in case faces were moved to known
            if recognition_enabled and face_db:
                print("Reloading known faces database...")
                face_db.load_known_faces()
                app.state.known_faces_count = face_db.get_person_count()
        state.action_exit_unknown_browse = False


# Handle capture face action in main loop
def _try_capture_face(
    app: ImguiApp,
    capture_session: FaceCaptureSession,
    frame,
    faces,
    frame_w: int,
    frame_h: int
):
    """Try to capture the central face."""
    central_face = find_central_face(faces, frame_w, frame_h)
    if central_face:
        captured = capture_session.capture_face(frame, central_face)
        if captured:
            # Add thumbnail to GUI
            thumbnail = capture_session.captures[-1].thumbnail
            app.add_capture_thumbnail(thumbnail)
            print(f"Captured face #{capture_session.get_capture_count()}")
            return True
    else:
        print("No face detected to capture")
    return False


if __name__ == "__main__":
    exit(main())
