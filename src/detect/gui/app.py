"""Main imgui application for face detection and recognition."""

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import numpy as np
import cv2
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class BrowsePersonInfo:
    """Info about a person in browse mode."""
    name: str
    image_count: int
    thumbnail: Optional[np.ndarray] = None
    texture_id: Optional[int] = None


@dataclass
class AppState:
    """Application state shared across UI components."""
    # Mode
    mode: str = "recognition"  # "detection", "recognition", "capture", "browse"

    # Detection settings
    conf_threshold: float = 0.5
    similarity_threshold: float = 0.4

    # Stats
    fps: float = 0.0
    known_faces_count: int = 0

    # Capture mode
    capture_count: int = 0
    capture_name: str = ""
    capture_thumbnails: List[np.ndarray] = field(default_factory=list)

    # Browse mode
    browse_people: List[BrowsePersonInfo] = field(default_factory=list)
    browse_selected_person: Optional[str] = None
    browse_person_images: List[Tuple[str, Optional[int]]] = field(default_factory=list)  # (path, texture_id)
    browse_delete_confirmed: Optional[str] = None  # path to delete after confirmation

    # Actions (set by UI, cleared by main loop)
    action_quit: bool = False
    action_capture_mode: bool = False
    action_capture_face: bool = False
    action_save_captures: bool = False
    action_cancel_capture: bool = False
    action_reload_db: bool = False
    action_toggle_fullscreen: bool = False
    action_browse_mode: bool = False
    action_exit_browse: bool = False


class TextureManager:
    """Manages OpenGL textures for video frames and thumbnails."""

    def __init__(self):
        self.video_texture: Optional[int] = None
        self.video_size: Tuple[int, int] = (0, 0)
        self.thumbnail_textures: List[int] = []
        self.browse_textures: List[int] = []  # For browse mode

    def update_video_texture(self, frame: np.ndarray) -> int:
        """Update or create video texture from BGR frame."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.video_texture is None or self.video_size != (w, h):
            # Create new texture
            if self.video_texture is not None:
                gl.glDeleteTextures(1, [self.video_texture])

            self.video_texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.video_texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                           gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb)
            self.video_size = (w, h)
        else:
            # Update existing texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.video_texture)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h,
                              gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb)

        return self.video_texture

    def create_thumbnail_texture(self, image: np.ndarray) -> int:
        """Create a new texture for a thumbnail."""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                       gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb)

        self.thumbnail_textures.append(texture_id)
        return texture_id

    def clear_thumbnails(self):
        """Delete all thumbnail textures."""
        if self.thumbnail_textures:
            gl.glDeleteTextures(len(self.thumbnail_textures), self.thumbnail_textures)
            self.thumbnail_textures.clear()

    def create_browse_texture(self, image: np.ndarray) -> int:
        """Create a texture for browse mode."""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                       gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb)

        self.browse_textures.append(texture_id)
        return texture_id

    def clear_browse_textures(self):
        """Delete all browse mode textures."""
        if self.browse_textures:
            gl.glDeleteTextures(len(self.browse_textures), self.browse_textures)
            self.browse_textures.clear()

    def cleanup(self):
        """Clean up all textures."""
        if self.video_texture is not None:
            gl.glDeleteTextures(1, [self.video_texture])
        self.clear_thumbnails()
        self.clear_browse_textures()


class ImguiApp:
    """Main imgui application window."""

    def __init__(self, title: str = "Face Recognition", width: int = 1280, height: int = 720):
        self.title = title
        self.width = width
        self.height = height
        self.window = None
        self.impl = None
        self.texture_manager = TextureManager()
        self.state = AppState()
        self.is_fullscreen = False
        self.windowed_size = (width, height)
        self.windowed_pos = (100, 100)
        self.dpi_scale = 1.0

        # Thumbnail texture IDs (parallel to state.capture_thumbnails)
        self.thumbnail_texture_ids: List[int] = []

    def init(self) -> bool:
        """Initialize GLFW and imgui."""
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False

        # Set OpenGL version hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)

        # Create window
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync

        # Get DPI scale
        self.dpi_scale = glfw.get_window_content_scale(self.window)[0]
        print(f"DPI scale: {self.dpi_scale}")

        # Initialize imgui
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        # Set up key callback
        glfw.set_key_callback(self.window, self._key_callback)

        # Configure DPI scaling
        self._setup_dpi_scaling()

        # Style
        self._setup_style()

        return True

    def _setup_dpi_scaling(self):
        """Configure imgui for high-DPI displays."""
        io = imgui.get_io()

        # Scale fonts - use a larger base size for readability
        font_size = 18.0 * self.dpi_scale  # 36px at 2x DPI

        # Clear and rebuild font atlas with scaled font
        io.fonts.clear()

        # Try to load a system font for better readability
        font_loaded = False
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]

        for font_path in font_paths:
            try:
                import os
                if os.path.exists(font_path):
                    io.fonts.add_font_from_file_ttf(font_path, font_size)
                    font_loaded = True
                    print(f"Loaded font: {font_path} at {font_size}px")
                    break
            except Exception as e:
                continue

        if not font_loaded:
            # Fall back to default font with glyph ranges for scaling
            io.fonts.add_font_default()
            print(f"Using default font (DPI scaling may be limited)")

        # Rebuild font atlas
        self.impl.refresh_font_texture()

        # Scale style manually with more aggressive padding for buttons
        style = imgui.get_style()
        style.window_padding = (8 * self.dpi_scale, 8 * self.dpi_scale)
        style.window_rounding = 4 * self.dpi_scale
        style.frame_padding = (8 * self.dpi_scale, 6 * self.dpi_scale)  # More vertical padding
        style.frame_rounding = 3 * self.dpi_scale
        style.item_spacing = (8 * self.dpi_scale, 6 * self.dpi_scale)
        style.item_inner_spacing = (6 * self.dpi_scale, 6 * self.dpi_scale)
        style.indent_spacing = 20 * self.dpi_scale
        style.scrollbar_size = 16 * self.dpi_scale
        style.scrollbar_rounding = 3 * self.dpi_scale
        style.grab_min_size = 12 * self.dpi_scale
        style.grab_rounding = 3 * self.dpi_scale

    def _setup_style(self):
        """Configure imgui style."""
        style = imgui.get_style()
        style.window_rounding = 4.0
        style.frame_rounding = 2.0
        style.scrollbar_rounding = 2.0

        # Dark color scheme
        imgui.style_colors_dark()

    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if action != glfw.PRESS:
            return

        # Check if imgui wants keyboard input (e.g., text field is active)
        io = imgui.get_io()
        imgui_wants_keyboard = io.want_capture_keyboard

        # Global shortcuts (always work)
        if key == glfw.KEY_Q and not imgui_wants_keyboard:
            self.state.action_quit = True
        elif key == glfw.KEY_F11:
            self.state.action_toggle_fullscreen = True
        elif key == glfw.KEY_ESCAPE:
            if self.state.mode == "capture":
                self.state.action_cancel_capture = True
            elif self.state.mode == "browse":
                self.state.action_exit_browse = True
            else:
                self.state.action_quit = True

        # Skip other shortcuts if imgui wants keyboard (typing in text field)
        if imgui_wants_keyboard:
            return

        # Mode-specific shortcuts
        if self.state.mode in ("detection", "recognition"):
            if key == glfw.KEY_C:
                self.state.action_capture_mode = True
            elif key == glfw.KEY_R and self.state.mode == "recognition":
                self.state.action_reload_db = True
            elif key == glfw.KEY_B:
                self.state.action_browse_mode = True

        elif self.state.mode == "capture":
            if key == glfw.KEY_SPACE:
                self.state.action_capture_face = True
            elif key == glfw.KEY_S:
                self.state.action_save_captures = True

        elif self.state.mode == "browse":
            # ESC is handled above (action_exit_browse or action_quit)
            pass

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        if self.is_fullscreen:
            # Restore windowed mode
            glfw.set_window_monitor(self.window, None,
                                   self.windowed_pos[0], self.windowed_pos[1],
                                   self.windowed_size[0], self.windowed_size[1], 0)
            self.is_fullscreen = False
        else:
            # Save current position/size
            self.windowed_pos = glfw.get_window_pos(self.window)
            self.windowed_size = glfw.get_window_size(self.window)

            # Go fullscreen
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(self.window, monitor, 0, 0,
                                   mode.size.width, mode.size.height, mode.refresh_rate)
            self.is_fullscreen = True

    def should_close(self) -> bool:
        """Check if window should close."""
        return glfw.window_should_close(self.window) or self.state.action_quit

    def begin_frame(self):
        """Begin a new frame."""
        glfw.poll_events()
        self.impl.process_inputs()
        imgui.new_frame()

    def end_frame(self):
        """End frame and render."""
        imgui.render()

        # Get framebuffer size for proper scaling
        display_w, display_h = glfw.get_framebuffer_size(self.window)
        gl.glViewport(0, 0, display_w, display_h)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    def update_video(self, frame: np.ndarray):
        """Update video texture with new frame."""
        self.texture_manager.update_video_texture(frame)

    def add_capture_thumbnail(self, thumbnail: np.ndarray):
        """Add a new capture thumbnail."""
        texture_id = self.texture_manager.create_thumbnail_texture(thumbnail)
        self.thumbnail_texture_ids.append(texture_id)
        self.state.capture_count = len(self.thumbnail_texture_ids)

    def clear_captures(self):
        """Clear all capture thumbnails."""
        self.texture_manager.clear_thumbnails()
        self.thumbnail_texture_ids.clear()
        self.state.capture_count = 0
        self.state.capture_name = ""

    def load_browse_data(self, known_faces_dir: str = "known-faces"):
        """Load browse mode data - list of people with thumbnails."""
        from pathlib import Path

        self.clear_browse_data()
        known_dir = Path(known_faces_dir)

        if not known_dir.exists():
            return

        for person_dir in sorted(known_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            name = person_dir.name
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            image_count = len(images)

            if image_count == 0:
                continue

            # Load first image as thumbnail
            thumbnail = cv2.imread(str(images[0]))
            if thumbnail is None:
                continue

            # Resize thumbnail to square
            h, w = thumbnail.shape[:2]
            size = min(h, w)
            y_off = (h - size) // 2
            x_off = (w - size) // 2
            thumbnail = thumbnail[y_off:y_off+size, x_off:x_off+size]
            thumbnail = cv2.resize(thumbnail, (128, 128))

            # Create texture
            texture_id = self.texture_manager.create_browse_texture(thumbnail)

            person_info = BrowsePersonInfo(
                name=name,
                image_count=image_count,
                thumbnail=thumbnail,
                texture_id=texture_id
            )
            self.state.browse_people.append(person_info)

    def load_person_images(self, person_name: str, known_faces_dir: str = "known-faces"):
        """Load all images for a specific person."""
        from pathlib import Path

        # Clear previous person images
        self.state.browse_person_images.clear()

        person_dir = Path(known_faces_dir) / person_name
        if not person_dir.exists():
            return

        images = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))

        for img_path in images:
            # Load and create texture
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Resize for display
            h, w = img.shape[:2]
            size = min(h, w)
            y_off = (h - size) // 2
            x_off = (w - size) // 2
            img = img[y_off:y_off+size, x_off:x_off+size]
            img = cv2.resize(img, (128, 128))

            texture_id = self.texture_manager.create_browse_texture(img)
            self.state.browse_person_images.append((str(img_path), texture_id))

        self.state.browse_selected_person = person_name

    def delete_person_image(self, image_path: str):
        """Delete an image from the filesystem."""
        from pathlib import Path
        import os

        path = Path(image_path)
        if path.exists():
            os.remove(path)
            print(f"Deleted: {image_path}")
            return True
        return False

    def clear_browse_data(self):
        """Clear browse mode state."""
        self.texture_manager.clear_browse_textures()
        self.state.browse_people.clear()
        self.state.browse_selected_person = None
        self.state.browse_person_images.clear()
        self.state.browse_delete_confirmed = None

    def render_ui(self, faces: list = None):
        """Render the complete UI."""
        window_w, window_h = glfw.get_window_size(self.window)

        # Browse mode has its own full-screen layout
        if self.state.mode == "browse":
            self._render_browse_mode(window_w, window_h)
            return

        # Main menu bar height
        menu_bar_height = 0

        # Calculate layout (scale for DPI)
        control_panel_width = int(220 * self.dpi_scale)
        sidebar_width = int(150 * self.dpi_scale) if self.state.mode == "capture" else 0
        status_bar_height = int(30 * self.dpi_scale)

        video_width = window_w - control_panel_width - sidebar_width
        video_height = window_h - status_bar_height - menu_bar_height

        # Set up docking/layout
        imgui.set_next_window_position(0, menu_bar_height)
        imgui.set_next_window_size(video_width, video_height)
        self._render_video_panel(video_width, video_height, faces)

        imgui.set_next_window_position(video_width, menu_bar_height)
        imgui.set_next_window_size(control_panel_width, video_height)
        self._render_control_panel(control_panel_width, video_height)

        if self.state.mode == "capture":
            imgui.set_next_window_position(video_width + control_panel_width, menu_bar_height)
            imgui.set_next_window_size(sidebar_width, video_height)
            self._render_sidebar(sidebar_width, video_height)

        imgui.set_next_window_position(0, window_h - status_bar_height)
        imgui.set_next_window_size(window_w, status_bar_height)
        self._render_status_bar(window_w, status_bar_height)

    def _render_video_panel(self, width: float, height: float, faces: list = None):
        """Render the video panel with face overlays."""
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR)

        imgui.begin("Video", flags=flags)

        if self.texture_manager.video_texture is not None:
            # Calculate scaled size to fit panel while maintaining aspect ratio
            tex_w, tex_h = self.texture_manager.video_size
            available_w = width - 20
            available_h = height - 20

            scale = min(available_w / tex_w, available_h / tex_h)
            display_w = int(tex_w * scale)
            display_h = int(tex_h * scale)

            # Center the image
            cursor_x = (available_w - display_w) / 2 + 10
            cursor_y = (available_h - display_h) / 2 + 10
            imgui.set_cursor_pos((cursor_x, cursor_y))

            # Get position for overlay drawing
            window_pos = imgui.get_window_position()
            img_pos = (window_pos.x + cursor_x, window_pos.y + cursor_y)

            # Draw video
            imgui.image(self.texture_manager.video_texture, display_w, display_h)

            # Draw face overlays
            if faces:
                draw_list = imgui.get_window_draw_list()
                self._draw_face_overlays(draw_list, faces, img_pos, scale)
        else:
            imgui.text("No video feed")

        imgui.end()

    def _draw_face_overlays(self, draw_list, faces, img_pos, scale):
        """Draw face detection/recognition overlays."""
        for face_data in faces:
            face = face_data.get('face')
            name = face_data.get('name')
            similarity = face_data.get('similarity', 0)
            is_central = face_data.get('is_central', False)

            if face is None:
                continue

            x1, y1, x2, y2 = face.bbox

            # Scale and offset
            x1 = img_pos[0] + x1 * scale
            y1 = img_pos[1] + y1 * scale
            x2 = img_pos[0] + x2 * scale
            y2 = img_pos[1] + y2 * scale

            # Determine color
            if is_central:
                color = imgui.get_color_u32_rgba(1, 1, 0, 1)  # Yellow
                thickness = 3.0
                label = "CAPTURE"
            elif name:
                color = imgui.get_color_u32_rgba(0, 1, 0, 1)  # Green
                thickness = 2.0
                label = f"{name} ({similarity:.0%})"
            else:
                color = imgui.get_color_u32_rgba(1, 0, 0, 1)  # Red
                thickness = 2.0
                label = "Unknown"

            # Draw box
            draw_list.add_rect(x1, y1, x2, y2, color, thickness=thickness * self.dpi_scale)

            # Draw label background (scale for DPI)
            font_size = 16 * self.dpi_scale
            label_h = font_size + 4 * self.dpi_scale
            label_w = len(label) * 8 * self.dpi_scale + 8 * self.dpi_scale

            bg_color = imgui.get_color_u32_rgba(0, 0, 0, 0.7)
            draw_list.add_rect_filled(x1, y1 - label_h, x1 + label_w, y1, bg_color)

            # Draw label text
            text_color = imgui.get_color_u32_rgba(1, 1, 1, 1)
            draw_list.add_text(x1 + 4 * self.dpi_scale, y1 - label_h + 2 * self.dpi_scale, text_color, label)

            # Draw landmarks if available
            if hasattr(face, 'landmarks') and face.landmarks is not None:
                landmark_color = color
                for lx, ly in face.landmarks:
                    cx = img_pos[0] + lx * scale
                    cy = img_pos[1] + ly * scale
                    draw_list.add_circle_filled(cx, cy, 3 * scale * self.dpi_scale, landmark_color)

    def _render_control_panel(self, width: float, height: float):
        """Render the control panel with buttons and sliders."""
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE)

        imgui.begin("Controls", flags=flags)

        # Mode display
        mode_colors = {
            "detection": (0.2, 0.6, 1.0),
            "recognition": (0.2, 0.8, 0.2),
            "capture": (1.0, 0.8, 0.2)
        }
        color = mode_colors.get(self.state.mode, (1, 1, 1))
        imgui.text_colored(f"Mode: {self.state.mode.upper()}", *color)

        imgui.separator()

        # Stats
        imgui.text(f"FPS: {self.state.fps:.1f}")
        if self.state.mode == "recognition":
            imgui.text(f"Known faces: {self.state.known_faces_count}")

        imgui.separator()
        imgui.spacing()

        # Sliders
        imgui.text("Detection Confidence")
        changed, self.state.conf_threshold = imgui.slider_float(
            "##conf", self.state.conf_threshold, 0.1, 1.0, "%.2f"
        )

        if self.state.mode == "recognition":
            imgui.spacing()
            imgui.text("Similarity Threshold")
            changed, self.state.similarity_threshold = imgui.slider_float(
                "##sim", self.state.similarity_threshold, 0.1, 1.0, "%.2f"
            )

        imgui.separator()
        imgui.spacing()

        # Buttons based on mode
        button_width = width - int(30 * self.dpi_scale)

        if self.state.mode in ("detection", "recognition"):
            if imgui.button("Capture Mode (C)", width=button_width):
                self.state.action_capture_mode = True

            if self.state.mode == "recognition":
                imgui.spacing()
                if imgui.button("Reload DB (R)", width=button_width):
                    self.state.action_reload_db = True

            imgui.spacing()
            if imgui.button("Browse Faces (B)", width=button_width):
                self.state.action_browse_mode = True

        elif self.state.mode == "capture":
            imgui.text(f"Captured: {self.state.capture_count}")
            imgui.spacing()

            if imgui.button("Capture Face (SPACE)", width=button_width):
                self.state.action_capture_face = True

            imgui.spacing()
            imgui.text("Name:")
            imgui.set_next_item_width(button_width)
            changed, self.state.capture_name = imgui.input_text(
                "##name", self.state.capture_name, 64
            )

            imgui.spacing()
            # Disable save if no captures or no name
            can_save = self.state.capture_count > 0 and len(self.state.capture_name.strip()) > 0

            if not can_save:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

            if imgui.button("Save (S)", width=button_width) and can_save:
                self.state.action_save_captures = True

            if not can_save:
                imgui.pop_style_var()

            imgui.spacing()
            if imgui.button("Cancel (ESC)", width=button_width):
                self.state.action_cancel_capture = True

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.button("Fullscreen (F11)", width=button_width):
            self.state.action_toggle_fullscreen = True

        imgui.spacing()
        if imgui.button("Quit (Q)", width=button_width):
            self.state.action_quit = True

        imgui.end()

    def _render_sidebar(self, width: float, height: float):
        """Render thumbnail sidebar for capture mode."""
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE)

        imgui.begin("Captures", flags=flags)
        imgui.text("Captures")
        imgui.separator()

        thumb_size = width - int(20 * self.dpi_scale)

        for i, tex_id in enumerate(self.thumbnail_texture_ids):
            imgui.text(f"#{i + 1}")
            imgui.image(tex_id, thumb_size, thumb_size)
            imgui.spacing()

        imgui.end()

    def _render_status_bar(self, width: float, height: float):
        """Render status bar at bottom."""
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR)

        imgui.begin("Status", flags=flags)

        # Mode indicator
        mode_text = f"[{self.state.mode.upper()}]"
        imgui.text(mode_text)

        imgui.same_line(spacing=20)

        # Keyboard shortcuts help
        if self.state.mode == "capture":
            help_text = "SPACE=capture  S=save  ESC=cancel"
        elif self.state.mode == "recognition":
            help_text = "C=capture  R=reload  B=browse  Q=quit  F11=fullscreen"
        elif self.state.mode == "browse":
            help_text = "ESC=back/close"
        else:
            help_text = "C=capture  B=browse  Q=quit  F11=fullscreen"

        imgui.text_colored(help_text, 0.7, 0.7, 0.7)

        imgui.end()

    def _render_browse_mode(self, window_w: float, window_h: float):
        """Render the browse faces mode."""
        # If a person is selected, show their images
        if self.state.browse_selected_person:
            self._render_person_detail(window_w, window_h)
            return

        # Otherwise show the person grid
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE)

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(window_w, window_h)
        imgui.begin("Browse Faces", flags=flags)

        # Header
        imgui.text_colored("BROWSE KNOWN FACES", 0.2, 0.8, 0.2)
        imgui.same_line(spacing=40)
        if imgui.button("Close (ESC)"):
            self.state.action_exit_browse = True
        imgui.separator()
        imgui.spacing()

        # Calculate grid layout
        tile_size = int(160 * self.dpi_scale)
        padding = int(10 * self.dpi_scale)
        thumb_size = int(128 * self.dpi_scale)

        available_width = window_w - int(40 * self.dpi_scale)
        cols = max(1, int(available_width // (tile_size + padding)))

        # Begin scrollable region
        imgui.begin_child("PersonGrid", 0, 0, border=False)

        for i, person in enumerate(self.state.browse_people):
            col = i % cols

            if col > 0:
                imgui.same_line(spacing=padding)

            # Person tile
            imgui.begin_group()

            # Thumbnail (clickable)
            if person.texture_id:
                clicked = imgui.image_button(
                    person.texture_id, thumb_size, thumb_size,
                    frame_padding=2
                )
                if clicked:
                    self.load_person_images(person.name)

            # Name and count below
            imgui.text(person.name)
            imgui.text_colored(f"{person.image_count} images", 0.6, 0.6, 0.6)

            imgui.end_group()

            # Add spacing after each row
            if col == cols - 1:
                imgui.spacing()

        imgui.end_child()
        imgui.end()

    def _render_person_detail(self, window_w: float, window_h: float):
        """Render detail view for a selected person."""
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE)

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(window_w, window_h)
        imgui.begin("Person Detail", flags=flags)

        # Header
        person_name = self.state.browse_selected_person
        imgui.text_colored(f"IMAGES: {person_name}", 0.2, 0.8, 0.2)
        imgui.same_line(spacing=40)
        if imgui.button("Back (ESC)"):
            # Go back to person list
            self.state.browse_selected_person = None
            self.state.browse_person_images.clear()
        imgui.separator()
        imgui.spacing()

        # Calculate grid layout
        tile_size = int(160 * self.dpi_scale)
        padding = int(10 * self.dpi_scale)
        thumb_size = int(128 * self.dpi_scale)

        available_width = window_w - int(40 * self.dpi_scale)
        cols = max(1, int(available_width // (tile_size + padding)))

        # Begin scrollable region
        imgui.begin_child("ImageGrid", 0, 0, border=False)

        # Track image to delete
        delete_path = None

        for i, (img_path, tex_id) in enumerate(self.state.browse_person_images):
            col = i % cols

            if col > 0:
                imgui.same_line(spacing=padding)

            # Image tile
            imgui.begin_group()

            # Thumbnail
            if tex_id:
                imgui.image(tex_id, thumb_size, thumb_size)

            # Delete button
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1.0, 0.3, 0.3, 1.0)
            if imgui.button(f"Delete##{i}", width=thumb_size):
                delete_path = img_path
            imgui.pop_style_color(2)

            imgui.end_group()

            # Add spacing after each row
            if col == cols - 1:
                imgui.spacing()

        imgui.end_child()

        # Handle deletion
        if delete_path:
            if self.delete_person_image(delete_path):
                # Reload person images
                self.texture_manager.clear_browse_textures()
                # Reload browse data to update counts
                self.load_browse_data()
                self.load_person_images(person_name)

        imgui.end()

    def shutdown(self):
        """Clean up resources."""
        self.texture_manager.cleanup()
        if self.impl:
            self.impl.shutdown()
        glfw.terminate()
