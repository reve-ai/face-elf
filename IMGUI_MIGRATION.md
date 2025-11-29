# imgui Migration Plan

## Overview

Migrate from OpenCV's `cv2.imshow()` to **imgui** with GLFW backend for a modern, GPU-accelerated UI.

## Why imgui?

- Immediate-mode GUI (rebuild UI every frame - perfect for video)
- GPU-accelerated rendering via OpenGL
- Low latency for real-time applications
- Easy to add controls (sliders, buttons) alongside video
- Similar philosophy to Dear PyGui

## Installation

```bash
pip install imgui[glfw] PyOpenGL
```

## Architecture Changes

### Current (OpenCV)
```
Camera → Frame → Draw overlays (cv2) → cv2.imshow() → cv2.waitKey()
```

### New (imgui)
```
GLFW Window + OpenGL Context
     │
     ├── Camera → Frame → OpenGL Texture
     │
     └── imgui render loop:
           ├── Video panel (texture display)
           ├── Control panel (buttons, sliders)
           ├── Sidebar (thumbnails)
           └── Status bar
```

## Component Mapping

| Current (display.py) | New (imgui) |
|---------------------|-------------|
| `cv2.imshow()` | `imgui.image()` with OpenGL texture |
| `cv2.rectangle()` | `draw_list.add_rect()` or pre-draw with cv2 |
| `cv2.putText()` | `draw_list.add_text()` or `imgui.text()` |
| `cv2.waitKey()` | GLFW key callbacks + `imgui.is_key_pressed()` |
| `DisplayWindow` class | `ImguiApp` class |

## Button & Keyboard Mapping

All actions accessible via both buttons and keyboard shortcuts:

| Action | Button Label | Keyboard | Mode |
|--------|--------------|----------|------|
| Enter capture mode | `[Capture Mode (C)]` | C | Recognition/Detection |
| Reload known faces | `[Reload DB (R)]` | R | Recognition |
| Quit application | `[Quit (Q)]` | Q | All modes |
| Capture face | `[Capture (SPACE)]` | SPACE | Capture |
| Save captures | `[Save (S)]` | S | Capture |
| Cancel/Exit capture | `[Cancel (ESC)]` | ESC | Capture |
| Toggle fullscreen | `[Fullscreen (F11)]` | F11 | All modes |

Buttons show keyboard hint in parentheses so users learn shortcuts over time.

## New File Structure

```
src/detect/
├── gui/
│   ├── __init__.py
│   ├── app.py           # Main imgui application class
│   ├── video_panel.py   # Video display with texture
│   ├── controls.py      # Buttons, sliders, mode controls
│   ├── overlays.py      # Face box drawing on video
│   └── sidebar.py       # Thumbnail sidebar for capture mode
├── main_gui.py          # New entry point using imgui
└── (existing files unchanged)
```

## Implementation Steps

### Phase 1: Basic Window & Video Display
1. Install dependencies (imgui, glfw, PyOpenGL)
2. Create `gui/app.py` with basic imgui+GLFW window
3. Create OpenGL texture from camera frames
4. Display video in imgui window
5. Verify FPS is acceptable (target: 30+ FPS)

### Phase 2: Face Detection Overlays
6. Draw face bounding boxes using imgui draw list
7. Draw landmarks as circles
8. Draw name labels for recognized faces
9. Color coding: green (recognized), red (unknown), yellow (capture target)

### Phase 3: Control Panel
10. Add mode indicator (Detection / Recognition / Capture)
11. Add FPS counter
12. Add control buttons (alternatives to keyboard shortcuts):
    - [Capture Mode] (C key)
    - [Reload DB] (R key)
    - [Quit] (Q key)
13. Add sliders: confidence threshold, similarity threshold
14. Add known faces count display
15. Show keyboard shortcut hints on buttons (e.g., "Capture Mode (C)")

### Phase 4: Capture Mode UI
16. Create thumbnail sidebar panel
17. Show captured face thumbnails
18. Add capture count indicator
19. Add capture mode buttons (alternatives to keyboard shortcuts):
    - [Capture Face] (SPACE key) - captures the central face
    - [Save & Exit] (S key) - opens name input, saves, exits capture mode
    - [Cancel] (ESC key) - discards captures and exits capture mode
20. Add text input field for person name (replaces console input)
21. Show keyboard shortcut hints on buttons

### Phase 5: Keyboard Shortcuts
22. Implement GLFW key callbacks
23. Map existing shortcuts (q, c, r, SPACE, s, ESC)
24. Ensure buttons and keyboard shortcuts trigger same actions
25. Show keyboard hints in status bar

### Phase 6: Polish & Integration
26. Add proper window resizing
27. Add fullscreen toggle (F11)
28. Clean up old cv2 display code (keep as fallback?)
29. Update documentation and help text

## Key Code Patterns

### OpenGL Texture from NumPy Array
```python
import OpenGL.GL as gl

def create_texture(image: np.ndarray) -> int:
    """Create OpenGL texture from BGR image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb)

    return texture_id

def update_texture(texture_id: int, image: np.ndarray):
    """Update existing texture with new frame."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb)
```

### Basic imgui Loop
```python
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

def main():
    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(1280, 720, "Face Recognition", None, None)
    glfw.make_context_current(window)

    # Initialize imgui
    imgui.create_context()
    impl = GlfwRenderer(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        # Your UI code here
        imgui.begin("Video")
        imgui.image(texture_id, width, height)
        imgui.end()

        imgui.render()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()
```

### Drawing Overlays on Video
```python
def draw_face_overlay(draw_list, face, offset_x, offset_y, scale=1.0):
    """Draw face box using imgui draw list."""
    x1, y1, x2, y2 = face.bbox

    # Scale and offset for window position
    x1 = offset_x + x1 * scale
    y1 = offset_y + y1 * scale
    x2 = offset_x + x2 * scale
    y2 = offset_y + y2 * scale

    # Draw rectangle
    color = imgui.get_color_u32_rgba(0, 1, 0, 1)  # Green
    draw_list.add_rect(x1, y1, x2, y2, color, thickness=2)
```

## UI Layout Mockup

```
┌─────────────────────────────────────────────────────────────────────┐
│  Face Recognition                                    [_][□][X]      │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────┐ ┌─────────────────┐ │
│ │                                             │ │   Controls      │ │
│ │                                             │ ├─────────────────┤ │
│ │           Video Feed                        │ │ Mode: RECOGNITION│
│ │                                             │ │ FPS: 28.5       │ │
│ │    ┌──────────┐                             │ │ Known: 3 people │ │
│ │    │  Jon     │                             │ │                 │ │
│ │    │  (72%)   │                             │ │ Confidence: 0.5 │ │
│ │    └──────────┘                             │ │ [────●────────] │ │
│ │                                             │ │                 │ │
│ │         ┌──────────┐                        │ │ Similarity: 0.4 │ │
│ │         │ Unknown  │                        │ │ [──────●──────] │ │
│ │         └──────────┘                        │ │                 │ │
│ │                                             │ │[Capture Mode (C)]│
│ │                                             │ │[Reload DB    (R)]│
│ │                                             │ │[Quit         (Q)]│
│ └─────────────────────────────────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ [RECOGNITION] Keyboard: Q=quit  C=capture  R=reload                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Capture Mode Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Face Recognition - CAPTURE MODE                     [_][□][X]      │
├─────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────────────────────────┐ ┌───────┐ ┌─────────────┐ │
│ │                                       │ │┌─────┐│ │  Capture    │ │
│ │           Video Feed                  │ ││ #1  ││ ├─────────────┤ │
│ │                                       │ │└─────┘│ │ Captured: 3 │ │
│ │    ┌──────────────┐                   │ │┌─────┐│ │             │ │
│ │    │   CAPTURE    │ ← Yellow box      │ ││ #2  ││ │[Capture(SPC)]│
│ │    │   (target)   │                   │ │└─────┘│ │             │ │
│ │    └──────────────┘                   │ │┌─────┐│ │ Name:       │ │
│ │                                       │ ││ #3  ││ │ [Alice____] │ │
│ │                                       │ │└─────┘│ │             │ │
│ │                                       │ │       │ │[Save    (S)]│ │
│ │                                       │ │       │ │[Cancel(ESC)]│ │
│ └───────────────────────────────────────┘ └───────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ [CAPTURE] Keyboard: SPACE=capture  S=save  ESC=cancel               │
└─────────────────────────────────────────────────────────────────────┘
```

## Testing Checklist

### Video & Detection
- [ ] Window opens and displays video
- [ ] Face detection boxes appear correctly
- [ ] Recognition labels show names and percentages
- [ ] Unknown faces show red boxes
- [ ] FPS counter updates
- [ ] Smooth performance (30+ FPS)

### Controls & Sliders
- [ ] Confidence slider adjusts detection threshold in real-time
- [ ] Similarity slider adjusts recognition threshold in real-time

### Buttons (with keyboard shortcut hints)
- [ ] [Capture Mode (C)] button enters capture mode
- [ ] [Reload DB (R)] button reloads known faces
- [ ] [Quit (Q)] button exits application
- [ ] [Capture (SPACE)] button captures central face
- [ ] [Save (S)] button saves captures with entered name
- [ ] [Cancel (ESC)] button discards and exits capture mode

### Keyboard Shortcuts (same behavior as buttons)
- [ ] Q key quits
- [ ] C key enters capture mode
- [ ] R key reloads database
- [ ] SPACE key captures face
- [ ] S key saves captures
- [ ] ESC key cancels/exits

### Capture Mode UI
- [ ] Thumbnails appear in sidebar
- [ ] Name text input field works
- [ ] Capture count updates

### Window Management
- [ ] Window resizes properly
- [ ] No memory leaks (texture cleanup)
- [ ] F11 toggles fullscreen

## Fallback Plan

Keep `--legacy` flag to use old cv2.imshow() interface:
```bash
python -m detect.main --legacy  # Uses OpenCV display
python -m detect.main           # Uses imgui (default)
```

## Timeline Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| Phase 1 | Basic window + video | Core foundation |
| Phase 2 | Face overlays | Drawing system |
| Phase 3 | Control panel | Interactive controls |
| Phase 4 | Capture mode | Complex UI state |
| Phase 5 | Keyboard shortcuts | Input handling |
| Phase 6 | Polish | Final integration |

## Dependencies to Add

```
# requirements.txt additions
imgui[glfw]>=2.0.0
PyOpenGL>=3.1.0
```
