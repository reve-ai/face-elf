# Phase 2 Implementation Plan

## Overview

Add a face capture mode that allows:
- Detecting the central face in the image
- Capturing multiple snapshots of that face
- Displaying captured snapshots in a side panel
- Saving captured faces to `known-faces/<name>/`

## User Interface

### Keyboard Controls

| Key | Action |
|-----|--------|
| `c` | Enter capture mode |
| `SPACE` | Capture current central face (in capture mode) |
| `s` | Save captured faces (prompts for name) |
| `ESC` | Exit capture mode (discard) / Quit app |
| `q` | Quit application |

### Display Layout

```
+---------------------------+--------+
|                           |  [1]   |
|                           |--------|
|    Main Camera View       |  [2]   |
|    (with face boxes)      |--------|
|                           |  [3]   |
|                           |--------|
|    [CAPTURE MODE]         |  ...   |
+---------------------------+--------+
     640px main              128px sidebar
```

- Main view: Live camera with face detection
- Sidebar: Captured face thumbnails (128x128 display size)
- Status indicator shows current mode

## Implementation Components

### 1. Application State

```python
class AppState(Enum):
    DETECTION = "detection"  # Normal face detection mode
    CAPTURE = "capture"      # Face capture mode
```

### 2. Face Processing

- **Central face detection**: Find face closest to image center
- **Crop and scale**: Extract face region, scale to 384px on longest axis
- **Thumbnail generation**: Scale to 128x128 for sidebar display

### 3. Data Structures

```python
@dataclass
class CapturedFace:
    image: np.ndarray      # Full 384px normalized image
    thumbnail: np.ndarray  # 128x128 display thumbnail
    timestamp: float       # Capture time
```

### 4. File Structure

```
known-faces/
├── alice/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
├── bob/
│   ├── 001.jpg
│   └── 002.jpg
```

## Implementation Steps

1. Add `AppState` enum and state management to main.py
2. Create `face_capture.py` module for capture logic
3. Add `find_central_face()` function
4. Add `crop_and_scale_face()` function (384px normalization)
5. Update display.py with sidebar rendering
6. Add keyboard handling for mode switching
7. Implement save dialog (text input via OpenCV or console)
8. Create `known-faces/` directory structure

## Verification

1. Press 'c' to enter capture mode - status indicator appears
2. Central face is highlighted differently
3. Press SPACE - face captured, thumbnail appears in sidebar
4. Capture multiple faces - thumbnails stack vertically
5. Press 's' - prompted for name, saves to `known-faces/<name>/`
6. Press ESC - exits capture mode, clears captures
7. Files saved as 384px JPEGs in correct directory
