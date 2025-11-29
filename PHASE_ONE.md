# Phase 1 Implementation Plan

## Project Overview

**Goal:** Create a real-time face detection application that captures webcam input, detects faces using an NVIDIA-optimized model, and displays bounding boxes around detected faces.

**Environment:**
- NVIDIA GB10 GPU with CUDA 13.0
- Python 3.12.3
- ARM64 (aarch64) architecture
- Webcams available at `/dev/video0`, `/dev/video1`

---

## Implementation Steps

### 1. Project Structure Setup

```
detect/
├── CLAUDE.md
├── PHASE_ONE.md
├── requirements.txt
├── src/
│   └── detect/
│       ├── __init__.py
│       ├── main.py          # Entry point
│       ├── camera.py        # Webcam capture module
│       ├── detector.py      # Face detection module
│       └── display.py       # Visualization module
└── tests/
    └── test_detector.py
```

### 2. Dependencies

Core packages needed:
- **opencv-python** - Webcam capture and image display
- **numpy** - Array operations
- **onnxruntime** - ONNX model inference (CPU on ARM64)
- **tensorrt** - GPU-accelerated inference (for future optimization)
- **Face detection model** - SCRFD det_10g.onnx from InsightFace buffalo_l package

**Note on ARM64:** The `onnxruntime-gpu` package doesn't have pre-built wheels for ARM64.
TensorRT is available for GPU acceleration and can be integrated for optimized inference.

### 3. Component Implementation

| Component | Responsibility | Key Functions |
|-----------|---------------|---------------|
| `camera.py` | Open webcam, read frames | `open_camera()`, `read_frame()`, `release()` |
| `detector.py` | Load model, run inference | `load_model()`, `detect_faces(frame)` -> list of bounding boxes |
| `display.py` | Draw boxes, show image | `draw_boxes(frame, boxes)`, `show_frame(frame)` |
| `main.py` | Main loop orchestration | Initialize -> capture -> detect -> display -> repeat |

### 4. Face Detection Model Selection

**Selected: SCRFD det_10g (Sample and Computation Redistribution for Face Detection)**
- Pre-trained ONNX model from InsightFace buffalo_l package
- 17MB model with excellent accuracy
- Returns bounding boxes and 5-point facial landmarks
- Input: dynamic size RGB image [1, 3, H, W]
- Output: multi-scale detection results (scores, bboxes, landmarks)

---

## Verification Plan

### Unit Tests

1. **Camera Module**
   - Verify webcam opens successfully
   - Verify frames are captured with correct dimensions
   - Verify camera releases properly

2. **Detector Module**
   - Verify model loads without error
   - Verify detection on test images with known faces returns boxes
   - Verify detection on images without faces returns empty list
   - Verify bounding box format (x, y, width, height or x1, y1, x2, y2)

3. **Display Module**
   - Verify bounding boxes are drawn correctly
   - Verify display window opens/closes properly

### Integration Tests

1. **End-to-end pipeline**
   - Capture frame -> detect -> display works in sequence
   - Verify no memory leaks over extended runtime
   - Verify GPU utilization (model runs on CUDA, not CPU)

### Performance Verification

1. **Frame rate target:** >=15 FPS for "real-time" feel (ideally 30 FPS)
2. **Latency:** Detection latency < 100ms per frame
3. **GPU memory:** Monitor with `nvidia-smi` during runtime

### Manual Verification

1. Run application with single face in frame -> verify single box
2. Run with multiple faces -> verify multiple boxes
3. Run with no faces -> verify no boxes drawn
4. Move face around -> verify boxes track smoothly
5. Test at various distances from camera

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| ARM64 lacks onnxruntime-gpu | Use CPU inference initially; TensorRT available for GPU optimization |
| CPU inference may be slow | Profile performance; consider TensorRT integration if needed |
| Model accuracy issues | Tune confidence threshold; det_10g is high-accuracy model |
| Display lag | Use separate threads for capture vs. display if needed |
| Webcam access conflicts | Check for exclusive access, handle gracefully |

---

## Estimated File Sizes

- `camera.py` - ~50 lines
- `detector.py` - ~100 lines (including model loading and preprocessing)
- `display.py` - ~40 lines
- `main.py` - ~60 lines
- `requirements.txt` - ~10 lines
