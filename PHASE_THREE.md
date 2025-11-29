# Phase 3 Implementation Plan

## Overview

Add face recognition capabilities:
- Compute facial embeddings for known faces in `known-faces/`
- Match detected faces against known embeddings in real-time
- Display names over recognized faces
- Save unknown faces to `new-faces/<timestamp>/`

## Requirements (from CLAUDE.md)

10. Calculate a facial embedding for each subdirectory of images in `known-faces/`
11. For each detected face:
    - Normalize to 384px (same as capture)
    - Calculate embedding
    - Search for matching known face
    - If match found: display `<name>` over face
12. For unrecognized faces:
    - Save to `new-faces/YYYY-MM-DD-HH-MM/`
    - Capture at most once per second

## Face Embedding Model

**Selected: ArcFace (w600k_r50.onnx)**
- Part of InsightFace buffalo_l package (already downloaded)
- 512-dimensional embeddings
- State-of-the-art face recognition accuracy

## Implementation Components

### 1. Face Embedder Module (`embedder.py`)

```python
class FaceEmbedder:
    def __init__(model_path)
    def get_embedding(face_image) -> np.ndarray  # 512-dim vector
    def compute_similarity(emb1, emb2) -> float  # cosine similarity
```

### 2. Known Faces Database (`face_database.py`)

```python
class FaceDatabase:
    def __init__(known_faces_dir)
    def load_known_faces()  # Load and compute embeddings
    def find_match(embedding) -> (name, similarity)
    def add_person(name, embeddings)
```

### 3. Unknown Face Tracker

- Track last save time per face region
- Save at most once per second
- Directory format: `new-faces/2025-11-28-15-30/`

## Data Flow

```
Camera Frame
     │
     ▼
Face Detection (SCRFD)
     │
     ▼
For each face:
  ├─► Crop & Normalize (384px)
  │        │
  │        ▼
  │   Compute Embedding (ArcFace)
  │        │
  │        ▼
  │   Search Known Faces Database
  │        │
  │        ├─► Match Found → Display Name
  │        │
  │        └─► No Match → Save to new-faces/ (rate limited)
  │
  ▼
Display Frame
```

## Matching Algorithm

1. Compute cosine similarity between detected face and all known embeddings
2. For each known person, use average of their embeddings
3. Match threshold: 0.4 (cosine similarity, tunable)
4. If best match > threshold, display that person's name

## File Structure

```
known-faces/
├── alice/
│   ├── 001.jpg
│   └── 002.jpg
├── bob/
│   └── 001.jpg

new-faces/
├── 2025-11-28-15-30/
│   ├── 001.jpg
│   └── 002.jpg
├── 2025-11-28-15-31/
│   └── 001.jpg
```

## Display Updates

- Recognized face: Green box with name label
- Unknown face: Red box with "Unknown" label
- Status bar shows: `[RECOGNITION] C=capture  R=reload database  Q=quit`

## Implementation Steps

1. Extract ArcFace model from buffalo_l.zip
2. Create `embedder.py` with TensorRT support
3. Create `face_database.py` for known faces
4. Update `main.py` with recognition mode
5. Update `display.py` for name labels
6. Add unknown face saving with rate limiting
7. Add 'R' key to reload database (after adding new people)
