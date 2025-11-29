#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Phase 1 Setup Script ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Found: $PYTHON_VERSION"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "Found: CUDA $CUDA_VERSION"
else
    echo "WARNING: nvidia-smi not found. GPU acceleration may not work."
fi

# Create virtual environment if it doesn't exist
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "=== Creating virtual environment ==="
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
else
    echo ""
    echo "=== Virtual environment already exists ==="
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Activated virtual environment"

# Upgrade pip
echo ""
echo "=== Upgrading pip ==="
pip install --upgrade pip

# Install requirements
echo ""
echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

# Create models directory
MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

# Download face detection model from buffalo_l package
SCRFD_MODEL="$MODELS_DIR/det_10g.onnx"
if [ ! -f "$SCRFD_MODEL" ]; then
    echo ""
    echo "=== Downloading face detection model (buffalo_l package) ==="
    BUFFALO_ZIP="$MODELS_DIR/buffalo_l.zip"
    wget -O "$BUFFALO_ZIP" \
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    echo "Extracting face detection model..."
    unzip -o "$BUFFALO_ZIP" "det_10g.onnx" -d "$MODELS_DIR"
    rm -f "$BUFFALO_ZIP"
    echo "Downloaded SCRFD model to $SCRFD_MODEL"
else
    echo ""
    echo "=== SCRFD model already exists ==="
fi

# Create project structure
echo ""
echo "=== Creating project structure ==="
mkdir -p "$SCRIPT_DIR/src/detect"
mkdir -p "$SCRIPT_DIR/tests"

# Create __init__.py if it doesn't exist
touch "$SCRIPT_DIR/src/detect/__init__.py"

# Verification
echo ""
echo "=== Verifying installation ==="
python3 -c "
import sys
print(f'Python: {sys.version}')

import numpy as np
print(f'NumPy: {np.__version__}')

import cv2
print(f'OpenCV: {cv2.__version__}')

import onnxruntime as ort
print(f'ONNX Runtime: {ort.__version__}')
providers = ort.get_available_providers()
print(f'ONNX Runtime providers: {providers}')

# Check TensorRT for GPU acceleration on ARM64
try:
    import tensorrt as trt
    print(f'TensorRT: {trt.__version__}')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print('TensorRT GPU acceleration: AVAILABLE')
except ImportError:
    print('TensorRT: NOT INSTALLED')
except Exception as e:
    print(f'TensorRT: ERROR - {e}')
"

# Check model file
if [ -f "$SCRFD_MODEL" ]; then
    MODEL_SIZE=$(du -h "$SCRFD_MODEL" | cut -f1)
    echo ""
    echo "Face detection model: $SCRFD_MODEL ($MODEL_SIZE)"
else
    echo ""
    echo "WARNING: Face detection model not found at $SCRFD_MODEL"
fi

# Check webcam
echo ""
echo "=== Checking webcam availability ==="
if [ -e /dev/video0 ]; then
    echo "Webcam found: /dev/video0"
else
    echo "WARNING: No webcam found at /dev/video0"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the application (once implemented):"
echo "  python -m src.detect.main"
