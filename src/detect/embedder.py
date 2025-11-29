"""Face embedding module using ArcFace model."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .trt_backend import TensorRTBackend, is_tensorrt_available


class FaceEmbedder:
    """ArcFace-based face embedder for face recognition."""

    INPUT_SIZE = (112, 112)  # ArcFace input size
    EMBEDDING_DIM = 512

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
    ):
        """Initialize face embedder.

        Args:
            model_path: Path to ArcFace ONNX model
            use_gpu: Use TensorRT GPU acceleration if available
        """
        self.model_path = model_path

        # Try to use TensorRT for GPU acceleration
        self.use_tensorrt = False
        self.trt_backend = None
        self.session = None

        if use_gpu and is_tensorrt_available():
            try:
                self.trt_backend = TensorRTBackend(
                    model_path,
                    input_size=self.INPUT_SIZE,
                    fp16=True
                )
                self.use_tensorrt = True
                print("FaceEmbedder: Using TensorRT GPU acceleration")
            except Exception as e:
                print(f"FaceEmbedder: TensorRT initialization failed: {e}")
                print("FaceEmbedder: Falling back to CPU inference")

        if not self.use_tensorrt:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print("FaceEmbedder: Using ONNX Runtime CPU inference")

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for embedding model.

        Args:
            face_image: BGR face image (any size)

        Returns:
            Preprocessed tensor (1, 3, 112, 112)
        """
        # Resize to 112x112
        resized = cv2.resize(face_image, self.INPUT_SIZE)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5

        # HWC -> CHW -> NCHW
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Compute face embedding.

        Args:
            face_image: BGR face image (any size, will be resized)

        Returns:
            512-dimensional normalized embedding vector
        """
        # Preprocess
        blob = self._preprocess(face_image)

        # Run inference
        if self.use_tensorrt:
            outputs = self.trt_backend.infer(blob)
            embedding = outputs[0].flatten()
        else:
            outputs = self.session.run(None, {self.input_name: blob})
            embedding = outputs[0].flatten()

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding (normalized)
            emb2: Second embedding (normalized)

        Returns:
            Cosine similarity [-1, 1], higher is more similar
        """
        return float(np.dot(emb1, emb2))

    @staticmethod
    def compute_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute Euclidean distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Euclidean distance, lower is more similar
        """
        return float(np.linalg.norm(emb1 - emb2))
