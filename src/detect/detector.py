"""Face detection module using SCRFD model."""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

from .trt_backend import TensorRTBackend, is_tensorrt_available


@dataclass
class Face:
    """Detected face with bounding box and landmarks."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    landmarks: Optional[np.ndarray] = None  # 5 keypoints: (x, y) pairs

    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class FaceDetector:
    """SCRFD-based face detector using TensorRT (GPU) or ONNX Runtime (CPU)."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        use_gpu: bool = True,
    ):
        """Initialize face detector.

        Args:
            model_path: Path to SCRFD ONNX model
            input_size: Model input size (width, height)
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            use_gpu: Use TensorRT GPU acceleration if available
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_path = model_path

        # Feature map strides for SCRFD
        self.strides = [8, 16, 32]

        # Try to use TensorRT for GPU acceleration
        self.use_tensorrt = False
        self.trt_backend = None
        self.session = None

        if use_gpu and is_tensorrt_available():
            try:
                self.trt_backend = TensorRTBackend(model_path, input_size)
                self.use_tensorrt = True
                print("Using TensorRT GPU acceleration")
            except Exception as e:
                print(f"TensorRT initialization failed: {e}")
                print("Falling back to CPU inference")

        if not self.use_tensorrt:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print("Using ONNX Runtime CPU inference")

        # Generate anchors for each stride
        self._generate_anchors()

    def _generate_anchors(self) -> None:
        """Generate anchor centers for each feature map stride."""
        self.anchor_centers = {}
        for stride in self.strides:
            h = self.input_size[1] // stride
            w = self.input_size[0] // stride
            # Create grid of anchor centers
            y, x = np.mgrid[:h, :w]
            centers = np.stack([(x.flatten() + 0.5) * stride,
                               (y.flatten() + 0.5) * stride], axis=1)
            # Each position has 2 anchors
            self.anchor_centers[stride] = np.repeat(centers, 2, axis=0)

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for model input.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Tuple of (preprocessed tensor, scale, padding offset)
        """
        img_h, img_w = image.shape[:2]
        input_w, input_h = self.input_size

        # Calculate scale to fit image in input size while preserving aspect ratio
        scale = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image (letterbox)
        padded = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Convert to float and normalize
        blob = padded.astype(np.float32)
        blob = (blob - 127.5) / 128.0

        # HWC -> CHW -> NCHW
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, (pad_x, pad_y)

    def _decode_outputs(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pad: Tuple[int, int]
    ) -> List[Face]:
        """Decode model outputs to face detections.

        Args:
            outputs: Raw model outputs
            scale: Preprocessing scale factor
            pad: Preprocessing padding offset (pad_x, pad_y)

        Returns:
            List of Face detections
        """
        # SCRFD outputs: 3 scales x (scores, bboxes, landmarks)
        # Order: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
        scores_list = outputs[0:3]
        bboxes_list = outputs[3:6]
        kps_list = outputs[6:9]

        all_scores = []
        all_bboxes = []
        all_kps = []

        for idx, stride in enumerate(self.strides):
            scores = scores_list[idx].flatten()
            bboxes = bboxes_list[idx].reshape(-1, 4)
            kps = kps_list[idx].reshape(-1, 10)
            anchors = self.anchor_centers[stride]

            # Filter by confidence
            mask = scores > self.conf_threshold
            if not np.any(mask):
                continue

            scores = scores[mask]
            bboxes = bboxes[mask]
            kps = kps[mask]
            anchors = anchors[mask]

            # Decode bboxes: distance from anchor to edges
            bboxes = self._decode_bboxes(anchors, bboxes, stride)

            # Decode keypoints
            kps = self._decode_keypoints(anchors, kps, stride)

            all_scores.extend(scores)
            all_bboxes.extend(bboxes)
            all_kps.extend(kps)

        if len(all_scores) == 0:
            return []

        all_scores = np.array(all_scores)
        all_bboxes = np.array(all_bboxes)
        all_kps = np.array(all_kps)

        # Apply NMS
        keep = self._nms(all_bboxes, all_scores)

        faces = []
        pad_x, pad_y = pad
        for i in keep:
            # Transform bbox back to original image coordinates
            bbox = all_bboxes[i].copy()
            bbox[0] = (bbox[0] - pad_x) / scale
            bbox[1] = (bbox[1] - pad_y) / scale
            bbox[2] = (bbox[2] - pad_x) / scale
            bbox[3] = (bbox[3] - pad_y) / scale

            # Grow bounding box by 20% height, 30% width
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox[0] -= w * 0.15
            bbox[1] -= h * 0.1
            bbox[2] += w * 0.15
            bbox[3] += h * 0.1

            # Transform keypoints
            kps = all_kps[i].copy().reshape(-1, 2)
            kps[:, 0] = (kps[:, 0] - pad_x) / scale
            kps[:, 1] = (kps[:, 1] - pad_y) / scale

            faces.append(Face(
                bbox=tuple(map(int, bbox)),
                score=float(all_scores[i]),
                landmarks=kps
            ))

        return faces

    def _decode_bboxes(
        self,
        anchors: np.ndarray,
        bboxes: np.ndarray,
        stride: int
    ) -> np.ndarray:
        """Decode bbox predictions from anchor-relative format."""
        # bboxes are distances: [left, top, right, bottom] * stride
        bboxes = bboxes * stride
        x1 = anchors[:, 0] - bboxes[:, 0]
        y1 = anchors[:, 1] - bboxes[:, 1]
        x2 = anchors[:, 0] + bboxes[:, 2]
        y2 = anchors[:, 1] + bboxes[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    def _decode_keypoints(
        self,
        anchors: np.ndarray,
        kps: np.ndarray,
        stride: int
    ) -> np.ndarray:
        """Decode keypoint predictions from anchor-relative format."""
        kps = kps * stride
        kps[:, 0::2] += anchors[:, 0:1]  # x coordinates
        kps[:, 1::2] += anchors[:, 1:2]  # y coordinates
        return kps

    def _nms(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray
    ) -> List[int]:
        """Non-maximum suppression."""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU below threshold
            mask = iou <= self.nms_threshold
            order = order[1:][mask]

        return keep

    def detect(self, image: np.ndarray) -> List[Face]:
        """Detect faces in image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            List of detected faces
        """
        # Preprocess
        blob, scale, pad = self._preprocess(image)

        # Run inference
        if self.use_tensorrt:
            outputs = self.trt_backend.infer(blob)
        else:
            outputs = self.session.run(None, {self.input_name: blob})

        # Decode outputs
        faces = self._decode_outputs(outputs, scale, pad)

        return faces


# Import cv2 here to avoid circular imports
import cv2
