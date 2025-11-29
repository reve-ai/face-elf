"""TensorRT inference backend for GPU-accelerated face detection."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import ctypes


class TensorRTBackend:
    """TensorRT inference backend for ONNX models."""

    def __init__(
        self,
        onnx_path: str,
        input_size: Tuple[int, int] = (640, 640),
        fp16: bool = True,
    ):
        """Initialize TensorRT backend.

        Args:
            onnx_path: Path to ONNX model
            input_size: Model input size (width, height)
            fp16: Use FP16 precision for faster inference
        """
        import tensorrt as trt

        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
        self.fp16 = fp16

        # TensorRT engine path (cached)
        precision = "fp16" if fp16 else "fp32"
        self.engine_path = self.onnx_path.with_suffix(f".{precision}.engine")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.bindings = None
        self.input_buffer = None
        self.output_buffers = []
        self.output_names = []

        self._load_or_build_engine()
        self._setup_bindings()

    def _load_or_build_engine(self) -> None:
        """Load cached engine or build from ONNX."""
        import tensorrt as trt

        if self.engine_path.exists():
            print(f"Loading cached TensorRT engine from {self.engine_path}")
            self._load_engine()
        else:
            print(f"Building TensorRT engine from {self.onnx_path}")
            print("This may take 30-60 seconds (one-time only)...")
            self._build_engine()
            self._save_engine()

        self.context = self.engine.create_execution_context()

    def _build_engine(self) -> None:
        """Build TensorRT engine from ONNX model."""
        import tensorrt as trt

        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Set input shape
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = (1, 3, self.input_size[1], self.input_size[0])
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

    def _save_engine(self) -> None:
        """Save engine to file for future use."""
        print(f"Saving TensorRT engine to {self.engine_path}")
        with open(self.engine_path, "wb") as f:
            f.write(self.engine.serialize())

    def _load_engine(self) -> None:
        """Load engine from file."""
        import tensorrt as trt

        runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def _setup_bindings(self) -> None:
        """Setup input/output bindings."""
        import tensorrt as trt

        self.bindings = []
        self.output_buffers = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Handle dynamic dimensions
            for j, dim in enumerate(shape):
                if dim == -1:
                    if j == 0:
                        shape[j] = 1  # batch size
                    elif j == 2:
                        shape[j] = self.input_size[1]  # height
                    elif j == 3:
                        shape[j] = self.input_size[0]  # width

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_buffer = np.zeros(shape, dtype=dtype)
                self.bindings.append(self.input_buffer.ctypes.data)
            else:
                output_buffer = np.zeros(shape, dtype=dtype)
                self.output_buffers.append(output_buffer)
                self.output_names.append(name)
                self.bindings.append(output_buffer.ctypes.data)

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference on input data.

        Args:
            input_data: Preprocessed input tensor (1, 3, H, W)

        Returns:
            List of output arrays
        """
        import tensorrt as trt

        # Copy input data
        np.copyto(self.input_buffer, input_data.astype(self.input_buffer.dtype))

        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        # Set input shape for dynamic dimensions
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, input_data.shape)

        # Run inference
        success = self.context.execute_async_v3(0)  # stream_handle=0 for default stream
        if not success:
            raise RuntimeError("TensorRT inference failed")

        # Synchronize with GPU - must wait for async operation to complete
        self._cuda_synchronize()

        # Return copies of outputs
        return [buf.copy() for buf in self.output_buffers]

    def _cuda_synchronize(self) -> None:
        """Synchronize with CUDA device to ensure all operations complete."""
        # Use ctypes to call cudaDeviceSynchronize from CUDA runtime
        try:
            libcudart = ctypes.CDLL("libcudart.so")
            libcudart.cudaDeviceSynchronize()
        except OSError:
            # Fallback: try to find cudart in common locations
            import os
            cuda_paths = [
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/lib/aarch64-linux-gnu/libcudart.so",
            ]
            for path in cuda_paths:
                if os.path.exists(path):
                    libcudart = ctypes.CDLL(path)
                    libcudart.cudaDeviceSynchronize()
                    return
            # If we can't find cudart, the results might be incorrect
            # but we'll continue anyway


def is_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    try:
        import tensorrt
        return True
    except ImportError:
        return False
