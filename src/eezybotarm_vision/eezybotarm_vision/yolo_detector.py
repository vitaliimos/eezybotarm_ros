"""Object detector for white and green cubes using TensorRT or ONNX."""

import ctypes
from dataclasses import dataclass
import os

import cv2
import numpy as np


@dataclass
class CubeDetection:
    """Represents a detected cube."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    angle: float = 0.0


class TrtRunner:
    """TensorRT inference runner for NVIDIA Jetson."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine from {engine_path}')

        self.context = self.engine.create_execution_context()
        self.cudart = self._load_cudart()

        self.input_name = 'images'
        self.output_name = 'output0'
        if hasattr(self.engine, 'num_io_tensors'):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    self.output_name = name

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        in_bytes = int(np.prod(self.input_shape) * 4)
        out_bytes = int(np.prod(self.output_shape) * 4)

        self.d_input = self._cuda_malloc(in_bytes)
        self.d_output = self._cuda_malloc(out_bytes)
        self.stream = self._cuda_stream_create()

    def _load_cudart(self):
        import glob

        for lib in ['libcudart.so', 'libcudart.so.13', 'libcudart.so.12']:
            try:
                return ctypes.CDLL(lib)
            except OSError:
                pass

        for path in glob.glob(
            '/usr/local/lib/python3*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*'
        ):
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass

        raise RuntimeError('libcudart.so not found on system.')

    def _cuda_malloc(self, size_bytes: int):
        ptr = ctypes.c_void_p()
        ret = self.cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
        if ret != 0:
            raise RuntimeError(f'cudaMalloc failed with code {ret}')
        return ptr

    def _cuda_stream_create(self):
        stream = ctypes.c_void_p()
        ret = self.cudart.cudaStreamCreate(ctypes.byref(stream))
        if ret != 0:
            raise RuntimeError(f'cudaStreamCreate failed with code {ret}')
        return stream

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        """Run TensorRT inference."""
        self.cudart.cudaMemcpyAsync(
            self.d_input,
            input_blob.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(input_blob.nbytes),
            ctypes.c_int(1),
            self.stream,
        )

        self.context.set_input_shape(self.input_name, input_blob.shape)
        self.context.set_tensor_address(self.input_name, self.d_input.value)
        self.context.set_tensor_address(self.output_name, self.d_output.value)
        self.context.execute_async_v3(self.stream.value)

        out_buf = np.empty(self.output_shape, dtype=np.float32)
        self.cudart.cudaMemcpyAsync(
            out_buf.ctypes.data_as(ctypes.c_void_p),
            self.d_output,
            ctypes.c_size_t(out_buf.nbytes),
            ctypes.c_int(2),
            self.stream,
        )
        self.cudart.cudaStreamSynchronize(self.stream)
        return out_buf

    def __del__(self):
        if hasattr(self, 'cudart'):
            if getattr(self, 'd_input', None):
                self.cudart.cudaFree(self.d_input)
            if getattr(self, 'd_output', None):
                self.cudart.cudaFree(self.d_output)
            if getattr(self, 'stream', None):
                self.cudart.cudaStreamDestroy(self.stream)


class OnnxRunner:
    """ONNX Runtime inference runner for workstation development and testing."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: input_blob})[0]


class YoloCubeDetector:
    """Detector for green_cube and white_cube using TensorRT or ONNX."""

    CLASS_NAMES = {0: 'green_cube', 1: 'white_cube'}

    def __init__(self, weights_path: str = '', conf_threshold: float = 0.5):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self._runner = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.weights_path:
            raise RuntimeError('YOLO weights_path parameter is empty.')

        path = self.weights_path
        # Allow automatic fallback to counterpart .onnx if .engine does not exist in same location
        if not os.path.exists(path) and path.endswith('.engine'):
            onnx_path = path[:-7] + '.onnx'
            if os.path.exists(onnx_path):
                path = onnx_path

        if not os.path.exists(path):
            raise RuntimeError(f"YOLO model file not found: '{self.weights_path}'")

        if path.endswith('.engine'):
            try:
                self._runner = TrtRunner(path)
                return
            except Exception:
                onnx_path = path[:-7] + '.onnx'
                if os.path.exists(onnx_path):
                    self._runner = OnnxRunner(onnx_path)
                    return
                raise

        self._runner = OnnxRunner(path)

    def is_loaded(self) -> bool:
        return self._runner is not None

    def predict(self, image: np.ndarray) -> list[CubeDetection]:
        """Detect cubes in image and return list of CubeDetection."""
        if not self.is_loaded():
            raise RuntimeError('Cannot predict: model is not loaded.')

        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]
        blob = cv2.resize(image, (640, 640))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.expand_dims(blob, axis=0))

        out = self._runner.infer(blob)
        return self._postprocess(out, orig_w, orig_h)

    def _postprocess(
        self,
        out: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> list[CubeDetection]:
        if out is None or len(out) == 0:
            return []

        preds = out[0]
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        detections: list[CubeDetection] = []
        for row in preds:
            if len(row) < 6:
                continue

            conf = float(row[4])
            if conf < self.conf_threshold:
                continue

            cx = float(row[0] * scale_x)
            cy = float(row[1] * scale_y)
            w = float(row[2] * scale_x)
            h = float(row[3] * scale_y)
            cls_id = int(row[5])
            angle = float(row[6]) if len(row) >= 7 else 0.0

            detections.append(
                CubeDetection(
                    class_id=cls_id,
                    class_name=self.CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                    confidence=conf,
                    bbox_xyxy=(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
                    center_px=(cx, cy),
                    angle=angle,
                )
            )

        return detections
