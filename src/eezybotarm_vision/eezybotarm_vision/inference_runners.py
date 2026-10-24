"""TensorRT and ONNX Runtime inference runners."""

import ctypes
import glob
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _load_cudart():
        for lib in ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12'):
            try:
                return ctypes.CDLL(lib)
            except OSError:
                pass

        for path in glob.glob(
            '/usr/local/lib/python3*/dist-packages/nvidia/*/lib/libcudart.so*'
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

        self.onnx_path = onnx_path
        self._ort = ort

        self._sess_options = ort.SessionOptions()
        # Disable extended/contrib kernel fusions (e.g. QuickGeluFusion) that may
        # lack compiled SASS/PTX for Jetson SM 8.7 in generic aarch64 wheels.
        self._sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        providers = [
            p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
            if p in ort.get_available_providers()
        ]

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=self._sess_options,
            providers=providers,
        )
        logger.info('Initialized ONNX Runtime session with providers: %s',
                    self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        try:
            return self.session.run(None, {self.input_name: input_blob})[0]
        except Exception as exc:
            if 'CUDAExecutionProvider' in self.session.get_providers():
                logger.warning(
                    'CUDAExecutionProvider failed during inference (%s); '
                    'falling back to CPUExecutionProvider.',
                    exc,
                )
                self.session = self._ort.InferenceSession(
                    self.onnx_path,
                    sess_options=self._sess_options,
                    providers=['CPUExecutionProvider'],
                )
                self.input_name = self.session.get_inputs()[0].name
                return self.session.run(None, {self.input_name: input_blob})[0]
            raise


def create_inference_runner(weights_path: str) -> TrtRunner | OnnxRunner:
    """Load TensorRT (.engine) or ONNX (.onnx) runner from weights path."""
    if not weights_path:
        raise RuntimeError('YOLO weights_path parameter is empty.')

    path = weights_path
    if not os.path.exists(path) and path.endswith('.engine'):
        onnx_path = path[:-7] + '.onnx'
        if os.path.exists(onnx_path):
            path = onnx_path

    if not os.path.exists(path):
        raise RuntimeError(f"YOLO model file not found: '{weights_path}'")

    if path.endswith('.engine'):
        try:
            return TrtRunner(path)
        except Exception as exc:
            onnx_path = path[:-7] + '.onnx'
            if os.path.exists(onnx_path):
                logger.warning(
                    "Failed to load TensorRT engine '%s': %s. Falling back to ONNX: '%s'",
                    path,
                    exc,
                    onnx_path,
                )
                return OnnxRunner(onnx_path)
            raise

    return OnnxRunner(path)
