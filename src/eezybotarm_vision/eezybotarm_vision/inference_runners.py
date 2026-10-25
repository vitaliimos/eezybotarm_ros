"""TensorRT and ONNX Runtime inference runners."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _cuda_check(result):
    """Unpack a cuda-python ``(err, *values)`` result, raising on error."""
    from cuda.bindings import runtime as cudart

    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f'CUDA error: {err}')
    return values[0] if values else None


class TrtRunner:
    """TensorRT inference runner (execute_async_v3 + cuda-python) for NVIDIA Jetson."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._cudart = cudart
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine from {engine_path}')
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.output_name = next(n for n in names if n != self.input_name)

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        input_bytes = int(
            np.prod(self.engine.get_tensor_shape(self.input_name))
            * np.dtype(self.input_dtype).itemsize
        )
        self._out = np.empty(
            self.engine.get_tensor_shape(self.output_name),
            dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_name)),
        )

        self._d_input = _cuda_check(cudart.cudaMalloc(input_bytes))
        self._d_output = _cuda_check(cudart.cudaMalloc(self._out.nbytes))
        self._stream = _cuda_check(cudart.cudaStreamCreate())
        self.context.set_tensor_address(self.input_name, int(self._d_input))
        self.context.set_tensor_address(self.output_name, int(self._d_output))

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        cudart = self._cudart
        blob = np.ascontiguousarray(input_blob, dtype=self.input_dtype)
        _cuda_check(cudart.cudaMemcpyAsync(
            self._d_input, blob.ctypes.data, blob.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream))
        if not self.context.execute_async_v3(int(self._stream)):
            raise RuntimeError('TensorRT execute_async_v3 failed.')
        _cuda_check(cudart.cudaMemcpyAsync(
            self._out.ctypes.data, self._d_output, self._out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream))
        _cuda_check(cudart.cudaStreamSynchronize(self._stream))
        return self._out


class OnnxRunner:
    """ONNX Runtime inference runner for workstation development and testing."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = [
            p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
            if p in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        logger.info('Initialized ONNX Runtime session with providers: %s',
                    self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: input_blob})[0]


def create_inference_runner(weights_path: str) -> TrtRunner | OnnxRunner:
    """Load TensorRT (.engine) or ONNX (.onnx) runner from weights path."""
    if not os.path.exists(weights_path):
        raise RuntimeError(f"YOLO model file not found: '{weights_path}'")
    if weights_path.endswith('.engine'):
        return TrtRunner(weights_path)
    return OnnxRunner(weights_path)
