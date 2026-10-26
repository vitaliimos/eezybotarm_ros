#!/usr/bin/env python3
"""Compile a YOLO ONNX model into a TensorRT .engine plan using the Python TensorRT API."""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, 'src', 'eezybotarm_vision', 'models')
DEFAULT_ONNX = os.path.join(DEFAULT_MODELS_DIR, 'model.onnx')
DEFAULT_ENGINE = os.path.join(DEFAULT_MODELS_DIR, 'model.engine')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compile ONNX model to TensorRT .engine plan.'
    )
    parser.add_argument(
        '--onnx',
        '-i',
        default=DEFAULT_ONNX,
        help='Input ONNX model file (default: src/eezybotarm_vision/models/model.onnx)',
    )
    parser.add_argument(
        '--output',
        '-o',
        default=DEFAULT_ENGINE,
        help='Output TensorRT .engine path (default: src/eezybotarm_vision/models/model.engine)',
    )
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Disable FP16 precision and build a full FP32 engine (default: FP16 enabled)',
    )
    return parser.parse_args()


def build_engine(onnx_path: str, output_path: str, use_fp16: bool = True) -> None:
    if not os.path.isfile(onnx_path):
        print(f"Error: ONNX file not found at '{onnx_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        import tensorrt as trt
    except ImportError:
        print(
            "Error: 'tensorrt' Python module is not installed.\n"
            'Run inside the dev container: ./docker-run.sh python3 scripts/build_trt_engine.py',
            file=sys.stderr,
        )
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    print(f'[TensorRT {trt.__version__}] Parsing ONNX model: {onnx_path}')
    if not parser.parse_from_file(os.path.abspath(onnx_path)):
        for i in range(parser.num_errors):
            print(f'ONNX Parser Error: {parser.get_error(i)}', file=sys.stderr)
        sys.exit(1)

    config = builder.create_builder_config()
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        precision_str = 'FP16'
    else:
        precision_str = 'FP32'

    abs_out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    cache_path = f'{os.path.splitext(abs_out)[0]}.timing.cache'
    cache_data = b''
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            cache_data = f.read()
        print(f'[TensorRT] Loaded timing cache: {cache_path}')
    timing_cache = config.create_timing_cache(cache_data)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    print(f'[TensorRT] Building {precision_str} engine -> {output_path} (may take 1-5 min)...')
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print('Error: Failed to build serialized TensorRT engine.', file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    with open(f'{cache_path}.tmp', 'wb') as f:
        f.write(timing_cache.serialize())
    os.replace(f'{cache_path}.tmp', cache_path)

    tmp_out = f'{abs_out}.tmp'
    with open(tmp_out, 'wb') as f:
        f.write(serialized_engine)
    os.replace(tmp_out, abs_out)

    size_mb = os.path.getsize(abs_out) / (1024 * 1024)
    print(f'[TensorRT] Engine compiled in {elapsed:.1f}s ({size_mb:.2f} MB): {output_path}')


def main():
    args = parse_args()
    build_engine(args.onnx, args.output, use_fp16=not args.fp32)


if __name__ == '__main__':
    main()
