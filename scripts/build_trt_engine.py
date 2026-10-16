#!/usr/bin/env python3
"""Build a TensorRT .engine plan from a YOLO ONNX model."""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='Compile ONNX model to TensorRT engine.')
    parser.add_argument(
        '--onnx',
        '-i',
        default='src/eezybotarm_vision/models/model.onnx',
        help='Input ONNX model file (default: src/eezybotarm_vision/models/model.onnx)',
    )
    parser.add_argument(
        '--output',
        '-o',
        default='src/eezybotarm_vision/models/model.engine',
        help='Output TensorRT .engine path (default: src/eezybotarm_vision/models/model.engine)',
    )
    parser.add_argument(
        '--input-name',
        default='images',
        help='Name of the input tensor (default: images)',
    )
    parser.add_argument(
        '--shape',
        default='1,3,640,640',
        help='Input tensor shape B,C,H,W (default: 1,3,640,640)',
    )

    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"Error: ONNX file not found at '{args.onnx}'", file=sys.stderr)
        sys.exit(1)

    try:
        import tensorrt as trt
    except ImportError:
        print(
            "Error: 'tensorrt' Python module not installed.\n"
            "Install TensorRT via: pip install tensorrt\n"
            "Or run: trtexec --onnx=" + args.onnx + " --saveEngine=" + args.output,
            file=sys.stderr,
        )
        sys.exit(1)

    print(f'[TensorRT] Building engine from: {args.onnx}')
    print(f'[TensorRT] Target output: {args.output}')

    b, c, h, w = [int(v.strip()) for v in args.shape.split(',')]
    input_shape = (b, c, h, w)
    print(f'[TensorRT] Input tensor "{args.input_name}" shape: {input_shape}')

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    else:
        network = builder.create_network()
    parser_onnx = trt.OnnxParser(network, logger)

    with open(args.onnx, 'rb') as f:
        if not parser_onnx.parse(f.read()):
            for error in range(parser_onnx.num_errors):
                print(f'Parser Error: {parser_onnx.get_error(error)}', file=sys.stderr)
            sys.exit(1)

    config = builder.create_builder_config()

    # Optimization profile for input shape
    profile = builder.create_optimization_profile()
    profile.set_shape(args.input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    print('[TensorRT] Compiling engine (this may take 1-3 minutes)...')
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print('Error: Failed to build serialized TensorRT engine.', file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f'[TensorRT] Engine successfully compiled in {elapsed:.1f}s!')
    print(f'[TensorRT] File size: {size_mb:.2f} MB saved to {args.output}')


if __name__ == '__main__':
    main()
