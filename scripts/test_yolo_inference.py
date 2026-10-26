#!/usr/bin/env python3
"""Script to test YOLO cube detector on PC using ONNX or TensorRT engine."""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Add package source to PYTHONPATH if not installed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PKG_SRC = os.path.join(REPO_ROOT, 'src', 'eezybotarm_vision')
if PKG_SRC not in sys.path:
    sys.path.insert(0, PKG_SRC)

from eezybotarm_vision.yolo_detector import YoloCubeDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Test YOLO cube detector with ONNX or TensorRT.')
    parser.add_argument(
        '--trt',
        action='store_true',
        help='Use TensorRT engine (src/eezybotarm_vision/models/model.engine) instead of ONNX',
    )
    parser.add_argument(
        '--weights',
        '-w',
        default=None,
        help='Path to .onnx or .engine file (default: model.onnx, or model.engine with --trt)',
    )
    parser.add_argument(
        '--image',
        '-i',
        default=os.path.join(PKG_SRC, 'resource', 'sample_workspace.png'),
        help='Path to input test image (default: sample_workspace.png)',
    )
    parser.add_argument(
        '--conf',
        '-c',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)',
    )
    parser.add_argument(
        '--output',
        '-o',
        default='/tmp/yolo_test_result.jpg',
        help='Output annotated image path (default: /tmp/yolo_test_result.jpg)',
    )

    args = parser.parse_args()

    weights_path = args.weights
    if weights_path is None:
        model_filename = 'model.engine' if args.trt else 'model.onnx'
        weights_path = os.path.join(PKG_SRC, 'models', model_filename)

    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.", file=sys.stderr)
        if weights_path.endswith('.engine'):
            print(
                'Compile the TensorRT engine first:\n'
                '  ./docker-run.sh python3 scripts/build_trt_engine.py',
                file=sys.stderr,
            )
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: Test image '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f'Loading model: {weights_path}')
    start_init = time.perf_counter()
    detector = YoloCubeDetector(
        weights_path=weights_path,
        conf_threshold=args.conf,
    )
    init_time_ms = (time.perf_counter() - start_init) * 1000.0
    print(f'Model loaded in {init_time_ms:.1f} ms')

    image = cv2.imread(args.image)
    if image is None:
        print(f'Failed to read image: {args.image}', file=sys.stderr)
        sys.exit(1)

    # Warmup
    _ = detector.predict(image)

    # Benchmark
    latencies = []
    num_runs = 10
    for _ in range(num_runs):
        t0 = time.perf_counter()
        detections = detector.predict(image)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = sum(latencies) / len(latencies)
    print(f'Inference latency: {avg_lat:.2f} ms (averaged over {num_runs} runs)')
    print(f'\nDetections ({len(detections)}):')
    for idx, d in enumerate(detections):
        print(
            f'  [{idx + 1}] {d.class_name:<11} | conf: {d.confidence:.2f} | '
            f'center: ({d.center_px[0]:.1f}, {d.center_px[1]:.1f}) | '
            f'bbox: [{d.bbox_xyxy[0]:.1f}, {d.bbox_xyxy[1]:.1f}, {d.bbox_xyxy[2]:.1f}, {d.bbox_xyxy[3]:.1f}] | '
            f'angle: {d.angle:.2f} rad ({d.angle * 180.0 / 3.14159:.1f} deg)'
        )

        # Annotate
        color = (0, 220, 0) if d.class_name == 'green_cube' else (255, 255, 255)
        pts = np.array(d.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        x1, y1 = int(d.bbox_xyxy[0]), int(d.bbox_xyxy[1])
        label = f'{d.class_name} {d.confidence:.2f}'
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cv2.circle(image, (int(d.center_px[0]), int(d.center_px[1])), 4, (0, 0, 255), -1)

    cv2.imwrite(args.output, image)
    print(f'\nAnnotated image saved to: {args.output}')


if __name__ == '__main__':
    main()
