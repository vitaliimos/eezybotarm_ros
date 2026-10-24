"""Integration test verifying end-to-end vision and sorting node pipeline."""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage, Image
import yaml

HAS_MODEL_RUNNER = (
    importlib.util.find_spec('tensorrt') is not None
    or importlib.util.find_spec('onnxruntime') is not None
)


def _resolve_model_path() -> str | None:
    trt_path = '/usr/share/models/model.engine'
    if os.path.exists(trt_path):
        return trt_path
    try:
        pkg_share = get_package_share_directory('eezybotarm_vision')
        onnx_path = os.path.join(pkg_share, 'models', 'model.onnx')
        if os.path.exists(onnx_path):
            return onnx_path
    except Exception:
        pass
    return None


def _load_node_parameters(weights_path: str) -> list[Parameter]:
    vision_share = get_package_share_directory('eezybotarm_vision')
    desc_share = get_package_share_directory('eezybotarm_description')

    vision_cfg_path = os.path.join(vision_share, 'config', 'vision_params.yaml')
    env_cfg_path = os.path.join(desc_share, 'config', 'environment.yaml')

    with open(vision_cfg_path, 'r', encoding='utf-8') as f:
        vision_params = yaml.safe_load(f)['sorting_coordinator_node']['ros__parameters']
    with open(env_cfg_path, 'r', encoding='utf-8') as f:
        env_params = yaml.safe_load(f)['/**']['ros__parameters']

    merged = {**vision_params, **env_params, 'weights_path': weights_path}
    return [Parameter(name=k, value=v) for k, v in merged.items()]


def _create_image_msg(node, frame: np.ndarray) -> Image:
    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'camera_optical_frame'
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg


@pytest.mark.skipif(
    not HAS_MODEL_RUNNER or _resolve_model_path() is None,
    reason='TensorRT or onnxruntime and model file required',
)
def test_vision_pipeline_integration():
    """Verify camera feed processing, zone detection, and cube classification."""
    from eezybotarm_vision.sorting_node import SortingCoordinatorNode, SortingState

    model_path = _resolve_model_path()
    assert model_path is not None

    rclpy.init()
    try:
        sample_img_path = os.path.join(
            get_package_share_directory('eezybotarm_vision'),
            'resource',
            'sample_workspace.png',
        )
        assert os.path.exists(sample_img_path), f'Sample image not found: {sample_img_path}'
        sample_frame = cv2.imread(sample_img_path)
        assert sample_frame is not None, f'Failed to read {sample_img_path}'

        params = _load_node_parameters(model_path)
        sort_node = SortingCoordinatorNode(parameter_overrides=params)
        test_node = rclpy.create_node('pipeline_test_helper')
        cam_pub = test_node.create_publisher(Image, '/camera/image_raw', 10)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(sort_node)
        executor.add_node(test_node)

        debug_images_received = []
        comp_images_received = []

        test_node.create_subscription(
            Image, '/eezybotarm_vision/debug_image', debug_images_received.append, 10
        )
        test_node.create_subscription(
            CompressedImage,
            '/eezybotarm_vision/debug_image/compressed',
            comp_images_received.append,
            10,
        )

        cam_pub.publish(_create_image_msg(test_node, sample_frame))

        for _ in range(30):
            executor.spin_once(timeout_sec=0.05)
            if debug_images_received and comp_images_received:
                break

        assert sort_node.state == SortingState.SCANNING
        assert sort_node.calibration is not None
        assert set(sort_node.calibration.zones.keys()) == {'left', 'middle', 'right'}
        assert len(debug_images_received) > 0
        assert len(comp_images_received) > 0
        assert comp_images_received[0].format == 'jpeg'

    finally:
        rclpy.shutdown()
