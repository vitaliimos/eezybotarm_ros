"""Integration test verifying end-to-end vision and sorting node pipeline."""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image

HAS_MODEL_RUNNER = (
    importlib.util.find_spec('tensorrt') is not None
    or importlib.util.find_spec('onnxruntime') is not None
)
HAS_MODEL_FILE = os.path.exists('/usr/share/models/model.engine')


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
    not (HAS_MODEL_RUNNER and HAS_MODEL_FILE),
    reason='TensorRT or onnxruntime and model file required',
)
def test_vision_pipeline_integration():
    """Verify camera feed processing, zone detection, and cube classification."""
    from eezybotarm_vision.sorting_node import SortingCoordinatorNode

    rclpy.init()
    try:
        sample_img_path = os.path.join(
            get_package_share_directory('eezybotarm_vision'),
            'resource',
            'sample_workspace.jpg',
        )
        assert os.path.exists(sample_img_path), f'Sample image not found: {sample_img_path}'
        sample_frame = cv2.imread(sample_img_path)
        assert sample_frame is not None, f'Failed to read {sample_img_path}'

        # Create sorting coordinator node
        sort_node = SortingCoordinatorNode()

        # Create test helper node to publish camera frames and listen to output topics
        test_node = rclpy.create_node('pipeline_test_helper')
        cam_pub = test_node.create_publisher(Image, '/camera/image_raw', 10)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(sort_node)
        executor.add_node(test_node)

        # Track messages
        debug_images_received = []

        def on_debug_img(msg):
            debug_images_received.append(msg)

        test_node.create_subscription(
            Image, '/eezybotarm_vision/debug_image', on_debug_img, 10
        )

        # Publish test image
        cam_pub.publish(_create_image_msg(test_node, sample_frame))

        # Spin executor to process subscriptions
        for _ in range(30):
            executor.spin_once(timeout_sec=0.05)
            if len(debug_images_received) > 0:
                break

        # Verify zones were calibrated
        assert sort_node.active_calibration is not None, 'Active calibration should be detected'
        assert 'left' in sort_node.active_calibration.zones
        assert 'middle' in sort_node.active_calibration.zones
        assert 'right' in sort_node.active_calibration.zones

        # Verify debug image published
        assert len(debug_images_received) > 0, 'Debug image should be published'

    finally:
        rclpy.shutdown()
