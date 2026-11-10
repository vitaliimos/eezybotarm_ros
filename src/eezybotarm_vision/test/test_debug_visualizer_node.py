"""Unit tests for DebugVisualizerNode."""

from unittest.mock import patch

from eezybotarm_vision.debug_visualizer_node import DebugVisualizerNode
import numpy as np
import pytest
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _make_test_node(**kwargs) -> DebugVisualizerNode:
    params = [
        Parameter('debug_image_jpeg_quality', Parameter.Type.INTEGER, 80),
    ]
    return DebugVisualizerNode(parameter_overrides=params, **kwargs)


def _create_mock_image_msg() -> Image:
    msg = Image()
    msg.height = 100
    msg.width = 100
    msg.encoding = 'bgr8'
    msg.step = 300
    msg.data = np.zeros((100, 100, 3), dtype=np.uint8).tobytes()
    return msg


@pytest.fixture(autouse=True)
def rclpy_init_shutdown():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_missing_required_parameters_raises():
    with pytest.raises(ParameterUninitializedException):
        DebugVisualizerNode()


def test_skips_image_conversion_when_no_subscribers():
    node = _make_test_node()
    try:
        with (
            patch.object(node.visualizer, 'has_subscribers', return_value=False),
            patch.object(node, '_convert_image') as mock_convert,
        ):
            node._handle_image(_create_mock_image_msg())
            mock_convert.assert_not_called()
    finally:
        node.destroy_node()


def test_publishes_when_subscribers_present():
    node = _make_test_node()
    try:
        node._status_callback(String(data='CONFIRMING (2/5)'))
        msg = _create_mock_image_msg()

        with (
            patch.object(node.visualizer, 'has_subscribers', return_value=True),
            patch.object(node.visualizer, 'publish') as mock_publish,
        ):
            node._handle_image(msg)
            mock_publish.assert_called_once()
            kwargs = mock_publish.call_args.kwargs
            assert kwargs['state_text'] == 'CALIBRATING'
    finally:
        node.destroy_node()
