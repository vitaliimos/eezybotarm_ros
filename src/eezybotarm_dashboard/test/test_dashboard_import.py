"""Test package import and basic node instantiation."""


def test_import_package():
    import eezybotarm_dashboard
    assert eezybotarm_dashboard is not None


def test_import_node():
    from eezybotarm_dashboard.dashboard_node import EezybotDashboardNode
    assert EezybotDashboardNode is not None


def test_missing_required_parameters_raises():
    import pytest
    import rclpy
    from eezybotarm_dashboard.dashboard_node import EezybotDashboardNode
    from rclpy.exceptions import ParameterUninitializedException

    if not rclpy.ok():
        rclpy.init()
    try:
        with pytest.raises(ParameterUninitializedException):
            EezybotDashboardNode()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_dashboard_page_build():
    import rclpy
    from eezybotarm_dashboard.dashboard_node import EezybotDashboardNode, build_dashboard_ui
    from nicegui import core, ui
    from rclpy.parameter import Parameter

    if not rclpy.ok():
        rclpy.init()
    params = [
        Parameter('host', Parameter.Type.STRING, '0.0.0.0'),
        Parameter('port', Parameter.Type.INTEGER, 8080),
        Parameter('camera_raw_topic', Parameter.Type.STRING, '/camera/image_raw/compressed'),
        Parameter(
            'camera_debug_topic',
            Parameter.Type.STRING,
            '/eezybotarm_vision/debug_image/compressed',
        ),
        Parameter('table_width_m', Parameter.Type.DOUBLE, 0.300),
        Parameter('table_depth_m', Parameter.Type.DOUBLE, 0.200),
        Parameter('robot_x_offset_m', Parameter.Type.DOUBLE, 0.102),
        Parameter('robot_y_offset_m', Parameter.Type.DOUBLE, 0.0),
        Parameter('table_z_m', Parameter.Type.DOUBLE, 0.095),
        Parameter('cube_height_m', Parameter.Type.DOUBLE, 0.025),
        Parameter('sticker_width_m', Parameter.Type.DOUBLE, 0.030),
        Parameter('sticker_height_m', Parameter.Type.DOUBLE, 0.015),
        Parameter('dataset_dir', Parameter.Type.STRING, '/tmp/test_dataset'),
    ]
    node = EezybotDashboardNode(parameter_overrides=params)

    assert hasattr(node, 'mapper')
    xyz = node.mapper.get_zone_center_robot_xyz('middle')
    assert len(xyz) == 3
    assert xyz[0] > 0.0

    @ui.page('/test')
    def test_page():
        build_dashboard_ui(node)

    # In proper page mode, script_mode remains False
    assert not core.script_mode
    node.destroy_node()
    rclpy.shutdown()
