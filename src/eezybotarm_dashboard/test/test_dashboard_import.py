"""Test package import and basic node instantiation."""


def test_import_package():
    import eezybotarm_dashboard
    assert eezybotarm_dashboard is not None


def test_import_node():
    from eezybotarm_dashboard.dashboard_node import EezybotDashboardNode
    assert EezybotDashboardNode is not None


def test_dashboard_page_build():
    import rclpy
    from eezybotarm_dashboard.dashboard_node import EezybotDashboardNode, build_dashboard_ui
    from nicegui import core, ui
    from rclpy.parameter import Parameter

    if not rclpy.ok():
        rclpy.init()
    params = [
        Parameter('table_width_m', Parameter.Type.DOUBLE, 0.300),
        Parameter('table_depth_m', Parameter.Type.DOUBLE, 0.200),
        Parameter('robot_x_offset_m', Parameter.Type.DOUBLE, 0.127),
        Parameter('robot_y_offset_m', Parameter.Type.DOUBLE, 0.0),
        Parameter('table_z_m', Parameter.Type.DOUBLE, 0.095),
        Parameter('cube_height_m', Parameter.Type.DOUBLE, 0.025),
    ]
    node = EezybotDashboardNode(parameter_overrides=params)

    @ui.page('/test')
    def test_page():
        build_dashboard_ui(node)

    # In proper page mode, script_mode remains False
    assert not core.script_mode
    node.destroy_node()
    rclpy.shutdown()
