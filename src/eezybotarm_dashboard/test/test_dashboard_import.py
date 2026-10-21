"""Test package import and basic node instantiation."""

import pytest


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

    if not rclpy.ok():
        rclpy.init()
    node = EezybotDashboardNode()

    @ui.page('/test')
    def test_page():
        build_dashboard_ui(node)

    # In proper page mode, script_mode remains False
    assert not core.script_mode
    node.destroy_node()
    rclpy.shutdown()

