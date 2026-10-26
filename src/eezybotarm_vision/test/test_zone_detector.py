"""Unit tests for zone_detector and coordinate_mapper."""

import cv2
from eezybotarm_vision.coordinate_mapper import CoordinateMapper
from eezybotarm_vision.zone_detector import (
    _RingCandidate,
    _select_best_ring_quad,
    detect_zones,
    get_real_coordinates,
    identify_zone,
    order_corners,
    smooth_calibration,
)
import numpy as np
import pytest


def test_order_corners():
    unordered = np.array([[100, 100], [0, 100], [0, 0], [100, 0]], dtype=np.float32)
    ordered = order_corners(unordered)
    assert np.allclose(ordered[0], [0, 0])      # TL
    assert np.allclose(ordered[1], [100, 0])    # TR
    assert np.allclose(ordered[2], [100, 100])  # BR
    assert np.allclose(ordered[3], [0, 100])    # BL

    # Verify tilted / skewed quadrilateral ordering without duplicate corners
    tilted = np.array([[10, 80], [90, 85], [15, 12], [85, 8]], dtype=np.float32)
    ordered_tilted = order_corners(tilted)
    assert np.allclose(ordered_tilted[0], [15, 12])  # TL
    assert np.allclose(ordered_tilted[1], [85, 8])   # TR
    assert np.allclose(ordered_tilted[2], [90, 85])  # BR
    assert np.allclose(ordered_tilted[3], [10, 80])  # BL

    # Ensure all 4 returned points are strictly unique
    assert len({tuple(pt) for pt in ordered_tilted}) == 4


def test_identify_zone():
    w, h = 300.0, 200.0
    assert identify_zone(50.0, 100.0, w, h) == 'left'
    assert identify_zone(150.0, 100.0, w, h) == 'middle'
    assert identify_zone(250.0, 100.0, w, h) == 'right'
    assert identify_zone(-10.0, 100.0, w, h) == 'outside'
    assert identify_zone(150.0, 250.0, w, h) == 'outside'


def test_coordinate_mapper():
    mapper = CoordinateMapper(
        table_width_m=0.300,
        table_depth_m=0.200,
        robot_x_offset_m=0.12,
        robot_y_offset_m=0.0,
        table_z_m=0.02,
        cube_height_m=0.03,
        sticker_width_m=0.030,
        sticker_height_m=0.015,
    )
    assert abs(mapper.marker_inset_x_mm - 15.0) < 1e-4
    assert abs(mapper.marker_inset_y_mm - 7.5) < 1e-4

    xr, yr, zr = mapper.table_to_robot_xyz(150.0, 100.0)
    assert abs(xr - 0.22) < 1e-4
    assert abs(yr - 0.0) < 1e-4
    assert abs(zr - 0.030) < 1e-4

    _, ly, _ = mapper.get_zone_center_robot_xyz('left')
    assert ly < 0.0

    _, ry, _ = mapper.get_zone_center_robot_xyz('right')
    assert ry > 0.0

    with pytest.raises(ValueError):
        mapper.get_zone_center_robot_xyz('invalid_zone')


def _make_workspace_image(shift_px: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Create a synthetic table image with 4 corner ring stickers."""
    dx, dy = shift_px
    img = np.full((600, 800, 3), (90, 140, 180), dtype=np.uint8)
    cv2.rectangle(img, (100 + dx, 100 + dy), (700 + dx, 500 + dy), (60, 215, 245), -1)

    ring_centers = [
        (130 + dx, 115 + dy),  # TL -> (15.0, 7.5) mm
        (670 + dx, 115 + dy),  # TR -> (285.0, 7.5) mm
        (670 + dx, 485 + dy),  # BR -> (285.0, 192.5) mm
        (130 + dx, 485 + dy),  # BL -> (15.0, 192.5) mm
    ]
    for cx, cy in ring_centers:
        cv2.rectangle(img, (cx - 30, cy - 15), (cx + 30, cy + 15), (250, 250, 250), -1)
        cv2.circle(img, (cx, cy), 12, (15, 15, 15), -1)
        cv2.circle(img, (cx, cy), 6, (250, 250, 250), -1)

    # Synthetic cube in the middle zone
    cv2.rectangle(img, (380 + dx, 280 + dy), (430 + dx, 330 + dy), (245, 245, 245), -1)
    return img


def test_detect_zones_from_corner_ring_stickers():
    img = _make_workspace_image()
    res = detect_zones(
        img,
        field_width_mm=300.0,
        field_height_mm=200.0,
        marker_inset_x_mm=15.0,
        marker_inset_y_mm=7.5,
    )
    assert res is not None
    assert set(res.zones.keys()) == {'left', 'middle', 'right'}

    tl_x, tl_y = get_real_coordinates(130.0, 115.0, res.homography_matrix)
    assert abs(tl_x - 15.0) < 1.0
    assert abs(tl_y - 7.5) < 1.0

    outer_tl_x, outer_tl_y = get_real_coordinates(100.0, 100.0, res.homography_matrix)
    assert abs(outer_tl_x - 0.0) < 1.5
    assert abs(outer_tl_y - 0.0) < 1.5

    outer_br_x, outer_br_y = get_real_coordinates(700.0, 500.0, res.homography_matrix)
    assert abs(outer_br_x - 300.0) < 1.5
    assert abs(outer_br_y - 200.0) < 1.5

    for zone_name in ('left', 'middle', 'right'):
        cx, cy = res.zones[zone_name].center_px
        rx, ry = get_real_coordinates(cx, cy, res.homography_matrix)
        assert identify_zone(rx, ry, 300.0, 200.0) == zone_name


def test_smooth_calibration():
    calib1 = detect_zones(
        _make_workspace_image((0, 0)),
        field_width_mm=300.0,
        field_height_mm=200.0,
        marker_inset_x_mm=15.0,
        marker_inset_y_mm=7.5,
    )
    calib2 = detect_zones(
        _make_workspace_image((10, 10)),
        field_width_mm=300.0,
        field_height_mm=200.0,
        marker_inset_x_mm=15.0,
        marker_inset_y_mm=7.5,
    )
    assert calib1 is not None and calib2 is not None

    smoothed = smooth_calibration(calib1, calib2, alpha=0.5)
    assert smoothed.marker_inset_x_mm == 15.0
    assert smoothed.marker_inset_y_mm == 7.5
    assert set(smoothed.zones.keys()) == {'left', 'middle', 'right'}

    expected_corners = 0.5 * calib1.field_corners_px + 0.5 * calib2.field_corners_px
    assert np.allclose(smoothed.field_corners_px, expected_corners, atol=1e-2)


def test_select_best_ring_quad_bounds_candidates():
    valid = [
        _RingCandidate(cx=100.0, cy=100.0, area=400.0, ratio_err=0.01),
        _RingCandidate(cx=700.0, cy=100.0, area=400.0, ratio_err=0.01),
        _RingCandidate(cx=700.0, cy=500.0, area=400.0, ratio_err=0.01),
        _RingCandidate(cx=100.0, cy=500.0, area=400.0, ratio_err=0.01),
    ]
    noise = [
        _RingCandidate(
            cx=float(150 + i * 10),
            cy=float(150 + (i % 7) * 30),
            area=350.0,
            ratio_err=0.20 + i * 0.01,
        )
        for i in range(40)
    ]
    quad = _select_best_ring_quad(noise + valid, min_field_area=10000.0)
    assert quad is not None
    assert np.allclose(quad[0], [100.0, 100.0])
    assert np.allclose(quad[2], [700.0, 500.0])
