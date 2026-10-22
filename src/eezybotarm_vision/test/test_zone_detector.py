"""Unit tests for zone_detector, coordinate_mapper, and detector logic."""

import cv2
from eezybotarm_vision.coordinate_mapper import CoordinateMapper
from eezybotarm_vision.yolo_detector import YoloCubeDetector
from eezybotarm_vision.zone_detector import (
    detect_zones,
    get_real_coordinates,
    identify_zone,
    order_corners,
    smooth_calibration,
)
import numpy as np


def test_order_corners():
    """Verify corner ordering to TL, TR, BR, BL."""
    unordered = np.array([[100, 100], [0, 100], [0, 0], [100, 0]], dtype=np.float32)
    ordered = order_corners(unordered)
    assert np.allclose(ordered[0], [0, 0])      # TL
    assert np.allclose(ordered[1], [100, 0])    # TR
    assert np.allclose(ordered[2], [100, 100])  # BR
    assert np.allclose(ordered[3], [0, 100])    # BL


def test_identify_zone():
    """Verify zone categorization based on width thirds."""
    w, h = 300.0, 200.0
    assert identify_zone(50.0, 100.0, w, h) == 'left'
    assert identify_zone(150.0, 100.0, w, h) == 'middle'
    assert identify_zone(250.0, 100.0, w, h) == 'right'
    assert identify_zone(-10.0, 100.0, w, h) == 'outside'
    assert identify_zone(150.0, 250.0, w, h) == 'outside'


def test_coordinate_mapper():
    """Verify mapping from paper mm to robot base_link xyz."""
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

    # Center of paper (x=150mm, y=100mm)
    xr, yr, zr = mapper.paper_to_robot_xyz(150.0, 100.0)
    assert abs(xr - 0.22) < 1e-4  # 0.12 + 0.100
    assert abs(yr - 0.0) < 1e-4   # 0.0 + 0.0
    assert abs(zr - 0.035) < 1e-4  # 0.02 + 0.015

    # Left zone should have negative Y in base_link (to robot's right)
    lx, ly, _ = mapper.get_zone_center_robot_xyz('left')
    assert ly < 0.0

    # Right zone should have positive Y in base_link (to robot's left)
    rx, ry, _ = mapper.get_zone_center_robot_xyz('right')
    assert ry > 0.0


def _draw_sticker_with_ring(
    img: np.ndarray,
    center_px: tuple[int, int],
    sticker_half_size_px: tuple[int, int] = (30, 15),
    outer_radius_px: int = 12,
    inner_radius_px: int = 6,
) -> None:
    """Draw a horizontal white sticker with a concentric black ring."""
    cx, cy = center_px
    hw, hh = sticker_half_size_px
    cv2.rectangle(img, (cx - hw, cy - hh), (cx + hw, cy + hh), (250, 250, 250), -1)
    cv2.circle(img, (cx, cy), outer_radius_px, (15, 15, 15), -1)
    cv2.circle(img, (cx, cy), inner_radius_px, (250, 250, 250), -1)


def test_detect_zones_from_corner_ring_stickers():
    """Verify calibration from 4 corner ring stickers on colored paper with 15x7.5mm inset."""
    # Create wooden table with a yellow 300x200mm workspace sheet (2 px/mm -> 600x400 px)
    img = np.full((600, 800, 3), (90, 140, 180), dtype=np.uint8)
    # Sheet spans x=[100..700] (600px = 300mm), y=[100..500] (400px = 200mm)
    cv2.rectangle(img, (100, 100), (700, 500), (60, 215, 245), -1)

    # Stickers (30x15mm -> 60x30px) flush with corners -> centers inset by (30px, 15px)
    ring_centers = [
        (130, 115),  # TL -> (15.0, 7.5) mm
        (670, 115),  # TR -> (285.0, 7.5) mm
        (670, 485),  # BR -> (285.0, 192.5) mm
        (130, 485),  # BL -> (15.0, 192.5) mm
    ]
    for pt in ring_centers:
        _draw_sticker_with_ring(img, pt)

    # Place a synthetic cube in the middle zone to verify it does not interfere with calibration
    cv2.rectangle(img, (380, 280), (430, 330), (245, 245, 245), -1)

    res = detect_zones(
        img,
        field_width_mm=300.0,
        field_height_mm=200.0,
        min_zone_area=5000.0,
        marker_inset_x_mm=15.0,
        marker_inset_y_mm=7.5,
    )
    assert res is not None
    assert set(res.zones.keys()) == {'left', 'middle', 'right'}

    # Ring center TL (130, 115) must map to (15.0, 7.5) mm
    tl_x, tl_y = get_real_coordinates(130.0, 115.0, res.homography_matrix)
    assert abs(tl_x - 15.0) < 1.0
    assert abs(tl_y - 7.5) < 1.0

    # Sheet outer corner TL (100, 100) must map to (0.0, 0.0) mm
    outer_tl_x, outer_tl_y = get_real_coordinates(100.0, 100.0, res.homography_matrix)
    assert abs(outer_tl_x - 0.0) < 1.5
    assert abs(outer_tl_y - 0.0) < 1.5

    # Sheet outer corner BR (700, 500) must map to (300.0, 200.0) mm
    outer_br_x, outer_br_y = get_real_coordinates(700.0, 500.0, res.homography_matrix)
    assert abs(outer_br_x - 300.0) < 1.5
    assert abs(outer_br_y - 200.0) < 1.5

    # Virtual zone centers must map to their respective zones
    for zone_name in ('left', 'middle', 'right'):
        cx, cy = res.zones[zone_name].center_px
        rx, ry = get_real_coordinates(cx, cy, res.homography_matrix)
        assert identify_zone(rx, ry, 300.0, 200.0) == zone_name


def test_detect_zones_synthetic():
    """Verify zone detection on synthetic 3-column image."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    # Draw 3 black rectangular zones on white background
    # Zone 1 (Left)
    cv2.rectangle(img, (100, 150), (280, 450), (0, 0, 0), 4)
    # Zone 2 (Middle)
    cv2.rectangle(img, (310, 150), (490, 450), (0, 0, 0), 4)
    # Zone 3 (Right)
    cv2.rectangle(img, (520, 150), (700, 450), (0, 0, 0), 4)

    res = detect_zones(img, min_zone_area=5000.0)
    assert res is not None
    assert 'left' in res.zones
    assert 'middle' in res.zones
    assert 'right' in res.zones

    # Check homography mapping on center point
    mid_center = res.zones['middle'].center_px
    rx, ry = get_real_coordinates(mid_center[0], mid_center[1], res.homography_matrix)
    zone_id = identify_zone(rx, ry, res.field_width_mm, res.field_height_mm)
    assert zone_id == 'middle'


def test_detect_zones_with_outer_box():
    """Verify that outer bounding box is ignored and 3 inner zones are detected."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    # Outer bounding box enclosing all 3 zones
    cv2.rectangle(img, (80, 130), (720, 470), (0, 0, 0), 4)
    # Zone 1 (Left)
    cv2.rectangle(img, (100, 150), (280, 450), (0, 0, 0), 4)
    # Zone 2 (Middle)
    cv2.rectangle(img, (310, 150), (490, 450), (0, 0, 0), 4)
    # Zone 3 (Right)
    cv2.rectangle(img, (520, 150), (700, 450), (0, 0, 0), 4)

    res = detect_zones(img, min_zone_area=5000.0)
    assert res is not None
    assert 'left' in res.zones
    assert 'middle' in res.zones
    assert 'right' in res.zones

    right_zone = res.zones['right']
    w = np.linalg.norm(right_zone.corners_px[1] - right_zone.corners_px[0])
    assert w < 300.0


def test_detect_zones_deduplicate_inner_outer_contours():
    """Verify that inner and outer contours of thick marker lines are deduplicated."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    # Zone 1 (Left): width 170, height 300
    cv2.rectangle(img, (100, 150), (270, 450), (0, 0, 0), 4)
    # Zone 2 (Middle): width 190, height 320, thicker border -> inner contour has large area
    cv2.rectangle(img, (310, 140), (500, 460), (0, 0, 0), 6)
    # Zone 3 (Right): width 160, height 280 (smaller area than middle inner contour)
    cv2.rectangle(img, (530, 160), (690, 440), (0, 0, 0), 4)

    res = detect_zones(img, min_zone_area=5000.0)
    assert res is not None
    assert 'left' in res.zones
    assert 'middle' in res.zones
    assert 'right' in res.zones

    # Centers must be strictly increasing in X and well separated
    lx = res.zones['left'].center_px[0]
    mx = res.zones['middle'].center_px[0]
    rx = res.zones['right'].center_px[0]
    assert lx < mx < rx
    assert (mx - lx) > 100
    assert (rx - mx) > 100


def test_detect_zones_with_broken_line_gap():
    """Verify zone detection recovers when glare causes a small gap in a zone border."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 150), (280, 450), (0, 0, 0), 4)
    cv2.rectangle(img, (310, 150), (490, 450), (0, 0, 0), 4)
    cv2.rectangle(img, (520, 150), (700, 450), (80, 80, 80), 3)
    # Introduce a 7-pixel gap in the top border of the right zone (too wide for 6x6 closing)
    cv2.line(img, (550, 148), (556, 152), (255, 255, 255), 5)

    res = detect_zones(img, min_zone_area=5000.0)
    assert res is not None
    assert 'left' in res.zones
    assert 'middle' in res.zones
    assert 'right' in res.zones


def test_letterbox_aspect_ratio_preservation():
    """Verify LetterBox padding and inverse coordinate mapping on 16:9 image."""
    orig_h, orig_w = 576, 1024
    img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = YoloCubeDetector._letterbox(img, (640, 640))

    assert padded.shape == (640, 640, 3)
    assert abs(scale - (640.0 / 1024.0)) < 1e-6
    assert abs(pad_x - 0.0) < 1e-6
    assert abs(pad_y - 140.0) < 1e-6

    # Verify _postprocess maps padded 640x640 coordinates back to original 1024x576
    det = object.__new__(YoloCubeDetector)
    det.conf_threshold = 0.5
    det.iou_threshold = 0.45
    fake_out = np.array([[[320.0, 320.0, 64.0, 64.0, 0.98, 0.0, 0.1]]], dtype=np.float32)
    fake_out = np.transpose(fake_out, (0, 2, 1))
    results = det._postprocess(fake_out, scale, pad_x, pad_y, orig_w, orig_h)

    assert len(results) == 1
    assert abs(results[0].center_px[0] - 512.0) < 1e-3
    assert abs(results[0].center_px[1] - 288.0) < 1e-3


def test_detector_fail_fast_on_missing_model():
    """Verify detector raises RuntimeError when model path does not exist."""
    import pytest

    with pytest.raises(RuntimeError):

        YoloCubeDetector(weights_path='/non/existent/model.engine')


def test_smooth_calibration():
    """Verify smooth_calibration interpolates field corners using EMA."""
    # Synthetic image with 3 black rectangular zones
    img1 = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(img1, (100, 150), (280, 450), (0, 0, 0), 4)
    cv2.rectangle(img1, (310, 150), (490, 450), (0, 0, 0), 4)
    cv2.rectangle(img1, (520, 150), (700, 450), (0, 0, 0), 4)
    calib1 = detect_zones(img1, min_zone_area=5000.0)
    assert calib1 is not None

    # Shifted image by 10 pixels right and down
    img2 = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(img2, (110, 160), (290, 460), (0, 0, 0), 4)
    cv2.rectangle(img2, (320, 160), (500, 460), (0, 0, 0), 4)
    cv2.rectangle(img2, (530, 160), (710, 460), (0, 0, 0), 4)
    calib2 = detect_zones(img2, min_zone_area=5000.0)
    assert calib2 is not None

    alpha = 0.5
    smoothed = smooth_calibration(calib1, calib2, alpha=alpha)
    assert smoothed is not None
    assert set(smoothed.zones.keys()) == {'left', 'middle', 'right'}

    expected_corners = 0.5 * calib1.field_corners_px + 0.5 * calib2.field_corners_px
    assert np.allclose(smoothed.field_corners_px, expected_corners, atol=1e-2)
