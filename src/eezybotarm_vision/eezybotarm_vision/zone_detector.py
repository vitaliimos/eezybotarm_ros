"""Workspace zone detection and homography calibration for EezyBotArm."""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import cv2
import numpy as np

ZoneName = Literal['left', 'middle', 'right']


@dataclass
class Zone:
    """Represents a single detected zone on the workspace."""

    name: ZoneName
    center_px: tuple[int, int]
    corners_px: np.ndarray  # shape (4, 2), ordered: TL, TR, BR, BL


@dataclass
class ZoneDetectionResult:
    """Container for workspace zones and homography transformation matrix."""

    zones: dict[ZoneName, Zone]
    field_corners_px: np.ndarray  # shape (4, 2): TL, TR, BR, BL
    homography_matrix: np.ndarray
    field_width_mm: float
    field_height_mm: float


@dataclass
class _Candidate:
    cx: int
    cy: int
    corners: np.ndarray
    cnt: np.ndarray
    area: float


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 quadrilateral points into [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]  # min (x + y)
    tr = pts[np.argmax(d)]  # max (x - y)
    br = pts[np.argmax(s)]  # max (x + y)
    bl = pts[np.argmin(d)]  # min (x - y)
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _select_best_triplet(candidates: list[_Candidate]) -> list[_Candidate] | None:
    """Select the most consistent left-middle-right triplet of zone candidates."""
    if len(candidates) < 3:
        return None

    best_triplet: list[_Candidate] | None = None
    best_score = float('inf')

    for triplet in combinations(candidates, 3):
        ordered = sorted(triplet, key=lambda c: c.cx)
        areas = [c.area for c in ordered]
        area_ratio = max(areas) / min(areas)
        if area_ratio > 2.5:
            continue

        mean_dim = float(np.mean([np.sqrt(a) for a in areas]))
        cys = [c.cy for c in ordered]
        cy_spread = (max(cys) - min(cys)) / mean_dim
        if cy_spread > 0.35:
            continue

        dx1 = ordered[1].cx - ordered[0].cx
        dx2 = ordered[2].cx - ordered[1].cx
        if min(dx1, dx2) < 0.4 * mean_dim:
            continue

        score = (area_ratio - 1.0) + (2.0 * cy_spread)
        if score < best_score:
            best_score = score
            best_triplet = ordered

    return best_triplet


def _find_triplet_with_params(
    blurred: np.ndarray,
    img_area: float,
    min_zone_area: float,
    block_size: int,
    c_val: int,
    morph_size: int,
) -> list[_Candidate] | None:
    """Run thresholding and contour extraction for a single parameter configuration."""
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_val,
    )

    kernel = np.ones((morph_size, morph_size), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[_Candidate] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_zone_area < area < (img_area * 0.35):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                hull = cv2.convexHull(cnt)
                hull_peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                poly_area = cv2.contourArea(approx)
                if not (min_zone_area < poly_area < (img_area * 0.35)):
                    continue
                m = cv2.moments(approx)
                if m['m00'] != 0:
                    cx = int(m['m10'] / m['m00'])
                    cy = int(m['m01'] / m['m00'])
                    candidates.append(
                        _Candidate(
                            cx=cx,
                            cy=cy,
                            corners=order_corners(approx),
                            cnt=approx,
                            area=poly_area,
                        )
                    )

    # 1. Discard any candidate enclosing another (e.g. outer paper border)
    filtered: list[_Candidate] = []
    for c in candidates:
        encloses_other = any(
            c.area > 1.8 * other.area
            and cv2.pointPolygonTest(c.cnt, (float(other.cx), float(other.cy)), False) >= 0
            for other in candidates
            if other is not c
        )
        if not encloses_other:
            filtered.append(c)
    candidates = filtered

    # 2. Deduplicate concentric contours produced by thick marker lines
    candidates.sort(key=lambda c: c.area, reverse=True)
    unique: list[_Candidate] = []
    for c in candidates:
        is_duplicate = False
        for u in unique:
            dist = float(np.hypot(c.cx - u.cx, c.cy - u.cy))
            zone_dim = float(np.sqrt(u.area))
            inside_u = cv2.pointPolygonTest(u.cnt, (float(c.cx), float(c.cy)), False) >= 0
            inside_c = cv2.pointPolygonTest(c.cnt, (float(u.cx), float(u.cy)), False) >= 0
            if dist < 0.3 * zone_dim or inside_u or inside_c:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(c)

    return _select_best_triplet(unique)


def detect_zones(
    img: np.ndarray,
    field_width_mm: float = 300.0,
    field_height_mm: float = 200.0,
    min_zone_area: float = 10000.0,
) -> ZoneDetectionResult | None:
    """Detect the three workspace zones (left, middle, right) and compute homography."""
    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    img_area = float(img.shape[0] * img.shape[1])
    param_passes = [
        (21, 10, 6),
        (21, 7, 7),
        (21, 7, 9),
        (31, 8, 9),
    ]

    selected: list[_Candidate] | None = None
    for block_size, c_val, morph_size in param_passes:
        selected = _find_triplet_with_params(
            blurred, img_area, min_zone_area, block_size, c_val, morph_size
        )
        if selected is not None:
            break

    if selected is None:
        return None

    zone_names: list[ZoneName] = ['left', 'middle', 'right']
    zones: dict[ZoneName, Zone] = {}
    for name, c in zip(zone_names, selected):
        zones[name] = Zone(
            name=name,
            center_px=(c.cx, c.cy),
            corners_px=c.corners,
        )

    # Outer sheet corners: left zone gives TL/BL, right zone gives TR/BR
    tl = zones['left'].corners_px[0]
    tr = zones['right'].corners_px[1]
    br = zones['right'].corners_px[2]
    bl = zones['left'].corners_px[3]

    field_corners = np.array([tl, tr, br, bl], dtype=np.float32)
    pts_dst = np.array(
        [
            [0.0, 0.0],
            [field_width_mm, 0.0],
            [field_width_mm, field_height_mm],
            [0.0, field_height_mm],
        ],
        dtype=np.float32,
    )

    h_matrix, _ = cv2.findHomography(field_corners, pts_dst)
    if h_matrix is None:
        return None

    return ZoneDetectionResult(
        zones=zones,
        field_corners_px=field_corners,
        homography_matrix=h_matrix,
        field_width_mm=field_width_mm,
        field_height_mm=field_height_mm,
    )


def get_real_coordinates(
    pixel_x: float, pixel_y: float, matrix: np.ndarray
) -> tuple[float, float]:
    """Transform pixel coordinates to real-world millimeters using homography matrix."""
    point_pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    point_real = cv2.perspectiveTransform(point_pixel, matrix)
    return float(point_real[0][0][0]), float(point_real[0][0][1])


def identify_zone(
    x_mm: float,
    y_mm: float,
    field_width_mm: float = 300.0,
    field_height_mm: float = 200.0,
    margin_mm: float = 5.0,
) -> str:
    """Identify which zone a physical (x_mm, y_mm) point falls into."""
    if not (
        -margin_mm <= x_mm <= field_width_mm + margin_mm
        and -margin_mm <= y_mm <= field_height_mm + margin_mm
    ):
        return 'outside'

    zone_width = field_width_mm / 3.0
    if x_mm < zone_width:
        return 'left'
    elif x_mm < 2.0 * zone_width:
        return 'middle'
    return 'right'


def draw_zones(
    image: np.ndarray,
    result: ZoneDetectionResult,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw detected zones and their labels onto a copy of the image."""
    annotated = image.copy()
    for name, zone in result.zones.items():
        pts = zone.corners_px.astype(np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
        cv2.putText(
            annotated,
            name.capitalize(),
            (zone.center_px[0] - 30, zone.center_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
    return annotated
