"""Workspace zone detection and homography calibration for EezyBotArm."""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import cv2
import numpy as np

ZoneName = Literal['left', 'middle', 'right']

# Ideal inner-to-outer area ratio for 12mm outer / 6mm inner diameter rings: (6/12)^2 = 0.25
IDEAL_RING_HOLE_RATIO = 0.25


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


@dataclass
class _RingCandidate:
    cx: float
    cy: float
    area: float
    ratio_err: float


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


def _contour_solidity_and_circularity(cnt: np.ndarray, area: float) -> tuple[float, float]:
    """Compute contour solidity (area / convex hull area) and circularity."""
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    peri = cv2.arcLength(cnt, True)
    circularity = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
    return float(solidity), float(circularity)


def _extract_ring_candidates_from_pass(
    blurred: np.ndarray,
    gray: np.ndarray,
    min_ring_area: float,
    max_ring_area: float,
    block_size: int,
    c_val: int,
) -> list[_RingCandidate]:
    """Extract concentric black-ring fiducial candidates for a threshold pass."""
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_val,
    )
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    hier = hierarchy[0]
    h_img, w_img = gray.shape[:2]
    found: list[_RingCandidate] = []

    for i, cnt_out in enumerate(contours):
        first_child = hier[i][2]
        if first_child < 0:
            continue

        area_out = cv2.contourArea(cnt_out)
        if not (min_ring_area <= area_out <= max_ring_area):
            continue

        sol_out, circ_out = _contour_solidity_and_circularity(cnt_out, area_out)
        if sol_out < 0.80 or circ_out < 0.38:
            continue

        m_out = cv2.moments(cnt_out)
        if m_out['m00'] == 0:
            continue
        cx_out = float(m_out['m10'] / m_out['m00'])
        cy_out = float(m_out['m01'] / m_out['m00'])
        outer_rad = float(np.sqrt(area_out / np.pi))

        child_idx = first_child
        while child_idx >= 0:
            cnt_in = contours[child_idx]
            child_idx = hier[child_idx][0]

            area_in = cv2.contourArea(cnt_in)
            if area_in <= 4.0:
                continue

            hole_ratio = area_in / area_out
            if not (0.06 <= hole_ratio <= 0.65):
                continue

            sol_in, circ_in = _contour_solidity_and_circularity(cnt_in, area_in)
            if sol_in < 0.75 or circ_in < 0.35:
                continue

            m_in = cv2.moments(cnt_in)
            if m_in['m00'] == 0:
                continue
            cx_in = float(m_in['m10'] / m_in['m00'])
            cy_in = float(m_in['m01'] / m_in['m00'])

            if np.hypot(cx_out - cx_in, cy_out - cy_in) > 0.22 * outer_rad:
                continue

            cx = 0.5 * (cx_out + cx_in)
            cy = 0.5 * (cy_out + cy_in)
            ix, iy = int(round(cx)), int(round(cy))
            if not (0 <= ix < w_img and 0 <= iy < h_img):
                continue

            # Verify inner hole is brighter than the surrounding black ring stroke
            hole_intensity = float(gray[iy, ix])
            mid_r = max(2, int(round(0.75 * outer_rad)))
            sample_vals = []
            for dx, dy in ((mid_r, 0), (-mid_r, 0), (0, mid_r), (0, -mid_r)):
                sx = min(max(ix + dx, 0), w_img - 1)
                sy = min(max(iy + dy, 0), h_img - 1)
                sample_vals.append(float(gray[sy, sx]))
            ring_intensity = float(np.mean(sample_vals))
            if hole_intensity - ring_intensity < 15.0:
                continue

            found.append(
                _RingCandidate(
                    cx=cx,
                    cy=cy,
                    area=area_out,
                    ratio_err=abs(hole_ratio - IDEAL_RING_HOLE_RATIO),
                )
            )
            break

    return found


def _select_best_ring_quad(
    candidates: list[_RingCandidate],
    min_field_area: float,
) -> np.ndarray | None:
    """Select 4 ring candidates forming the most consistent workspace quadrilateral."""
    if len(candidates) < 4:
        return None

    best_corners: np.ndarray | None = None
    best_score = float('inf')

    for quad in combinations(candidates, 4):
        areas = [c.area for c in quad]
        area_ratio = max(areas) / min(areas)
        if area_ratio > 4.5:
            continue

        pts = np.array([[c.cx, c.cy] for c in quad], dtype=np.float32)
        ordered = order_corners(pts)
        cnt = ordered.reshape((-1, 1, 2)).astype(np.float32)
        if not cv2.isContourConvex(cnt):
            continue

        poly_area = float(cv2.contourArea(cnt))
        if poly_area < min_field_area:
            continue

        top_w = float(np.linalg.norm(ordered[1] - ordered[0]))
        bot_w = float(np.linalg.norm(ordered[2] - ordered[3]))
        left_h = float(np.linalg.norm(ordered[3] - ordered[0]))
        right_h = float(np.linalg.norm(ordered[2] - ordered[1]))

        max_w = max(top_w, bot_w)
        max_h = max(left_h, right_h)
        if max_w <= 0 or max_h <= 0:
            continue
        if min(top_w, bot_w) < 0.45 * max_w or min(left_h, right_h) < 0.45 * max_h:
            continue

        ratio_err_sum = sum(c.ratio_err for c in quad)
        symmetry_err = abs(top_w - bot_w) / max_w + abs(left_h - right_h) / max_h
        score = (area_ratio - 1.0) + 2.0 * ratio_err_sum + symmetry_err
        if score < best_score:
            best_score = score
            best_corners = ordered

    return best_corners


def _build_result_from_corners(
    marker_corners_px: np.ndarray,
    field_width_mm: float,
    field_height_mm: float,
    marker_inset_x_mm: float = 0.0,
    marker_inset_y_mm: float = 0.0,
) -> ZoneDetectionResult | None:
    """Compute homography from 4 corner markers and reconstruct 3 virtual zones."""
    pts_dst = np.array(
        [
            [marker_inset_x_mm, marker_inset_y_mm],
            [field_width_mm - marker_inset_x_mm, marker_inset_y_mm],
            [field_width_mm - marker_inset_x_mm, field_height_mm - marker_inset_y_mm],
            [marker_inset_x_mm, field_height_mm - marker_inset_y_mm],
        ],
        dtype=np.float32,
    )

    h_matrix, _ = cv2.findHomography(marker_corners_px, pts_dst)
    if h_matrix is None:
        return None

    try:
        h_inv = np.linalg.inv(h_matrix)
    except np.linalg.LinAlgError:
        return None

    zone_names: list[ZoneName] = ['left', 'middle', 'right']
    zone_width_mm = field_width_mm / 3.0
    zones: dict[ZoneName, Zone] = {}

    for idx, name in enumerate(zone_names):
        x0 = idx * zone_width_mm
        x1 = (idx + 1) * zone_width_mm
        rect_mm = np.array(
            [[[x0, 0.0], [x1, 0.0], [x1, field_height_mm], [x0, field_height_mm]]],
            dtype=np.float32,
        )
        corners_px = cv2.perspectiveTransform(rect_mm, h_inv)[0].astype(np.float32)

        center_mm = np.array(
            [[[0.5 * (x0 + x1), 0.5 * field_height_mm]]],
            dtype=np.float32,
        )
        center_px_f = cv2.perspectiveTransform(center_mm, h_inv)[0][0]
        zones[name] = Zone(
            name=name,
            center_px=(int(round(center_px_f[0])), int(round(center_px_f[1]))),
            corners_px=corners_px,
        )

    return ZoneDetectionResult(
        zones=zones,
        field_corners_px=marker_corners_px.astype(np.float32),
        homography_matrix=h_matrix,
        field_width_mm=field_width_mm,
        field_height_mm=field_height_mm,
    )


def smooth_calibration(
    prev: ZoneDetectionResult,
    new: ZoneDetectionResult,
    alpha: float,
    marker_inset_x_mm: float = 0.0,
    marker_inset_y_mm: float = 0.0,
) -> ZoneDetectionResult:
    """Blend previous and new field corners via exponential moving average (EMA)."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    smoothed_corners = (1.0 - alpha) * prev.field_corners_px + alpha * new.field_corners_px
    result = _build_result_from_corners(
        smoothed_corners.astype(np.float32),
        field_width_mm=new.field_width_mm,
        field_height_mm=new.field_height_mm,
        marker_inset_x_mm=marker_inset_x_mm,
        marker_inset_y_mm=marker_inset_y_mm,
    )
    return result if result is not None else new


def _detect_zones_from_rings(
    blurred: np.ndarray,
    gray: np.ndarray,
    img_area: float,
    field_width_mm: float,
    field_height_mm: float,
    min_zone_area: float,
    marker_inset_x_mm: float,
    marker_inset_y_mm: float,
) -> ZoneDetectionResult | None:
    """Detect workspace from 4 corner ring stickers and project 3 virtual zones."""
    min_ring_area = max(45.0, img_area * 0.00008)
    max_ring_area = img_area * 0.025
    unique: list[_RingCandidate] = []

    for block_size, c_val in ((21, 8), (31, 10), (15, 6)):
        pass_candidates = _extract_ring_candidates_from_pass(
            blurred, gray, min_ring_area, max_ring_area, block_size, c_val
        )
        for cand in pass_candidates:
            merged = False
            for idx, existing in enumerate(unique):
                tol = 0.5 * float(np.sqrt(max(cand.area, existing.area)))
                if np.hypot(cand.cx - existing.cx, cand.cy - existing.cy) <= tol:
                    if cand.ratio_err < existing.ratio_err:
                        unique[idx] = cand
                    merged = True
                    break
            if not merged:
                unique.append(cand)

    marker_corners = _select_best_ring_quad(unique, min_field_area=min_zone_area * 2.0)
    if marker_corners is None:
        return None

    return _build_result_from_corners(
        marker_corners,
        field_width_mm=field_width_mm,
        field_height_mm=field_height_mm,
        marker_inset_x_mm=marker_inset_x_mm,
        marker_inset_y_mm=marker_inset_y_mm,
    )


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
    marker_inset_x_mm: float = 15.0,
    marker_inset_y_mm: float = 7.5,
) -> ZoneDetectionResult | None:
    """Detect workspace calibration via 4 corner ring stickers or 3 drawn zone rectangles."""
    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    img_area = float(img.shape[0] * img.shape[1])

    # Primary method: 4 corner ring fiducial stickers
    ring_result = _detect_zones_from_rings(
        blurred=blurred,
        gray=gray,
        img_area=img_area,
        field_width_mm=field_width_mm,
        field_height_mm=field_height_mm,
        min_zone_area=min_zone_area,
        marker_inset_x_mm=marker_inset_x_mm,
        marker_inset_y_mm=marker_inset_y_mm,
    )
    if ring_result is not None:
        return ring_result

    # Fallback: 3 drawn rectangular zone contours
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
    """Draw detected zones, corner fiducial markers, and labels onto a copy of the image."""
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

    for corner in result.field_corners_px:
        cx, cy = int(round(corner[0])), int(round(corner[1]))
        cv2.circle(annotated, (cx, cy), 6, (0, 165, 255), 2)
        cv2.drawMarker(
            annotated,
            (cx, cy),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            markerSize=8,
            thickness=1,
        )
    return annotated
