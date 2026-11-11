"""Unit tests for LocalizedCube message round-trip."""

from eezybotarm_vision.localized_cube import LocalizedCube
from eezybotarm_vision.yolo_detector import CubeDetection


def test_msg_round_trip():
    det = CubeDetection(
        class_id=1,
        class_name='green',
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        center_px=(2.0, 3.0),
        angle=0.5,
        corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    )
    cube = LocalizedCube(detection=det, zone='middle', x_mm=10.0, y_mm=20.0)
    assert LocalizedCube.from_msg(cube.to_msg()) == cube
