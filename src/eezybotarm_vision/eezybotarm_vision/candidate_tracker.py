"""Candidate cube stability tracking for pick-and-place sorting."""

import math

from .localized_cube import LocalizedCube


class CandidateTracker:
    """Track stability of a candidate cube across video frames."""

    def __init__(self, stable_frames_required: int, tolerance_mm: float) -> None:
        """
        Initialize the tracker.

        :param stable_frames_required: Consecutive frames a cube must be seen to confirm.
        :param tolerance_mm: Maximum Euclidean distance in table mm to match.
        """
        self.stable_frames_required = stable_frames_required
        self.tolerance_mm = tolerance_mm
        self.candidate: LocalizedCube | None = None
        self.frame_count = 0

    @staticmethod
    def _distance_mm(a: LocalizedCube, b: LocalizedCube) -> float:
        """Return Euclidean distance between two cubes in table mm."""
        return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm)

    def is_match(self, a: LocalizedCube, b: LocalizedCube) -> bool:
        """Return True if both cubes share a class and are within tolerance."""
        return (
            a.detection.class_id == b.detection.class_id
            and self._distance_mm(a, b) < self.tolerance_mm
        )

    def update(self, cubes: list[LocalizedCube]) -> LocalizedCube | None:
        """
        Update tracking with this frame's cubes.

        :param cubes: Localized cubes in the zone of interest.
        :return: The confirmed cube once the stability threshold is reached, else None.
        """
        if not cubes:
            self.reset()
            return None

        if self.candidate is not None:
            matches = [c for c in cubes if self.is_match(self.candidate, c)]
            if matches:
                self.candidate = min(
                    matches, key=lambda c: self._distance_mm(self.candidate, c)
                )
                self.frame_count += 1
            else:
                self.candidate = min(
                    cubes, key=lambda c: self._distance_mm(self.candidate, c)
                )
                self.frame_count = 1
        else:
            self.candidate = max(cubes, key=lambda c: c.detection.confidence)
            self.frame_count = 1

        return (
            self.candidate if self.frame_count >= self.stable_frames_required else None
        )

    def reset(self) -> None:
        """Clear the current candidate."""
        self.candidate = None
        self.frame_count = 0
