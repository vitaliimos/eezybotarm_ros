"""Candidate cube stability tracking for pick-and-place sorting."""

from .localized_cube import LocalizedCube


class CandidateTracker:
    """Track stability of a candidate cube across video frames."""

    def __init__(self, stable_frames_required: int, tolerance_px: float = 25.0) -> None:
        """
        Initialize the tracker.

        :param stable_frames_required: Consecutive frames a cube must be seen to confirm.
        :param tolerance_px: Maximum per-axis center distance in pixels to match.
        """
        self.stable_frames_required = stable_frames_required
        self.tolerance_px = tolerance_px
        self.candidate: LocalizedCube | None = None
        self.frame_count = 0

    def is_match(self, a: LocalizedCube, b: LocalizedCube) -> bool:
        """Return True if both cubes share a class and are within tolerance."""
        (ax, ay), (bx, by) = a.detection.center_px, b.detection.center_px
        return (
            a.detection.class_id == b.detection.class_id
            and abs(ax - bx) < self.tolerance_px
            and abs(ay - by) < self.tolerance_px
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

        match = None
        if self.candidate is not None:
            match = next((c for c in cubes if self.is_match(self.candidate, c)), None)

        if match is not None:
            self.candidate = match
            self.frame_count += 1
        else:
            self.candidate = cubes[0]
            self.frame_count = 1

        return self.candidate if self.frame_count >= self.stable_frames_required else None

    def reset(self) -> None:
        """Clear the current candidate."""
        self.candidate = None
        self.frame_count = 0
