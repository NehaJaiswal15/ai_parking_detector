"""
Tests for utility functions in utils.py.
These tests validate core logic (scaling, polygon occupancy, label rendering)
without requiring a YOLO model or video file.
"""

import numpy as np
import cv2
import pytest

from src.utils import scale_polygons, drawPolygons, label_detection


class TestScalePolygons:
    """Tests for the scale_polygons function."""

    def test_no_scaling_when_same_resolution(self):
        """Polygons should remain unchanged if reference and current resolution match."""
        polys = [[(0, 0), (100, 0), (100, 100), (0, 100)]]
        ref_size = (1920, 1080)     # (width, height)
        cur_size = (1080, 1920)     # (height, width) — OpenCV convention

        result = scale_polygons(polys, ref_size, cur_size)

        assert result == polys

    def test_scaling_doubles_coordinates(self):
        """Doubling the resolution should double all polygon coordinates."""
        polys = [[(10, 20), (30, 40)]]
        ref_size = (100, 100)       # reference: 100x100
        cur_size = (200, 200)       # current: 200x200 (h, w)

        result = scale_polygons(polys, ref_size, cur_size)

        assert result == [[(20, 40), (60, 80)]]

    def test_scaling_halves_coordinates(self):
        """Halving the resolution should halve all polygon coordinates."""
        polys = [[(100, 200), (300, 400)]]
        ref_size = (1000, 1000)
        cur_size = (500, 500)       # (h, w)

        result = scale_polygons(polys, ref_size, cur_size)

        assert result == [[(50, 100), (150, 200)]]

    def test_empty_polygon_list(self):
        """Should return an empty list for empty input."""
        result = scale_polygons([], (1920, 1080), (1080, 1920))
        assert result == []

    def test_multiple_polygons(self):
        """Should scale all polygons in the list."""
        polys = [
            [(0, 0), (10, 10)],
            [(20, 20), (30, 30)],
        ]
        ref_size = (100, 100)
        cur_size = (200, 200)       # double

        result = scale_polygons(polys, ref_size, cur_size)

        assert len(result) == 2
        assert result[0] == [(0, 0), (20, 20)]
        assert result[1] == [(40, 40), (60, 60)]


class TestDrawPolygons:
    """Tests for the drawPolygons function."""

    def _make_frame(self, width: int = 200, height: int = 200) -> np.ndarray:
        """Create a blank black frame for testing."""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def test_no_detections_all_available(self):
        """With no detection points, all slots should be available."""
        frame = self._make_frame()
        polys = [[(10, 10), (50, 10), (50, 50), (10, 50)]]

        result_frame, occupied, status = drawPolygons(frame, polys, detection_points=[])

        assert occupied == 0
        assert status == [False]
        assert result_frame.shape == frame.shape

    def test_detection_inside_polygon_is_occupied(self):
        """A detection center inside a polygon should mark it occupied."""
        frame = self._make_frame()
        polys = [[(10, 10), (90, 10), (90, 90), (10, 90)]]
        detection_points = [(50, 50)]  # center of polygon

        _, occupied, status = drawPolygons(frame, polys, detection_points=detection_points)

        assert occupied == 1
        assert status == [True]

    def test_detection_outside_polygon_is_available(self):
        """A detection center outside all polygons should not occupy any slot."""
        frame = self._make_frame()
        polys = [[(10, 10), (30, 10), (30, 30), (10, 30)]]
        detection_points = [(150, 150)]  # far outside

        _, occupied, status = drawPolygons(frame, polys, detection_points=detection_points)

        assert occupied == 0
        assert status == [False]

    def test_multiple_slots_mixed_status(self):
        """Test with 2 slots: one occupied, one available."""
        frame = self._make_frame(400, 200)
        poly_a = [(10, 10), (90, 10), (90, 90), (10, 90)]
        poly_b = [(200, 10), (290, 10), (290, 90), (200, 90)]
        polys = [poly_a, poly_b]

        # Detection inside poly_a only
        detection_points = [(50, 50)]

        _, occupied, status = drawPolygons(frame, polys, detection_points=detection_points)

        assert occupied == 1
        assert status == [True, False]

    def test_none_detection_points(self):
        """Should handle None detection_points gracefully."""
        frame = self._make_frame()
        polys = [[(10, 10), (50, 10), (50, 50), (10, 50)]]

        _, occupied, status = drawPolygons(frame, polys, detection_points=None)

        assert occupied == 0
        assert status == [False]

    def test_returns_modified_frame(self):
        """The returned frame should be different from the input (overlays drawn)."""
        frame = self._make_frame()
        polys = [[(10, 10), (50, 10), (50, 50), (10, 50)]]

        result_frame, _, _ = drawPolygons(frame, polys, detection_points=[])

        # The overlay blending should change at least some pixels
        assert not np.array_equal(result_frame, np.zeros_like(result_frame))


class TestLabelDetection:
    """Tests for the label_detection function."""

    def test_modifies_frame_in_place(self):
        """label_detection should draw on the frame (mutate it)."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        original = frame.copy()

        label_detection(frame, text="car", x1=10, y1=50, x2=100, y2=150)

        # Frame should be modified
        assert not np.array_equal(frame, original)

    def test_does_not_crash_with_edge_coordinates(self):
        """Should handle bounding boxes at frame edges without errors."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Box at the very edge of the frame
        label_detection(frame, text="truck", x1=0, y1=0, x2=99, y2=99)

        # If we got here without an exception, the test passes
        assert True
