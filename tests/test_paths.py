import numpy as np
import pytest

from src.path.CirclePath import CirclePath
from src.path.LinePath import LinePath
from src.path.PolygonPath import PolygonPath


def test_circle_path_reference_has_expected_position_velocity_and_acceleration():
    path = CirclePath(
        center=np.array([1.0, 2.0]), radius=2.0, cruise_angular_speed=0.5
    )

    position, velocity, acceleration = path.compute_reference(
        np.array([3.0, 2.0]), np.zeros(2), max_aceleration=5.0
    )

    np.testing.assert_allclose(position, [3.0, 2.0])
    np.testing.assert_allclose(velocity, [0.0, 1.0])
    np.testing.assert_allclose(acceleration, [-0.5, 0.0])


def test_circle_path_at_center_uses_start_angle_without_nan():
    path = CirclePath(
        center=np.zeros(2), radius=1.5, start_angle=np.pi / 2
    )

    reference = path.compute_reference(np.zeros(2), np.zeros(2), 5.0)

    assert all(np.all(np.isfinite(value)) for value in reference)
    np.testing.assert_allclose(reference[0], [0.0, 1.5], atol=1e-12)


def test_line_path_projects_position_and_brakes_at_end():
    path = LinePath(np.array([0.0, 0.0]), np.array([2.0, 0.0]), cruise_speed=1.0)

    before_end = path.compute_reference(np.array([1.9, 0.2]), np.zeros(2), 2.0)
    at_end = path.compute_reference(np.array([2.0, 0.0]), np.zeros(2), 2.0)

    np.testing.assert_allclose(before_end[0], [2.0, 0.0])
    np.testing.assert_allclose(at_end[0], [2.0, 0.0])
    np.testing.assert_allclose(at_end[1], np.zeros(2))
    np.testing.assert_allclose(at_end[2], np.zeros(2))


def test_line_path_handles_zero_length_segment():
    point = np.array([0.5, -0.25])
    path = LinePath(point, point.copy())

    reference = path.compute_reference(np.zeros(2), np.zeros(2), 2.0)

    assert all(np.all(np.isfinite(value)) for value in reference)
    np.testing.assert_allclose(reference[0], point)


def test_polygon_path_requires_at_least_two_vertices():
    with pytest.raises(ValueError, match="at least two vertices"):
        PolygonPath(np.array([[0.0, 0.0]]))


def test_closed_polygon_path_wraps_to_first_segment():
    path = PolygonPath(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        desired_speed=0.5,
        stop_at_end=False,
    )
    path.current_segment_idx = len(path._segments) - 1

    path.compute_reference(np.array([0.0, 0.0]), np.zeros(2), 2.0)

    assert path.current_segment_idx == 0
