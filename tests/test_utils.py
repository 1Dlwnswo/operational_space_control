import numpy as np

from src.utils import get_ee_convergence_time, wrap_to_pi


def test_wrap_to_pi_returns_angles_in_expected_interval():
    wrapped = wrap_to_pi(np.array([-3 * np.pi, -np.pi / 2, 3 * np.pi]))

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    np.testing.assert_allclose(wrapped, [-np.pi, -np.pi / 2, -np.pi])


def test_convergence_time_is_after_last_sample_outside_threshold():
    positions = np.array(
        [[0.0, 0.0], [0.5, 0.0], [0.95, 0.0], [1.02, 0.0], [1.0, 0.0]]
    )

    convergence_time = get_ee_convergence_time(
        positions, target_x=np.array([1.0, 0.0]), dt=0.01, error_threshold=0.1
    )

    assert convergence_time == 0.02


def test_convergence_time_is_zero_when_always_within_threshold():
    positions = np.array([[1.0, 0.0], [1.005, 0.0]])

    assert get_ee_convergence_time(positions, np.array([1.0, 0.0]), 0.01) == 0.0
