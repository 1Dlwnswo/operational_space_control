import numpy as np

from src.control.acceleration_command import impedance_control, path_following
from src.control.osc import compute_osc_torque, get_osc_parameters
from src.control.torque_limiter import limit_torque
from src.parameters import ImpedanceControlParameters, RobotParameters
from src.path.LinePath import LinePath


def test_path_following_returns_column_vector_and_respects_acceleration_limit():
    params = ImpedanceControlParameters(
        Kp=100.0 * np.eye(2), Kd=10.0 * np.eye(2), max_torque=20.0
    )
    path = LinePath(np.zeros(2), np.array([1.0, 1.0]), cruise_speed=0.0)

    acceleration = path_following(
        np.array([-1.0, -1.0]), np.zeros(2), path, params, max_acceleration=3.0
    )

    assert acceleration.shape == (2, 1)
    np.testing.assert_allclose(np.linalg.norm(acceleration), 3.0)


def test_impedance_control_returns_pd_acceleration_command():
    params = ImpedanceControlParameters(
        Kp=2.0 * np.eye(2), Kd=3.0 * np.eye(2), max_torque=20.0
    )
    path = LinePath(np.zeros(2), np.array([1.0, 2.0]))

    command = impedance_control(np.zeros(2), np.array([0.5, 1.0]), path, params)

    np.testing.assert_allclose(command, [[0.5], [1.0]])


def test_osc_and_joint_torque_have_finite_expected_shapes():
    q = np.array([0.4, 0.7])
    qdot = np.array([0.1, -0.2])
    params = RobotParameters(1.0, 0.8, 0.1, 0.08, 1.0, 0.8, 0.5, 0.4, 9.81)

    operational_inertia, coriolis, gravity = get_osc_parameters(q, qdot, params)
    torque = compute_osc_torque(q, qdot, np.array([[1.0], [-0.5]]), params)

    assert operational_inertia.shape == (2, 2)
    assert coriolis.shape == gravity.shape == (2, 1)
    assert torque.shape == (2,)
    assert np.all(np.isfinite(torque))


def test_torque_limiter_caps_values_and_preserves_direction():
    torque = np.array([8.0, 10.0])

    limited = limit_torque(torque, max_torque=5.0)

    assert np.max(np.abs(limited)) <= 5.0
    np.testing.assert_allclose(
        limited / np.linalg.norm(limited),
        np.array([8.0, 10.0]) / np.linalg.norm([8.0, 10.0]),
    )


def test_torque_limiter_leaves_in_range_torque_unchanged():
    torque = np.array([2.0, -3.0])

    limited = limit_torque(torque.copy(), max_torque=5.0)

    np.testing.assert_array_equal(limited, torque)
