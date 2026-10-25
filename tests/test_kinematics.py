import numpy as np

from src.robot.arm_2dof.kinematics import (
    get_end_effector_position,
    get_J_dot,
    get_jacobian,
    inverse_kinematics,
)


def test_forward_kinematics_and_jacobian_match_finite_difference():
    q = np.array([0.4, -0.7])
    lengths = (1.2, 0.8)
    eps = 1e-6

    numerical_jacobian = np.column_stack(
        [
            (
                get_end_effector_position(*(q + np.eye(2)[i] * eps), *lengths)
                - get_end_effector_position(*(q - np.eye(2)[i] * eps), *lengths)
            )
            / (2 * eps)
            for i in range(2)
        ]
    )

    np.testing.assert_allclose(
        get_jacobian(*q, *lengths), numerical_jacobian, atol=1e-8
    )


def test_jacobian_derivative_matches_finite_difference():
    q = np.array([0.4, -0.7])
    qdot = np.array([0.3, -0.2])
    lengths = (1.2, 0.8)
    eps = 1e-6

    numerical_derivative = (
        get_jacobian(*(q + qdot * eps), *lengths)
        - get_jacobian(*(q - qdot * eps), *lengths)
    ) / (2 * eps)

    np.testing.assert_allclose(
        get_J_dot(*q, *qdot, *lengths), numerical_derivative, atol=1e-8
    )


def test_inverse_kinematics_round_trip():
    lengths = (1.0, 0.7)
    expected_q = np.array([0.3, 0.8])
    position = get_end_effector_position(*expected_q, *lengths)

    actual_q = inverse_kinematics(position, *lengths, q_current=expected_q)

    np.testing.assert_allclose(
        get_end_effector_position(*actual_q, *lengths), position, atol=1e-10
    )


def test_inverse_kinematics_clips_unreachable_target_to_maximum_reach():
    lengths = (1.0, 0.7)
    target = np.array([10.0, 0.0])

    q = inverse_kinematics(target, *lengths, q_current=None)

    np.testing.assert_allclose(
        np.linalg.norm(get_end_effector_position(*q, *lengths)),
        sum(lengths),
        atol=1e-10,
    )
