import numpy as np

from src.robot.arm_2dof.dynamics import (
    calcuate_inertia_matrix,
    calculate_coriolis_centrifugal_torques,
    calculate_gravity_torques,
)
from src.robot.arm_2dof.kinematics import get_J_dot, get_jacobian


def get_osc_parameters(q, qdot, robot_params):
    q1, q2 = q
    q1_dot, q2_dot = qdot

    # Forward kinematics
    J = get_jacobian(q1, q2, robot_params.L1, robot_params.L2)
    Jdot = get_J_dot(q1, q2, q1_dot, q2_dot, robot_params.L1, robot_params.L2)

    # Dynamics
    M = calcuate_inertia_matrix(
        q1,
        q2,
        robot_params.m1,
        robot_params.m2,
        robot_params.I1,
        robot_params.I2,
        robot_params.c1,
        robot_params.c2,
        robot_params.L1,
        robot_params.L2,
    )
    C = calculate_coriolis_centrifugal_torques(q1, q2, q1_dot, q2_dot, robot_params.m2, robot_params.c2, robot_params.L1)
    G = calculate_gravity_torques(
        q1,
        q2,
        robot_params.m1,
        robot_params.m2,
        robot_params.c1,
        robot_params.c2,
        robot_params.L1,
        robot_params.g,
    )
    M_inv = np.linalg.inv(M)

    # Operational Space Inertia
    lambda_damping = 0.0001
    Lambda_inv = J @ M_inv @ J.T
    Lambda = np.linalg.inv(Lambda_inv + lambda_damping * np.eye(2))

    # Operational Space Compensation - convert qdot, C, G to column vectors
    term_for_mu = -Jdot @ qdot.reshape(-1, 1) + J @ M_inv @ C.reshape(-1, 1)  # (2,1)
    Mu = Lambda @ term_for_mu  # (2,1)

    term_for_p = J @ M_inv @ G.reshape(-1, 1)  # (2,1)
    P = Lambda @ term_for_p  # (2,1)

    return Lambda, Mu, P


def compute_osc_torque(q, qdot, a_cmd, robot_params):
    q1, q2 = q
    Lambda, Mu, P = get_osc_parameters(q, qdot, robot_params)
    F = Lambda @ a_cmd + Mu + P  # Lambda(2,2) @ a(2,1) -> (2,1) + Mu(2,1) + P(2,1) -> (2,1)

    J = get_jacobian(q1, q2, robot_params.L1, robot_params.L2)
    tau = (J.T @ F).squeeze()  # J.T(2,2) @ F(2,1) -> (2,1) then squeeze -> (2,)

    return tau
