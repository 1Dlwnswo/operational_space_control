# @title
import numpy as np
import mujoco
from src.dataclasses import RobotParameters, ImpedanceControlParameters, SimulationResult
from src.mujoco_setup import get_jacobian, get_J_dot, get_end_effector_position, calcuate_inertia_matrix, calculate_coriolis_centrifugal_torques, calculate_gravity_torques
from simulation.acceleration_command import path_following, impedance_control


def perform_simulation_step(model, data, robot_params: RobotParameters, impedance_control_params: ImpedanceControlParameters, \
                            control_type: str, path_obj, max_acceleration, step_index: int = 0, save_operational_space = False):

    q = data.qpos.copy()
    qdot = data.qvel.copy()
    qddot = data.qacc.copy()

    q1, q2 = q
    q1_dot, q2_dot = qdot

    # Forward kinematics
    J = get_jacobian(q1, q2, robot_params.L1, robot_params.L2)
    Jdot = get_J_dot(q1, q2, q1_dot, q2_dot, robot_params.L1, robot_params.L2)

    x = get_end_effector_position(q1, q2, robot_params.L1, robot_params.L2) # (2,)
    xdot = J @ qdot # (2,2) @ (2,) -> (2,)
    xddot = Jdot @ qdot + J @ qddot # (2,2) @ (2,) + (2,2) @ (2,) -> (2,)

    # Dynamics
    M = calcuate_inertia_matrix(q1, q2, robot_params.m1, robot_params.m2, robot_params.I1, robot_params.I2, robot_params.c1, robot_params.c2, robot_params.L1, robot_params.L2)
    C = calculate_coriolis_centrifugal_torques(q1, q2, q1_dot, q2_dot, robot_params.m2, robot_params.c2, robot_params.L1)
    G = calculate_gravity_torques(q1, q2, robot_params.m1, robot_params.m2, robot_params.c1, robot_params.c2, robot_params.L1, robot_params.g)
    M_inv = np.linalg.inv(M)

    # Operational Space Inertia
    lambda_damping = 0.0001
    Lambda_inv = J @ M_inv @ J.T
    Lambda = np.linalg.inv(Lambda_inv + lambda_damping * np.eye(2))

    # Operational Space Compensation - convert qdot, C, G to column vectors
    term_for_mu = -Jdot @ qdot.reshape(-1, 1) + J @ M_inv @ C.reshape(-1, 1) # (2,1)
    Mu = Lambda @ term_for_mu # (2,1)

    term_for_p = J @ M_inv @ G.reshape(-1, 1) # (2,1)
    P = Lambda @ term_for_p # (2,1)

    if control_type == 'imp':
      a_cmd = impedance_control(x, xdot, path_obj, impedance_control_params)
    elif control_type == 'pw':
      a_cmd = path_following(x, xdot, path_obj, impedance_control_params, max_acceleration, step_index=step_index)
    else:
      raise ValueError(f"Unknown control_type: {control_type}")

    # Desired Operational Force - F is now (2,1)
    F = Lambda @ a_cmd + Mu + P # Lambda(2,2) @ a(2,1) -> (2,1) + Mu(2,1) + P(2,1) -> (2,1)

    # Joint Torque - tau is (2,1), then squeeze to (2,)
    tau = (J.T @ F).squeeze() # J.T(2,2) @ F(2,1) -> (2,1) then squeeze -> (2,)

    tau = np.clip(tau, -impedance_control_params.max_torque, impedance_control_params.max_torque)

    data.ctrl[:] = tau

    mujoco.mj_step(model, data)

    if save_operational_space:
        return SimulationResult(q, qdot, qddot, tau, a_cmd, True, x, xdot, xddot)
    return SimulationResult(q, qdot, qddot, tau, a_cmd)