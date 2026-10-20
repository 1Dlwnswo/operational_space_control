# @title
import mujoco
from src.parameters import RobotParameters, ImpedanceControlParameters, SimulationResult
from src.robot.arm_2dof.kinematics import get_jacobian, get_J_dot, get_end_effector_position
from src.control.osc import compute_osc_torque
from src.control.torque_limiter import limit_torque
from src.control.acceleration_command import path_following, impedance_control
from src.robot.arm_2dof.sensors import read_sensors


def perform_simulation_step(model, data, robot_params: RobotParameters, impedance_control_params: ImpedanceControlParameters, \
                            control_type: str, path_obj, max_acceleration, step_index: int = 0, save_operational_space = False, viewer = None):
    # Read joint positions and velocities from sensors.
    q, qdot = read_sensors(model, data)
    
    q1, q2 = q
    q1_dot, q2_dot = qdot

    J = get_jacobian(q1, q2, robot_params.L1, robot_params.L2)
    Jdot = get_J_dot(q1, q2, q1_dot, q2_dot, robot_params.L1, robot_params.L2)
    
    x = get_end_effector_position(q1, q2, robot_params.L1, robot_params.L2) # (2,)
    xdot = J @ qdot # (2,2) @ (2,) -> (2,)
    
    # Depending on the control type ('imp' for impedance control or 'pw' for path following),
    # compute the desired acceleration command (a_cmd) for the end-effector.
    if control_type == 'imp':
      a_cmd = impedance_control(x, xdot, path_obj, impedance_control_params)
    elif control_type == 'pw':
      a_cmd = path_following(x, xdot, path_obj, impedance_control_params, max_acceleration, step_index=step_index)
    else:
      raise ValueError(f"Unknown control_type: {control_type}")

    # Calculate desired force F
    tau = compute_osc_torque(q, qdot, a_cmd, robot_params)

    # Apply torque limits to ensure the computed torque does not exceed the maximum allowable torque.
    tau = limit_torque(tau, max_torque = impedance_control_params.max_torque)

    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

    if viewer is not None:
        viewer.sync()

    if save_operational_space:
        qddot = data.qacc.copy()
        xddot = Jdot @ qdot + J @ qddot # (2,2) @ (2,) + (2,2) @ (2,) -> (2,)
        return SimulationResult(q, qdot, tau, a_cmd, True, qddot, x, xdot, xddot)
    
    return SimulationResult(q, qdot, tau, a_cmd)