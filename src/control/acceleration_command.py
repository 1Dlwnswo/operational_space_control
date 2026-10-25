import numpy as np

from src.parameters import ImpedanceControlParameters


def path_following(
    x,
    xdot,
    path_obj,
    impedance_control_params: ImpedanceControlParameters,
    max_acceleration,
    step_index: int = 0,
):
    x_des, x_des_dot, x_des_ddot = path_obj.compute_reference(x, xdot, max_acceleration, step_index)

    e = (x_des - x).reshape(-1, 1)  # (2,1)
    edot = (x_des_dot - xdot).reshape(-1, 1)  # (2,1)
    x_des_ddot = x_des_ddot.reshape(-1, 1)  # (2,1)

    a_cmd = x_des_ddot + impedance_control_params.Kp @ e + impedance_control_params.Kd @ edot

    # Clip acceleration if its magnitude is greater than max_acceleration
    a_magnitude = np.linalg.norm(a_cmd)

    if a_magnitude > max_acceleration:
        a_cmd = a_cmd * (max_acceleration / a_magnitude)

    return a_cmd


def impedance_control(x, xdot, path_obj, impedance_control_params: ImpedanceControlParameters):

    x_des = path_obj.end_point  # (2,)
    x_des_dot = np.zeros(2)

    e = (x_des - x).reshape(-1, 1)  # (2,1)
    edot = (x_des_dot - xdot).reshape(-1, 1)  # (2,1) - changed to use x_des_dot
    a_cmd = impedance_control_params.Kp @ e + impedance_control_params.Kd @ edot  # Kp(2,2) @ e(2,1) -> (2,1) ; Kd(2,2) @ edot(2,1) -> (2,1) ;

    return a_cmd
