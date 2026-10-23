import numpy as np
from dataclasses import dataclass, field
from typing import Union

from src.parameters import RobotParameters, SimulationParameters
from src.path.CirclePath import CirclePath
from src.path.LinePath import LinePath
from src.robot.arm_2dof.kinematics import get_jacobian

@dataclass
class metrics:
    simulation_logs: list
    sim_params: SimulationParameters
    path_obj: Union[LinePath, CirclePath]
    robot_params: RobotParameters
    
    rmse: float = field(init=False)
    errors: np.ndarray = field(init=False)
    distance_errors: np.ndarray = field(init=False)
    convergence_time: float = field(init=False)
    manipulability: np.ndarray = field(init=False)

    def __post_init__(self):
        if not self.simulation_logs:
            raise ValueError("simulation_logs must contain at least one result.")

        x_log = np.asarray([result.x for result in self.simulation_logs], dtype=float)
        q_log = np.asarray([result.q for result in self.simulation_logs], dtype=float)

        if isinstance(self.path_obj, CirclePath):
            self.errors = (
                np.linalg.norm(x_log - self.path_obj.center, axis=1)
                - self.path_obj.radius
            )
            self.distance_errors = np.abs(self.errors)
            self.convergence_time = get_convergence_time_from_errors(
                self.distance_errors, self.sim_params.dt
            )
        else:
            self.errors = self.path_obj.calculate_path_errors(x_log)
            self.distance_errors = np.linalg.norm(
                x_log - self.path_obj.end_point, axis=1
            )
            self.convergence_time = get_ee_convergence_time(
                x_log, self.path_obj.end_point, self.sim_params.dt
            )

        self.rmse = float(np.sqrt(np.mean(self.errors**2)))

        manipulability = []
        for q in q_log:
            jacobian = get_jacobian(
                q[0], q[1], self.robot_params.L1, self.robot_params.L2
            )
            manipulability.append(
                np.sqrt(max(0.0, np.linalg.det(jacobian @ jacobian.T)))
            )
        self.manipulability = np.asarray(manipulability)

# Define convergence time function for end-effector position
def get_ee_convergence_time(ee_x_log: np.ndarray, target_x: np.ndarray, dt: float, error_threshold = 0.01):

    dist_errors = np.linalg.norm(np.array(ee_x_log) - target_x, axis=1)
    return get_convergence_time_from_errors(dist_errors, dt, error_threshold)


def get_convergence_time_from_errors(errors: np.ndarray, dt: float, error_threshold = 0.01):

    last_exceeding_idx = -1
    for i in range(len(errors) - 1, -1, -1):
        if errors[i] >= error_threshold:
            last_exceeding_idx = i
            break

    return (last_exceeding_idx + 1) * dt


def calculate_linepath_rmse(x_log: np.ndarray, linepath_obj: LinePath):
    
    errors = linepath_obj.calculate_path_errors(x_log)
    
    return np.sqrt(np.mean(errors**2))
