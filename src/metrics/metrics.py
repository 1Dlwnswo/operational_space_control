import numpy as np
from dataclasses import dataclass, field

from src.parameters import RobotParameters, SimulationParameters
from src.path.LinePath import LinePath
from src.robot.arm_2dof.kinematics import get_jacobian
from src.utils import get_ee_convergence_time

@dataclass
class metrics:
    simulation_logs: list
    sim_params: SimulationParameters
    path_obj: LinePath
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

        self.errors = self.path_obj.calculate_path_errors(x_log)
        self.rmse = float(np.sqrt(np.mean(self.errors**2)))
        
        self.distance_errors = np.linalg.norm(
            x_log - self.path_obj.end_point, axis=1
        )
        
        self.convergence_time = get_ee_convergence_time(
            x_log, self.path_obj.end_point, self.sim_params.dt
        )

        manipulability = []
        for q in q_log:
            jacobian = get_jacobian(
                q[0], q[1], self.robot_params.L1, self.robot_params.L2
            )
            manipulability.append(
                np.sqrt(max(0.0, np.linalg.det(jacobian @ jacobian.T)))
            )
        self.manipulability = np.asarray(manipulability)

def calculate_linepath_rmse(x_log: np.ndarray, linepath_obj: LinePath):
    
    errors = linepath_obj.calculate_path_errors(x_log)
    
    return np.sqrt(np.mean(errors**2))
