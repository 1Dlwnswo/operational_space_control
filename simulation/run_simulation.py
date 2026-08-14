import numpy as np
from src.parameters import RobotParameters, ImpedanceControlParameters,SimulationParameters
from src.mujoco_setup import get_end_effector_position
from simulation.perform_simulation_step import perform_simulation_step

# ============================================
# Define the simulation function
# ============================================
def run_simulation(model, data, q_init: np.ndarray, \
                   control_type: str, robot_params: RobotParameters, sim_params: SimulationParameters, \
                   path_obj, max_acceleration = 10, save_operational_space = False):
    # Reset initial condition for each simulation
    data.qpos[0] = q_init[0]
    data.qpos[1] = q_init[1]
    data.qvel[0] = 0
    data.qvel[1] = 0

    simulation_log = []

    # Initialize end-effector position

    ee_start_pos = get_end_effector_position(q_init[0], q_init[1], robot_params.L1, robot_params.L2)

    # Operational Space Parameters
    Kp_osc_sim = (0.25 * max_acceleration**2) * np.eye(2)
    Kd_osc_sim = max_acceleration * np.eye(2)
    max_torque_osc = 1000 # Maximum torque for OSC
    impedance_control_params = ImpedanceControlParameters(Kp=Kp_osc_sim, Kd=Kd_osc_sim, max_torque=max_torque_osc)

    for i in range(sim_params.num_steps):
        sim_results = perform_simulation_step(model, data, robot_params, impedance_control_params, \
                                              control_type, path_obj, max_acceleration, step_index=i, save_operational_space = save_operational_space)
        simulation_log.append(sim_results)

    return simulation_log