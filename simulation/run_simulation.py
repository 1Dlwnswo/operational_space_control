import numpy as np
import mujoco
import mujoco.viewer
import time
from src.parameters import RobotParameters, ImpedanceControlParameters,SimulationParameters
from simulation.perform_simulation_step import perform_simulation_step
from src.mujoco_setup import get_robot_parameters


# ============================================
# Define the simulation function
# ============================================
def run_simulation(model, data, q_init: np.ndarray, \
                   control_type: str, sim_params: SimulationParameters, \
                   path_obj, max_acceleration = 10, save_operational_space = False, use_viewer = False):
    # Reset initial condition for each simulation
    data.qpos[0] = q_init[0]
    data.qpos[1] = q_init[1]
    data.qvel[0] = 0
    data.qvel[1] = 0

    # Reset simulation time and derived quantities
    data.time = 0.0
    mujoco.mj_forward(model, data)
    
    # Operational Space Parameters
    Kp_osc_sim = (0.25 * max_acceleration**2) * np.eye(2)
    Kd_osc_sim = max_acceleration * np.eye(2)
    max_torque_osc = 1000 # Maximum torque for OSC
    
    robot_params = get_robot_parameters(model)
    impedance_control_params = ImpedanceControlParameters(Kp=Kp_osc_sim, Kd=Kd_osc_sim, max_torque=max_torque_osc)

    simulation_log = []
    
    def simulation_loop(viewer=None):
        timestep = sim_params.dt
        
        for i in range(sim_params.num_steps):
            
            start_time = time.perf_counter() if viewer is not None else None
            
            sim_results = perform_simulation_step(model, data, robot_params, impedance_control_params, \
                                                  control_type, path_obj, max_acceleration, step_index=i, save_operational_space = save_operational_space, viewer=viewer)
            simulation_log.append(sim_results)
            
            if viewer is not None:
                elapsed = time.perf_counter() - start_time
                remaining = timestep - elapsed

                if remaining > 0:
                    time.sleep(remaining)
    
    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            
            viewer.cam.distance = 5.0
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]

            viewer.cam.azimuth = 90
            viewer.cam.elevation = 0
                        
            simulation_loop(viewer)
    else:
        simulation_loop()
    
    return simulation_log