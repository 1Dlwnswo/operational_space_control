import numpy as np
import mujoco
import mujoco.viewer
import time

from src.parameters import ImpedanceControlParameters,SimulationParameters
from simulation.perform_simulation_step import perform_simulation_step
from src.mujoco_setup import get_robot_parameters
from src.utils import save_config as save_simulation_config


# ============================================
# Define the simulation function
# ============================================
def run_simulation(model, data, q_init: np.ndarray, control_type: str, sim_params: SimulationParameters,
                   path_obj, max_acceleration = 10, save_operational_space = False, use_viewer = False, experiment_dir = None):
    # Reset initial condition for each simulation
    data.qpos[0] = q_init[0]
    data.qpos[1] = q_init[1]
    data.qvel[0] = 0
    data.qvel[1] = 0

    # Reset simulation time and derived quantities
    data.time = 0.0
    mujoco.mj_forward(model, data)
    
    # Get robot parameters for kinematics calculation for control
    robot_params = get_robot_parameters(model)
    
    # Operational Space Parameters
    impedance_control_params = ImpedanceControlParameters(max_acceleration)
    
    if experiment_dir is not None:
        save_simulation_config(
            experiment_dir=experiment_dir,
            sim_params=sim_params,
            control_params=impedance_control_params,
            robot_params=robot_params,
        )

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
