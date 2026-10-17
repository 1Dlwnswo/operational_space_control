import numpy as np
from src.parameters import RobotParameters, SimulationParameters
from src.mujoco_setup import model, data, dt, L1, L2, m1, m2, c1, c2, g, I1, I2, get_end_effector_position, get_jacobian, get_ee_convergence_time
from src.path.LinePath import LinePath
from simulation.run_simulation import run_simulation
from experiment.LinePath.generate_grid import uniform_grid
from experiment.LinePath.plot_combined_results import plot_combined_results
from experiment.gif_generation import generate_robot_arm_full_trajectory_gif
from typing import Optional

# ============================================
# Run simulation for multiple target positions
# ============================================
def run_experiment(q_init: np.ndarray, target_positions: np.ndarray, robot_params: Optional[RobotParameters] = None, \
    cruise_speed = 2.0, max_acceleration = 10.0, verbose = False):

    if robot_params is None:
        robot_params = RobotParameters(L1=L1, L2=L2, m1=m1, m2=m2, c1=c1, c2=c2, g=g, I1=I1, I2=I2)

    # Define simulation parameters based on cruise speed and max acceleration
    sim_params = SimulationParameters(dt=dt, num_steps=int(4000/cruise_speed + 2000*cruise_speed/max_acceleration))

    # Initialize logs for all simulations
    simulation_logs = []

    if verbose:
        print(f"Number of OSC simulations: {len(target_positions)}")

    # Run simulation for each target position
    for idx, (target_x, target_z) in enumerate(target_positions):
        ee_start_pos = get_end_effector_position(q_init[0], q_init[1], robot_params.L1, robot_params.L2)
        
        path_obj = LinePath(start_point=ee_start_pos, end_point=np.array([target_x, target_z]), cruise_speed=cruise_speed)

        current_simulation_log = run_simulation(model, data, q_init, "pw", robot_params, sim_params, path_obj, max_acceleration = max_acceleration, save_operational_space=True)

        simulation_logs.append(current_simulation_log)
        
    if verbose:
        print("Simulation completed for all target positions.")
    
    return simulation_logs, sim_params


# ============================================
# Process simulation logs to extract relevant data for plotting
# ============================================
def process_simulation_logs(simulation_logs: list, sim_params: SimulationParameters, target_positions: np.ndarray, \
    robot_params = RobotParameters(L1=L1, L2=L2, m1=m1, m2=m2, c1=c1, c2=c2, g=g, I1=I1, I2=I2)):
    
    osc_all_q_logs = []
    osc_all_torque_logs = []
    osc_all_a_cmd_logs = []
    osc_all_xdot_logs = []
    osc_all_xddot_logs = []
    osc_all_dist_logs = []
    osc_ee_convergence_times = []
    osc_all_manipulability_logs = []
    osc_all_x_logs = []

    for idx, log in enumerate(simulation_logs):
        # Extract data from the simulation log
        q_log = np.array([sim_result.q for sim_result in log])
        torque_log = np.array([sim_result.tau for sim_result in log])
        a_cmd_log = np.array([sim_result.a_cmd for sim_result in log])
        
        x_log = np.array([sim_result.x for sim_result in log])
        xdot_log = np.array([sim_result.xdot for sim_result in log])
        xddot_log = np.array([sim_result.xddot for sim_result in log])
        
        osc_all_q_logs.append(q_log)
        osc_all_torque_logs.append(torque_log)
        osc_all_a_cmd_logs.append(a_cmd_log)
        osc_all_x_logs.append(x_log)
        osc_all_xdot_logs.append(xdot_log)
        osc_all_xddot_logs.append(xddot_log)

        # Calculate distance error over time
        dist_log_current_sim = np.linalg.norm(np.array(x_log) - np.array(target_positions[idx]), axis=1)
        osc_all_dist_logs.append(dist_log_current_sim)

        # Calculate convergence time
        conv_time_ee = get_ee_convergence_time(np.array(x_log), np.array(target_positions[idx]), sim_params.dt)
        osc_ee_convergence_times.append(conv_time_ee)  
        
        # Calulate Manipulability over time
        current_sim_manipulability = []
        for q1_k, q2_k in q_log:
            J_k = get_jacobian(q1_k, q2_k, robot_params.L1, robot_params.L2)
            manipulability_k = np.linalg.det(J_k @ J_k.T)
            current_sim_manipulability.append(manipulability_k)
        osc_all_manipulability_logs.append(current_sim_manipulability)
    
    return (osc_all_q_logs, osc_all_torque_logs, osc_all_a_cmd_logs, 
            osc_all_x_logs, osc_all_xdot_logs, osc_all_xddot_logs, 
            osc_all_dist_logs, osc_ee_convergence_times, osc_all_manipulability_logs)


# ============================================
# Plot combined results for all target positions
# ============================================
def plot_results_for_all_targets(simulation_logs: list, sim_params: SimulationParameters, q_init: np.ndarray, target_positions: np.ndarray, robot_params: RobotParameters):
    
    processed_results = process_simulation_logs(simulation_logs, sim_params, target_positions, robot_params)
    plot_combined_results(processed_results, target_positions, sim_params, robot_params, q_init)


# ============================================
# Generate GIF for the full trajectory across all target positions
# ============================================
def generate_gif_for_full_trajectory(simulation_logs: list, sim_params: SimulationParameters, robot_params: RobotParameters, target_positions: np.ndarray):
    
    concatenated_q1_log = []
    concatenated_q2_log = []
    concatenated_target_x_log = []
    concatenated_target_z_log = []

    for log, (target_x, target_z) in zip(simulation_logs, target_positions):
        q_log = np.array([sim_result.q for sim_result in log])
        
        # Extend with the actual logged joint positions for this target
        concatenated_q2_log.extend(q_log[:,1])
        concatenated_q1_log.extend(q_log[:,0])
        
        # Repeat the current target for the duration of this simulation segment
        concatenated_target_x_log.extend([target_x] * len(q_log))
        concatenated_target_z_log.extend([target_z] * len(q_log))


    # Call the new function to generate the GIF
    generate_robot_arm_full_trajectory_gif(
        concatenated_q1_log,
        concatenated_q2_log,
        sim_params,
        robot_params,
        output_gif_filename='experiment/results/robot_arm_full_trajectory.gif',
        desired_total_gif_frames=200,
        target_trajectory_x=concatenated_target_x_log,
        target_trajectory_z=concatenated_target_z_log
)
    
def main(generate_plot=True, generate_gif=True):
    # Define initial joint angles (q1, q2)
    q_init = np.array([np.pi*5/12, np.pi * (1 - 10/12)])
    
    # Generate a uniform grid of target positions
    target_positions = uniform_grid(L1+L2, 0.8)

    # Run simulations for all target positions
    simulation_logs, sim_params = run_experiment(q_init, target_positions, cruise_speed=2.0, max_acceleration=10.0, verbose=True)

    # Plot results for all target positions
    if generate_plot:
        print("Generating plots...")
        plot_results_for_all_targets(simulation_logs, sim_params, q_init, target_positions, RobotParameters(L1=L1, L2=L2, m1=m1, m2=m2, c1=c1, c2=c2, g=g, I1=I1, I2=I2))

    # Generate GIF for the full trajectory across all target positions
    if generate_gif:
        print("Generating GIF...")
        generate_gif_for_full_trajectory(simulation_logs, sim_params, RobotParameters(L1=L1, L2=L2, m1=m1, m2=m2, c1=c1, c2=c2, g=g, I1=I1, I2=I2), target_positions)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot",
        dest="generate_plot",
        action="store_true",
    )
    parser.add_argument(
        "--no-plot",
        dest="generate_plot",
        action="store_false",
    )
    parser.add_argument(
        "--gif",
        dest="generate_gif",
        action="store_true",
    )
    parser.add_argument(
        "--no-gif",
        dest="generate_gif",
        action="store_false",
    )

    parser.set_defaults(
        generate_plot=True,
        generate_gif=True,
    )

    args = parser.parse_args()

    print(f"Generate plot: {args.generate_plot}")
    print(f"Generate GIF: {args.generate_gif}")

    main(
        generate_plot=args.generate_plot,
        generate_gif=args.generate_gif,
    )