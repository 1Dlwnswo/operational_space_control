import numpy as np
from src.parameters import RobotParameters, SimulationParameters
from src.mujoco_setup import model, data, dt, L1, L2, m1, m2, c1, c2, g, I1, I2, get_end_effector_position, get_jacobian, get_ee_convergence_time
from src.path.LinePath import LinePath
from simulation.run_simulation import run_simulation
from experiment.LinePath.generate_grid import uniform_grid
from experiment.LinePath.plot_combined_results import plot_combined_results
from experiment.gif_generation import generate_robot_arm_full_trajectory_gif

# ============================================
# Instantiate Dataclasses
# ============================================
cruise_speed = 2.0
max_acceleration = 10.0

robot_params = RobotParameters(L1=L1, L2=L2, m1=m1, m2=m2, c1=c1, c2=c2, g=g, I1=I1, I2=I2)
sim_params = SimulationParameters(dt=dt, num_steps=int(4000/cruise_speed + 2000*cruise_speed/max_acceleration))

q_init = np.array([np.pi*5/12, np.pi * (1 - 10/12)]) # Initial joint angles [q1, q2]


# ============================================
# Set target positions
# ============================================

target_positions = uniform_grid(L1+L2, 0.8)
'''
initial_ee_pos = np.array([-1.5, 1])
final_ee_pos = np.array([-1.5, -1])
target_positions = linear_grid(initial_ee_pos=initial_ee_pos, final_ee_pos=final_ee_pos, n = 4)
'''


# ============================================
# Initialize logs for all simulations
# ============================================
osc_all_q_logs = []
osc_all_qdot_logs = []
osc_all_qddot_logs = []
osc_all_torque_logs = []
osc_all_a_cmd_logs = []

osc_all_x_logs = []
osc_all_xdot_logs = []
osc_all_xddot_logs = []

osc_all_manipulability_logs = []
osc_all_dist_logs = []
osc_ee_convergence_times = []

print(f"Number of OSC simulations: {len(target_positions)}")


# ============================================
# Run simulation for each target position
# ============================================
for idx, (target_x, target_z) in enumerate(target_positions):

    ee_start_pos = get_end_effector_position(q_init[0], q_init[1], robot_params.L1, robot_params.L2)

    path_obj = LinePath(start_point=ee_start_pos, end_point=np.array([target_x, target_z]), cruise_speed=cruise_speed)

    current_simulation_log = run_simulation(model, data, q_init, "pw", robot_params, sim_params, path_obj, max_acceleration = max_acceleration, save_operational_space=True)

    q_log_current = np.array([sim_result.q for sim_result in current_simulation_log])
    qdot_log_current = np.array([sim_result.qdot for sim_result in current_simulation_log])
    qddot_log_current = np.array([sim_result.qddot for sim_result in current_simulation_log])
    torque_log_current = np.array([sim_result.tau for sim_result in current_simulation_log])
    a_cmd_log_current = np.array([sim_result.a_cmd for sim_result in current_simulation_log])

    x_log_current = np.array([sim_result.x for sim_result in current_simulation_log])
    xdot_log_current = np.array([sim_result.xdot for sim_result in current_simulation_log])
    xddot_log_current = np.array([sim_result.xddot for sim_result in current_simulation_log])

    osc_all_q_logs.append(q_log_current)
    osc_all_torque_logs.append(torque_log_current)
    osc_all_a_cmd_logs.append(a_cmd_log_current)

    osc_all_x_logs.append(x_log_current)
    osc_all_xdot_logs.append(xdot_log_current)
    osc_all_xddot_logs.append(xddot_log_current)

    # Calulate Manipulability over time
    current_sim_manipulability = []
    for k in range(sim_params.num_steps):
        q1_k = q_log_current[k, 0]
        q2_k = q_log_current[k, 1]
        J_k = get_jacobian(q1_k, q2_k, robot_params.L1, robot_params.L2)
        manipulability_k = np.linalg.det(J_k @ J_k.T)
        current_sim_manipulability.append(manipulability_k)
    osc_all_manipulability_logs.append(current_sim_manipulability)

    # Calculate distance error over time
    dist_log_current_sim = np.linalg.norm(np.array(x_log_current) - np.array(target_positions[idx]), axis=1)
    osc_all_dist_logs.append(dist_log_current_sim)

    # Calculate convergence time
    conv_time_ee = get_ee_convergence_time(np.array(x_log_current), np.array(target_positions[idx]), sim_params.dt)
    osc_ee_convergence_times.append(conv_time_ee)

print("Operational Space Control simulation complete.\n")


# ============================================
# Plot combined results for all target positions
# ============================================
plot_combined_results(osc_all_q_logs, osc_all_torque_logs, osc_all_a_cmd_logs, osc_all_xdot_logs, osc_all_xddot_logs, 
                      osc_all_dist_logs, osc_ee_convergence_times, osc_all_manipulability_logs, osc_all_x_logs, target_positions, sim_params, robot_params, q_init)


# ============================================
# Generate GIF for the full trajectory across all target positions
# ============================================
concatenated_q1_log = []
concatenated_q2_log = []
concatenated_target_x_log = []
concatenated_target_z_log = []

for idx, (target_x, target_z) in enumerate(target_positions):
    # Extend with the actual logged joint positions for this target
    concatenated_q1_log.extend(osc_all_q_logs[idx][:,0])
    concatenated_q2_log.extend(osc_all_q_logs[idx][:,1])
    # Repeat the current target for the duration of this simulation segment
    concatenated_target_x_log.extend([target_x] * sim_params.num_steps)
    concatenated_target_z_log.extend([target_z] * sim_params.num_steps)


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