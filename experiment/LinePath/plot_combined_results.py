# @title
import matplotlib.pyplot as plt
import numpy as np

from src.parameters import RobotParameters, SimulationParameters
from src.robot.arm_2dof.kinematics import get_end_effector_position
from src.utils import create_experiment_dir

# ============================================
# Process simulation logs to extract relevant data for plotting
# ============================================
def process_simulation_logs(simulation_logs: list, simulation_metrics: list):
    
    osc_all_q_logs = []
    osc_all_torque_logs = []
    osc_all_a_cmd_logs = []
    
    osc_all_x_logs = []
    osc_all_xdot_logs = []
    osc_all_xddot_logs = []
    
    osc_all_rmse_log = []
    osc_all_error_logs = []
    osc_all_dist_logs = []
    osc_ee_convergence_times = []
    osc_all_manipulability_logs = []

    for log, metrics in zip(simulation_logs, simulation_metrics):
        # Extract data from the simulation log
        q_log = np.array([sim_result.q for sim_result in log])
        torque_log = np.array([sim_result.tau for sim_result in log])
        a_cmd_log = np.array([sim_result.a_cmd for sim_result in log])
        
        x_log = np.array([sim_result.x for sim_result in log])
        xdot_log = np.array([sim_result.xdot for sim_result in log])
        xddot_log = np.array([sim_result.xddot for sim_result in log])
        
        rmse_log = metrics.rmse
        error_log = metrics.errors
        dist_logs = metrics.distance_errors
        convergence_time = metrics.convergence_time
        manipulability_log = metrics.manipulability
        
        osc_all_q_logs.append(q_log)
        osc_all_torque_logs.append(torque_log)
        osc_all_a_cmd_logs.append(a_cmd_log)
        osc_all_x_logs.append(x_log)
        osc_all_xdot_logs.append(xdot_log)
        osc_all_xddot_logs.append(xddot_log)
        
        osc_all_rmse_log.append(rmse_log)
        osc_all_error_logs.append(error_log)
        osc_all_dist_logs.append(dist_logs)
        osc_ee_convergence_times.append(convergence_time)
        osc_all_manipulability_logs.append(manipulability_log)

    return (osc_all_q_logs, osc_all_torque_logs, osc_all_a_cmd_logs,
            osc_all_x_logs, osc_all_xdot_logs, osc_all_xddot_logs,
            osc_all_error_logs, osc_all_rmse_log, 
            osc_all_dist_logs, osc_ee_convergence_times,
            osc_all_manipulability_logs)


def plot_combined_results(simulation_logs: list, simulation_metrics: list, q_init: np.ndarray, target_positions: np.ndarray, 
                          sim_params: SimulationParameters, robot_params: RobotParameters, experiment_dir = None,
                          save_individual_figures: bool = False):
    
    osc_all_q_logs, osc_all_torque_logs, osc_all_a_cmd_logs, \
        osc_all_x_logs, osc_all_xdot_logs, osc_all_xddot_logs, \
            osc_all_error_logs, osc_all_rmse_log, osc_all_dist_logs, \
                osc_ee_convergence_times, osc_all_manipulability_logs = process_simulation_logs(simulation_logs, simulation_metrics)
    
    # Regenerate time_log to match the current simulation steps
    time_log = [i * sim_params.dt for i in range(sim_params.num_steps)]

    # Calculate initial end-effector position (constant for all targets)
    initial_ee_pos = get_end_effector_position(q_init[0], q_init[1], robot_params.L1, robot_params.L2)
    
    height = len(simulation_logs)

    fig = plt.figure(figsize=(24, height))
    fig.suptitle('Operational Space Control - Combined Simulation Results', y=0.99, fontsize=20)

    # Create subplots using gridspec-like indexing for flexible layout
    # Row 1: Joint 1 Angle, Joint 2 Angle, EE Distance Error
    ax_q1 = fig.add_subplot(4, 4, 1)
    ax_q2 = fig.add_subplot(4, 4, 2)

    # Row 2: Joint 1 Torque, Joint 2 Torque
    ax_q1_torque = fig.add_subplot(4, 4, 3)
    ax_q2_torque = fig.add_subplot(4, 4, 4)

    # Row 3: Acceleration Command, Acceleration Magnitude
    ax_a_cmd = fig.add_subplot(4, 4, 5)
    ax_accel = fig.add_subplot(4, 4, 6)

    # Row 4: End Effector Speed, Distance Error
    ax_ee_speed = fig.add_subplot(4, 4, 7)
    ax_dist = fig.add_subplot(4, 4, 8)

    # Row 5: Convergence Time, Manipulability
    ax_conv = fig.add_subplot(4, 4, 9)
    ax_manip = fig.add_subplot(4, 4, 10)

    # Row 6: Average Speed, EE Trajectory
    ax_avg_speed = fig.add_subplot(4, 4, 11)
    ax_ee_traj = fig.add_subplot(4, 4, 12)

    # Row 7: Path Tracking Error
    ax_error = fig.add_subplot(4, 4, 13)

    # --- Plot 1: Joint 1 Angle over Time ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ax_q1.plot(time_log[::10], osc_all_q_logs[idx][::10, 0], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_q1.set_title('Joint 1 Angle')
    ax_q1.set_xlabel("Time [s]")
    ax_q1.set_ylabel("Angle [rad]")
    ax_q1.grid(True)
    ax_q1.legend(loc='upper right', fontsize='x-small')

    # --- Plot 2: Joint 2 Angle over Time ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ax_q2.plot(time_log[::10], osc_all_q_logs[idx][::10, 1], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_q2.set_title('Joint 2 Angle')
    ax_q2.set_xlabel("Time [s]")
    ax_q2.set_ylabel("Angle [rad]")
    ax_q2.grid(True)
    ax_q2.legend(loc='upper right', fontsize='x-small')

    # --- Plot 3: Joint 1 Torque over Time ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ax_q1_torque.plot(time_log[::10], osc_all_torque_logs[idx][::10, 0], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_q1_torque.set_title('Joint 1 Torque')
    ax_q1_torque.set_xlabel("Time [s]")
    ax_q1_torque.set_ylabel("Torque [Nm]")
    ax_q1_torque.grid(True)
    ax_q1_torque.legend(loc='upper right', fontsize='x-small')

    # --- Plot 4: Joint 2 Torque over Time ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ax_q2_torque.plot(time_log[::10], osc_all_torque_logs[idx][::10, 1], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_q2_torque.set_title('Joint 2 Torque')
    ax_q2_torque.set_xlabel("Time [s]")
    ax_q2_torque.set_ylabel("Torque [Nm]")
    ax_q2_torque.grid(True)
    ax_q2_torque.legend(loc='upper right', fontsize='x-small')

    # --- Plot 5: Acceleration Command ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        a_cmd = np.array(osc_all_a_cmd_logs[idx]) # Shape (num_steps, 2, 1)
        a_cmd_magnitude = np.linalg.norm(a_cmd, axis=1).flatten() # Shape (num_steps,)
        ax_a_cmd.plot(time_log[::10], a_cmd_magnitude[::10], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_a_cmd.set_title('Acceleration Command')
    ax_a_cmd.set_xlabel("Time [s]")
    ax_a_cmd.set_ylabel("Acceleration Command [m/s^2]")
    ax_a_cmd.grid(True)
    ax_a_cmd.legend(loc='upper right', fontsize='x-small')

    # --- Plot 6: EE Acceleration Magnitude ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ee_acceleration = np.array(osc_all_xddot_logs[idx]) # Shape (num_steps, 2)
        ee_acceleration_magnitude = np.linalg.norm(ee_acceleration, axis=1) # Shape (num_steps,)
        ax_accel.plot(time_log[::10], ee_acceleration_magnitude[::10], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_accel.set_title('EE Acceleration Magnitude')
    ax_accel.set_xlabel("Time [s]")
    ax_accel.set_ylabel("Acceleration Magnitude [m/s^2]")
    ax_accel.grid(True)
    ax_accel.legend(loc='upper right', fontsize='x-small')

    # --- Plot 7: EE Speed ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        velocity = np.array(osc_all_xdot_logs[idx]) # Shape (num_steps, 2)
        speed = np.linalg.norm(velocity, axis=1) # Shape (num_steps,)
        ax_ee_speed.plot(time_log[::10], speed[::10], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_ee_speed.set_title('EE Speed')
    ax_ee_speed.set_xlabel("Time [s]")
    ax_ee_speed.set_ylabel("Speed [m/s]")
    ax_ee_speed.grid(True)
    ax_ee_speed.legend(loc='upper right', fontsize='x-small')

    # --- Plot 8: End-effector Distance Error over Time ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ax_dist.plot(time_log[::10], osc_all_dist_logs[idx][::10], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_dist.set_title('EE Distance Error')
    ax_dist.set_xlabel("Time [s]")
    ax_dist.set_ylabel("Distance Error [m]")
    ax_dist.grid(True)
    ax_dist.legend(loc='upper right', fontsize='x-small')

    # --- Plot 9: End-effector Convergence Time ---
    bar_width = 0.25
    index = np.arange(len(target_positions))
    target_labels = [f'({tx:.1f}, {tz:.1f})' for tx, tz in target_positions]
    ax_conv.bar(index, osc_ee_convergence_times, bar_width, label='Operational Space', color='lightgreen')
    ax_conv.set_title('EE Convergence Time (+/-0.01 m error)')
    ax_conv.set_xlabel('Target Position (X, Z)')
    ax_conv.set_ylabel('Convergence Time [s]')
    ax_conv.set_xticks(index)
    ax_conv.set_xticklabels(target_labels, rotation=45, ha='right', fontsize='x-small')
    ax_conv.legend(fontsize='x-small')
    ax_conv.grid(True)

    # --- Plot 10: Manipulability ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ax_manip.plot(time_log[::10], osc_all_manipulability_logs[idx][::10], label=f'Target ({target_x:.2f}, {target_z:.2f})')
    ax_manip.set_title('Manipulability det(JJ^T)')
    ax_manip.set_xlabel("Time [s]")
    ax_manip.set_ylabel("Manipulability")
    ax_manip.grid(True)
    ax_manip.legend(loc='upper right', fontsize='x-small')

    # --- Plot 11: Average End-effector Speed ---
    average_speeds = []
    for idx, (target_x, target_z) in enumerate(target_positions):
        target_pos_np = np.array([target_x, target_z])
        straight_line_distance = np.linalg.norm(target_pos_np - initial_ee_pos)
        convergence_time = osc_ee_convergence_times[idx]
        if convergence_time > 0:
            avg_speed = straight_line_distance / convergence_time
        else:
            avg_speed = 0.0 # Or np.nan, depending on desired handling of unconverged targets
        average_speeds.append(avg_speed)

    ax_avg_speed.bar(index, average_speeds, bar_width, color='skyblue')
    ax_avg_speed.set_title('Average EE Speed (Distance / Convergence Time)')
    ax_avg_speed.set_xlabel('Target Position (X, Z)')
    ax_avg_speed.set_ylabel('Average Speed [m/s]')
    ax_avg_speed.set_xticks(index)
    ax_avg_speed.set_xticklabels(target_labels, rotation=45, ha='right', fontsize='x-small')
    ax_avg_speed.grid(True)

    # --- Plot 12: End-effector XY Path Plot ---
    workspace_radius = robot_params.L1 + robot_params.L2
    circle_osc = plt.Circle((0, 0), workspace_radius, facecolor='gray', alpha=0.1, fill=True, linestyle='--', edgecolor='black')
    ax_ee_traj.add_patch(circle_osc)

    start_ee_x = robot_params.L1 * np.sin(q_init[0]) + robot_params.L2 * np.sin(q_init[0] + q_init[1])
    start_ee_z = -robot_params.L1 * np.cos(q_init[0]) - robot_params.L2 * np.cos(q_init[0] + q_init[1])
    ax_ee_traj.plot(start_ee_x, start_ee_z, 'go', markersize=6, label='Start')

    ax_ee_traj.set_title('EE Path')
    ax_ee_traj.set_xlabel("X Position [m]")
    ax_ee_traj.set_ylabel("Z Position [m]")
    ax_ee_traj.set_aspect('equal', adjustable='box')
    ax_ee_traj.grid(True)
    for idx, (target_x, target_z) in enumerate(target_positions):
        # Extract x and z components from the ee_x_log for the current target
        ee_x_coords = osc_all_x_logs[idx][:, 0]
        ee_z_coords = osc_all_x_logs[idx][:, 1]
        ax_ee_traj.plot(target_x, target_z, 'rx', markersize=6, mew=2) # Mark target points
        ax_ee_traj.plot(ee_x_coords[::10], ee_z_coords[::10], label=f'Path to ({target_x:.2f}, {target_z:.2f})')
        
    # --- Plot 13: Path Tracking Error over Time ---
    for idx, (target_x, target_z) in enumerate(target_positions):
        ax_error.plot(
            time_log[::10], osc_all_error_logs[idx][::10],
            label=f'Target ({target_x:.2f}, {target_z:.2f})'
        )
    ax_error.set_title('Path Tracking Error')
    ax_error.set_xlabel("Time [s]")
    ax_error.set_ylabel("Path Error [m]")
    ax_error.grid(True)
    ax_error.legend(loc='upper right', fontsize='x-small')

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent overlapping titles/labels
    
    if experiment_dir != None:
        figures_dir = experiment_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "combined_figures.png", dpi=300, bbox_inches='tight')

        if save_individual_figures:
            individual_figures = [
                ("joint_1_angle.png", ax_q1),
                ("joint_2_angle.png", ax_q2),
                ("joint_1_torque.png", ax_q1_torque),
                ("joint_2_torque.png", ax_q2_torque),
                ("acceleration_command.png", ax_a_cmd),
                ("ee_acceleration_magnitude.png", ax_accel),
                ("ee_speed.png", ax_ee_speed),
                ("ee_distance_error.png", ax_dist),
                ("ee_convergence_time.png", ax_conv),
                ("manipulability.png", ax_manip),
                ("average_ee_speed.png", ax_avg_speed),
                ("ee_path.png", ax_ee_traj),
                ("path_tracking_error.png", ax_error),
            ]
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for filename, ax in individual_figures:
                bbox = ax.get_tightbbox(renderer).transformed(
                    fig.dpi_scale_trans.inverted()
                ).padded(0.1)
                fig.savefig(
                    figures_dir / filename,
                    dpi=300,
                    bbox_inches=bbox,
                )
    
    plt.show(block=False)
    plt.pause(5)
    plt.close()
