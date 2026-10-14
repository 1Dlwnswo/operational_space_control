import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from src.mujoco_setup import get_end_effector_position
import tempfile

def plot_robot_arm(ax, q1, q2, L1, L2, current_ee_x_history, current_ee_z_history, desired_full_path_x=None, desired_full_path_z=None):
    # Joint 1 position (base)
    x0, z0 = 0.0, 0.0

    # Joint 2 position (end of link1)
    x1 = L1 * np.sin(q1)
    z1 = -L1 * np.cos(q1)

    # End-effector position (end of link2)
    x_ee = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    z_ee = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)

    # Draw robot arm
    ax.plot([x0, x1], [z0, z1], 'o-', lw=4, color='blue', label='Link 1')
    ax.plot([x1, x_ee], [z1, z_ee], 'o-', lw=4, color='green', label='Link 2')

    # Draw the current end-effector position
    ax.plot(x_ee, z_ee, 'ro', markersize=8, label='Current EE')

    # Draw the actual end-effector trajectory (history up to current frame) as a dashed line
    if current_ee_x_history and current_ee_z_history: # Check if history is not empty
        ax.plot(current_ee_x_history, current_ee_z_history, 'b:', alpha=0.7, label='Actual EE Trajectory')

    # Draw the desired full path as a dashed line (e.g., the full circle)
    if desired_full_path_x is not None and desired_full_path_z is not None:
        ax.plot(desired_full_path_x, desired_full_path_z, 'r--', alpha=0.8, label='Desired Path')

    # Draw workspace circle
    workspace_radius = L1 + L2
    circle_patch = plt.Circle((0, 0), workspace_radius, color='gray', alpha=0.1, fill=True, linestyle='--', edgecolor='black')
    ax.add_patch(circle_patch)

    ax.set_xlim(-(workspace_radius + 0.5), (workspace_radius + 0.5))
    ax.set_ylim(-(workspace_radius + 0.5), (workspace_radius + 0.5))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Z Position [m]')
    ax.grid(True)


def generate_robot_arm_full_trajectory_gif(concatenated_q1_log, concatenated_q2_log, sim_params, robot_params, output_gif_filename='robot_arm_full_trajectory.gif', desired_total_gif_frames=200, full_desired_path_x=None, full_desired_path_z=None, target_trajectory_x=None, target_trajectory_z=None):
    """
    Generates a GIF visualizing the robot arm's full trajectory.

    Args:
        concatenated_q1_log (list): List of all q1 joint angles concatenated.
        concatenated_q2_log (list): List of all q2 joint angles concatenated.
        sim_params (SimulationParameters): Simulation parameters object.
        robot_params (RobotParameters): Robot parameters object.
        output_gif_filename (str): Name of the output GIF file.
        desired_total_gif_frames (int): Desired total number of frames for the GIF.
        full_desired_path_x (list, optional): List of x-coordinates for the *entire* desired reference path (e.g., a full circle or a line).
        full_desired_path_z (list, optional): List of z-coordinates for the *entire* desired reference path.
        target_trajectory_x (list, optional): List of x-coordinates for the *current target sequence* (e.g., for point-to-point, the target for each segment).
        target_trajectory_z (list, optional): List of z-coordinates for the *current target sequence*.
    """
    total_sim_steps_for_gif = len(concatenated_q1_log)
    print(f"Total simulation steps for GIF: {total_sim_steps_for_gif}")
    
    frame_filenames = []
    frame_dir = tempfile.TemporaryDirectory()
    fig, ax = plt.subplots(figsize=(8, 8))

    frame_interval = max(1, total_sim_steps_for_gif // desired_total_gif_frames)

    simulation_real_duration = total_sim_steps_for_gif * sim_params.dt
    actual_num_gif_frames = len(concatenated_q1_log[::frame_interval])
    
    if simulation_real_duration > 0 and actual_num_gif_frames > 0:
        fps = actual_num_gif_frames / simulation_real_duration
    else:
        fps = 60 # Default FPS if calculation is not possible

    print(f"Generating GIF frames... Total {actual_num_gif_frames} frames with {fps:.2f} FPS for real-time playback.")

    # Initialize lists to store the end-effector trajectory history
    ee_x_history = []
    ee_z_history = []

    for i in range(0, total_sim_steps_for_gif, frame_interval):
        q1 = concatenated_q1_log[i]
        q2 = concatenated_q2_log[i]

        # Calculate current end-effector position
        x_ee_current, z_ee_current = get_end_effector_position(q1, q2, robot_params.L1, robot_params.L2)
        ee_x_history.append(x_ee_current)
        ee_z_history.append(z_ee_current)

        ax.clear()
        plot_robot_arm(ax, q1, q2, robot_params.L1, robot_params.L2,
                       ee_x_history, ee_z_history, # Pass the history of actual EE positions
                       full_desired_path_x, full_desired_path_z) # Pass the full desired path

        # Plot current target point if provided (e.g., for point-to-point targets)
        if target_trajectory_x is not None and target_trajectory_z is not None:
            ax.plot(target_trajectory_x[i], target_trajectory_z[i], 'rx', markersize=10, mew=2, label='Current Target')
            ax.legend()

        # Set title dynamically
        if full_desired_path_x is not None: # For circular or continuous path
            ax.set_title(f'Robot Arm Trajectory Tracking (Time: {i * sim_params.dt:.2f}s)')
        elif target_trajectory_x is not None: # For point-to-point targets
            ax.set_title(f'Robot Arm Point-to-Point (Target: ({target_trajectory_x[i]:.2f}, {target_trajectory_z[i]:.2f}))')
        else:
            ax.set_title(f'Robot Arm Simulation (Time: {i * sim_params.dt:.2f}s)')
        
        frame_filename = os.path.join(frame_dir.name, f'frame_{i:04d}.png')
        plt.savefig(frame_filename)
        frame_filenames.append(frame_filename)

    with imageio.get_writer(output_gif_filename, mode='I', duration = 1000 / fps) as writer:
        for filename in frame_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    plt.close(fig)
    frame_dir.cleanup()

    print(f"'{output_gif_filename}' file has been saved locally.")