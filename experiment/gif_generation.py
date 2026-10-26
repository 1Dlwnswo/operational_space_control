import os
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np

from src.robot.arm_2dof.kinematics import get_end_effector_position

GIF_FPS = 30


def plot_robot_arm(
    ax,
    q1,
    q2,
    L1,
    L2,
    current_ee_x_history,
    current_ee_z_history,
    desired_full_path_x=None,
    desired_full_path_z=None,
):
    # Joint 1 position (base)
    x0, z0 = 0.0, 0.0

    # Joint 2 position (end of link1)
    x1 = L1 * np.sin(q1)
    z1 = -L1 * np.cos(q1)

    # End-effector position (end of link2)
    x_ee = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    z_ee = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)

    # Draw robot arm
    ax.plot([x0, x1], [z0, z1], "o-", lw=4, color="blue", label="Link 1")
    ax.plot([x1, x_ee], [z1, z_ee], "o-", lw=4, color="green", label="Link 2")

    # Draw the current end-effector position
    ax.plot(x_ee, z_ee, "ro", markersize=8, label="Current EE")

    # Draw the actual end-effector trajectory (history up to current frame) as a dashed line
    if current_ee_x_history and current_ee_z_history:  # Check if history is not empty
        ax.plot(
            current_ee_x_history,
            current_ee_z_history,
            "b:",
            alpha=0.7,
            label="Actual EE Trajectory",
        )

    # Draw the desired full path as a dashed line (e.g., the full circle)
    if desired_full_path_x is not None and desired_full_path_z is not None:
        ax.plot(
            desired_full_path_x,
            desired_full_path_z,
            "r--",
            alpha=0.8,
            label="Desired Path",
        )

    # Draw workspace circle
    workspace_radius = L1 + L2
    circle_patch = plt.Circle(
        (0, 0),
        workspace_radius,
        facecolor="gray",
        alpha=0.1,
        fill=True,
        linestyle="--",
        edgecolor="black",
    )
    ax.add_patch(circle_patch)

    ax.set_xlim(-(workspace_radius + 0.5), (workspace_radius + 0.5))
    ax.set_ylim(-(workspace_radius + 0.5), (workspace_radius + 0.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Z Position [m]")
    ax.grid(True)

# Generates a GIF visualizing the robot arm's full trajectory.
def generate_robot_arm_full_trajectory_gif(
    concatenated_q1_log,
    concatenated_q2_log,
    sim_params,
    output_gif_filename="robot_arm_full_trajectory.gif",
    full_desired_path_x=None,
    full_desired_path_z=None,
    target_trajectory_x=None,
    target_trajectory_z=None,
):
    total_sim_steps_for_gif = len(concatenated_q1_log)
    print(f"Total simulation steps for GIF: {total_sim_steps_for_gif}")

    frame_filenames = []
    frame_dir = tempfile.TemporaryDirectory()
    fig, ax = plt.subplots(figsize=(8, 8))

    simulation_real_duration = total_sim_steps_for_gif * sim_params.dt
    frame_count = int(np.ceil(simulation_real_duration * GIF_FPS))
    frame_indices = np.minimum(
        np.rint(np.arange(frame_count) / GIF_FPS / sim_params.dt).astype(int),
        total_sim_steps_for_gif - 1,
    )

    print(f"Generating GIF frames... Total {len(frame_indices)} frames with {GIF_FPS} FPS for real-time playback.")

    # Initialize lists to store the end-effector trajectory history
    ee_x_history = []
    ee_z_history = []

    for frame_number, i in enumerate(frame_indices):
        q1 = concatenated_q1_log[i]
        q2 = concatenated_q2_log[i]

        L1 = 1.0
        L2 = 1.0

        # Calculate current end-effector position
        x_ee_current, z_ee_current = get_end_effector_position(q1, q2, L1, L2)
        ee_x_history.append(x_ee_current)
        ee_z_history.append(z_ee_current)

        ax.clear()
        plot_robot_arm(
            ax,
            q1,
            q2,
            L1,
            L2,
            ee_x_history,
            ee_z_history,  # Pass the history of actual EE positions
            full_desired_path_x,
            full_desired_path_z,
        )  # Pass the full desired path

        # Plot current target point if provided (e.g., for point-to-point targets)
        if target_trajectory_x is not None and target_trajectory_z is not None:
            ax.plot(
                target_trajectory_x[i],
                target_trajectory_z[i],
                "rx",
                markersize=10,
                mew=2,
                label="Current Target",
            )
            ax.legend()

        # Set title dynamically
        if full_desired_path_x is not None:  # For circular or continuous path
            ax.set_title(f"Robot Arm Trajectory Tracking (Time: {i * sim_params.dt:.2f}s)")
        elif target_trajectory_x is not None:  # For point-to-point targets
            ax.set_title(f"Robot Arm Point-to-Point (Target: ({target_trajectory_x[i]:.2f}, {target_trajectory_z[i]:.2f}))")
        else:
            ax.set_title(f"Robot Arm Simulation (Time: {i * sim_params.dt:.2f}s)")

        frame_filename = os.path.join(frame_dir.name, f"frame_{frame_number:04d}.png")
        plt.savefig(frame_filename)
        frame_filenames.append(frame_filename)

    with imageio.get_writer(output_gif_filename, mode="I", duration=1000 / GIF_FPS) as writer:
        for filename in frame_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    plt.close(fig)
    frame_dir.cleanup()

    print(f"'{output_gif_filename}' file has been saved locally.")
