import matplotlib.pyplot as plt
import numpy as np

from src.robot.arm_2dof.dynamics import calcuate_inertia_matrix, calculate_coriolis_centrifugal_torques, calculate_gravity_torques
from src.robot.arm_2dof.kinematics import get_jacobian, get_J_dot

def plot_polygonal_path_results(x_log, q_log, torque_log, polygon_path_obj, sim_params, robot_params):
        
    x1_log = x_log[:, 0]
    x2_log = x_log[:, 1]
    q1_log = q_log[:, 0]
    q2_log = q_log[:, 1]
    torque1_log = torque_log[:, 0]
    torque2_log = torque_log[:, 1]
    
    polygon_vertices = polygon_path_obj.vertices
    
    fig_polygon_results = plt.figure(figsize=(10, 12)) # Increased figure height to accommodate the new plot
    fig_polygon_results.suptitle('Polygon Path Following Results', y=0.99, fontsize=20)

    # Create subplots using gridspec-like indexing for flexible layout
    # Row 1: Joint Angles, Joint Torque
    ax_q = fig_polygon_results.add_subplot(3, 2, 1)
    ax_torque = fig_polygon_results.add_subplot(3, 2, 2)

    # Row 2: EE position, EE speed
    ax_ee = fig_polygon_results.add_subplot(3, 2, 3)
    ax_speed = fig_polygon_results.add_subplot(3, 2, 4)

    # Row 3: EE trajectory
    ax_traj = fig_polygon_results.add_subplot(3, 2, 5)
    ax_accel = fig_polygon_results.add_subplot(3, 2, 6)

    # --- Joint Angles and Torques over Time ---

    time_log = [i * sim_params.dt for i in range(sim_params.num_steps)]

    ax_q.plot(time_log, q1_log, label='Joint 1 Angle')
    ax_q.plot(time_log, q2_log, label='Joint 2 Angle')
    ax_q.set_title('Joint Angles for Polygonal Path')
    ax_q.set_xlabel('Time [s]')
    ax_q.set_ylabel('Angle [rad]')
    ax_q.grid(True)
    ax_q.legend(loc='lower right', fontsize='x-small')

    ax_torque.plot(time_log, torque1_log, label='Joint 1 Torque')
    ax_torque.plot(time_log, torque2_log, label='Joint 2 Torque')
    ax_torque.set_title('Joint Torques for Polygonal Path')
    ax_torque.set_xlabel('Time [s]')
    ax_torque.set_ylabel('Torque [Nm]')
    ax_torque.grid(True)
    ax_torque.legend(loc='lower right', fontsize='x-small')

    ax_ee.plot(time_log, x1_log, label='End-effector X')
    ax_ee.plot(time_log, x2_log, label='End-effector Z')
    ax_ee.set_title('End-effector X, Z Position for Polygonal Path')
    ax_ee.set_xlabel('Time [s]')
    ax_ee.set_ylabel('Position [m]')
    ax_ee.grid(True)
    ax_ee.legend(loc='lower right', fontsize='x-small')

    # --- End-effector Velocities and Speed over Time (NEW PLOT) ---
    # Calculate end-effector velocities by numerical differentiation
    x_vel_polygon = np.gradient(np.array(x1_log), np.array(time_log))
    z_vel_polygon = np.gradient(np.array(x2_log), np.array(time_log))

    # Calculate the total speed (magnitude of the velocity vector)
    speed_polygon = np.sqrt(x_vel_polygon**2 + z_vel_polygon**2)

    ax_speed.plot(time_log, x_vel_polygon, label='End-effector X Velocity', color='blue')
    ax_speed.plot(time_log, z_vel_polygon, label='End-effector Z Velocity', color='green')
    ax_speed.plot(time_log, speed_polygon, label='End-effector Total Speed', color='red', linestyle='--')
    ax_speed.set_title('End-effector Velocities and Speed for Polygonal Path')
    ax_speed.set_xlabel('Time [s]')
    ax_speed.set_ylabel('Velocity [m/s] / Speed [m/s]')
    ax_speed.grid(True)
    ax_speed.legend(loc='lower right', fontsize='x-small')

    # --- Calculate End-effector Acceleration Magnitude ---
    print("\nCalculating end-effector acceleration magnitudes for polygonal path...")
    q_log_np = np.array([q1_log, q2_log])  # shape (2, num_steps)
    tau_log_np = np.array([torque1_log, torque2_log])  # shape (2, num_steps)

    ddot_mags_polygon = []

    # Estimate qdot using numerical differentiation (central difference for interior points)
    qdot_estimated_polygon = np.zeros_like(q_log_np, dtype=float)
    if sim_params.num_steps > 1:
        # Forward difference for the first step
        qdot_estimated_polygon[:, 0] = (q_log_np[:, 1] - q_log_np[:, 0]) / sim_params.dt
        # Backward difference for the last step
        qdot_estimated_polygon[:, -1] = (q_log_np[:, -1] - q_log_np[:, -2]) / sim_params.dt
        # Central difference for interior points
        if sim_params.num_steps > 2:
            qdot_estimated_polygon[:, 1:-1] = (q_log_np[:, 2:] - q_log_np[:, :-2]) / (2 * sim_params.dt)
    else:
        # If only one step, qdot is zero
        qdot_estimated_polygon[:, 0] = 0.0


    for i in range(sim_params.num_steps):
        q1, q2 = q_log_np[:, i]
        q1_dot, q2_dot = qdot_estimated_polygon[:, i]
        tau_current = tau_log_np[:, i]

        M = calcuate_inertia_matrix(q1, q2, robot_params.m1, robot_params.m2, robot_params.I1, robot_params.I2, robot_params.c1, robot_params.c2, robot_params.L1, robot_params.L2)
        C = calculate_coriolis_centrifugal_torques(q1, q2, q1_dot, q2_dot, robot_params.m2, robot_params.c2, robot_params.L1)
        G = calculate_gravity_torques(q1, q2, robot_params.m1, robot_params.m2, robot_params.c1, robot_params.c2, robot_params.L1, robot_params.g)
        J = get_jacobian(q1, q2, robot_params.L1, robot_params.L2)
        Jdot = get_J_dot(q1, q2, q1_dot, q2_dot, robot_params.L1, robot_params.L2)

        M_inv = np.linalg.inv(M)
        q_ddot = M_inv @ (tau_current - C - G) # Using the inverse dynamics to get joint acceleration

        # End-effector acceleration: x_ddot = Jdot @ qdot + J @ q_ddot
        x_ddot = Jdot @ qdot_estimated_polygon[:, i] + J @ q_ddot
        x_ddot_magnitude = np.linalg.norm(x_ddot)
        ddot_mags_polygon.append(x_ddot_magnitude)

    print("End-effector acceleration magnitudes calculated for polygonal path.")

    # --- Plot EE Acceleration Magnitude (NEW PLOT) ---
    ax_accel.plot(time_log, ddot_mags_polygon, label='EE Acceleration Magnitude')
    ax_accel.set_title('EE Acceleration Magnitude for Polygonal Path')
    ax_accel.set_xlabel('Time [s]')
    ax_accel.set_ylabel('Acceleration Magnitude [m/s^2]')
    ax_accel.grid(True)
    ax_accel.legend()

    # --- End-effector XY Trajectory Plot for Polygonal Trajectory (MOVED TO LAST SUBPLOT) ---
    ax_traj.plot(x1_log, x2_log, label='End-effector Trajectory', color='blue')

    # Plot the desired Polygonal path
    # Create a closed loop by adding the first vertex at the end
    desired_polygon_path_x = np.append(polygon_vertices[:, 0], polygon_vertices[0, 0])
    desired_polygon_path_z = np.append(polygon_vertices[:, 1], polygon_vertices[0, 1])
    ax_traj.plot(desired_polygon_path_x, desired_polygon_path_z, 'r--', label='Desired Polygonal Path')

    # Plot workspace boundary
    workspace_radius = robot_params.L1 + robot_params.L2
    circle_osc = plt.Circle((0, 0), workspace_radius, color='gray', alpha=0.1, fill=True, linestyle='--', edgecolor='black')
    ax_traj.add_patch(circle_osc)

    ax_traj.set_title('End-effector Trajectory for Polygonal Path')
    ax_traj.set_xlabel('X Position [m]')
    ax_traj.set_ylabel('Z Position [m]')
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.grid(True)
    ax_traj.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent overlapping titles/labels
    plt.savefig('experiment/results/polygonal_path_simulation_results.png', dpi=300, bbox_inches='tight') # Save the figure as a high-resolution PNG
    
    plt.show(block=False)
    plt.pause(5)
    plt.close()