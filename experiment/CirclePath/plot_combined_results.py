import matplotlib.pyplot as plt
import numpy as np
from src.mujoco_setup import dt
from src.parameters import RobotParameters, SimulationParameters

def plot_circular_path_results(x_log, q_log, torque_log, circular_path_center, circular_path_radius ,sim_params: SimulationParameters, robot_params: RobotParameters):
    _, axes_circular = plt.subplots(2, 2, figsize=(10, 10))
    
    ee_x_log_circular = x_log[:, 0]
    ee_z_log_circular = x_log[:, 1]
    q1_log_circular = q_log[:, 0]
    q2_log_circular = q_log[:, 1]
    torque1_log_circular = torque_log[:, 0]
    torque2_log_circular = torque_log[:, 1]
    
    # Create time_log for circular trajectory based on its simulation parameters
    time_log_circular = [i * dt for i in range(sim_params.num_steps)]

    # --- End-effector X, Z Position over Time ---
    axes_circular[0,0].plot(time_log_circular, ee_x_log_circular, label='End-effector X')
    axes_circular[0,0].plot(time_log_circular, ee_z_log_circular, label='End-effector Z')
    axes_circular[0,0].set_title('End-effector X, Z Position for Circular Path')
    axes_circular[0,0].set_xlabel('Time [s]')
    axes_circular[0,0].set_ylabel('Position [m]')
    axes_circular[0,0].grid(True)
    axes_circular[0,0].legend()

    # --- Joint Angles (q1, q2) over Time ---
    axes_circular[0,1].plot(time_log_circular, q1_log_circular, label='Joint 1 Angle')
    axes_circular[0,1].plot(time_log_circular, q2_log_circular, label='Joint 2 Angle')
    axes_circular[0,1].set_title('Joint Angles for Circular Path')
    axes_circular[0,1].set_xlabel('Time [s]')
    axes_circular[0,1].set_ylabel('Angle [rad]')
    axes_circular[0,1].grid(True)
    axes_circular[0,1].legend()

    # --- Joint Torques (torque1, torque2) over Time ---
    axes_circular[1,0].plot(time_log_circular, torque1_log_circular, label='Joint 1 Torque')
    axes_circular[1,0].plot(time_log_circular, torque2_log_circular, label='Joint 2 Torque')
    axes_circular[1,0].set_title('Joint Torques for Circular Path')
    axes_circular[1,0].set_xlabel('Time [s]')
    axes_circular[1,0].set_ylabel('Torque [Nm]')
    axes_circular[1,0].grid(True)
    axes_circular[1,0].legend()

    # --- End-effector XY Trajectory Plot for Circular Trajectory ---
    axes_circular[1,1].plot(ee_x_log_circular, ee_z_log_circular, label='End-effector Trajectory', color='blue')

    # Plot the desired circular path
    theta = np.linspace(0, 2 * np.pi, 100) # Generate angles for a full circle
    desired_x = circular_path_center[0] + circular_path_radius * np.cos(theta)
    desired_z = circular_path_center[1] + circular_path_radius * np.sin(theta)
    axes_circular[1,1].plot(desired_x, desired_z, 'r--', label='Desired Circular Path')

    # Plot workspace boundary
    workspace_radius = robot_params.L1 + robot_params.L2
    circle_osc = plt.Circle((0, 0), workspace_radius, color='gray', alpha=0.1, fill=True, linestyle='--', edgecolor='black')
    axes_circular[1,1].add_patch(circle_osc)

    axes_circular[1,1].set_title('End-effector Trajectory for Circular Path')
    axes_circular[1,1].set_xlabel('X Position [m]')
    axes_circular[1,1].set_ylabel('Z Position [m]')
    axes_circular[1,1].set_aspect('equal', adjustable='box')
    axes_circular[1,1].grid(True)
    axes_circular[1,1].legend()

    plt.tight_layout()
    plt.suptitle('Circular Path Tracking Simulation Results', y=1.02, fontsize=16)    
    plt.savefig('experiment/results/circular_path_simulation_results.png', dpi=300, bbox_inches='tight') # Save the figure as a high-resolution PNG
    
    plt.show(block=False)
    plt.pause(5)
    plt.close()