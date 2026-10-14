# @title
import numpy as np
from src.parameters import RobotParameters, SimulationParameters
from src.mujoco_setup import model, data, dt, L1, L2, m1, m2, c1, c2, g, I1, I2, get_end_effector_position, get_jacobian, get_ee_convergence_time
from src.path.CirclePath import CirclePath
from simulation.run_simulation import run_simulation
from experiment.gif_generation import generate_robot_arm_full_trajectory_gif

circular_path_center = np.array([0.3, 0.0]) # Center of Circle (x, z)
circular_path_radius = 0.7 # Radius of Circle

circular_path_obj = CirclePath(
    center=circular_path_center,
    radius=circular_path_radius
)

print(f"Circular Path Parameters: Center={circular_path_obj.center}, Radius={circular_path_obj.radius}")

circular_path_sim_params = SimulationParameters(dt=dt, num_steps=int(20 / dt))
robot_params = RobotParameters(L1=L1, L2=L2, m1=m1, m2=m2, c1=c1, c2=c2, g=g, I1=I1, I2=I2)

q_init = np.array([np.pi*5/12, np.pi * (1 - 10/12)]) # Initial joint angles [q1, q2]

max_acceleration = 15

# Run the simulation for circular path tracking
circular_simulation_log = run_simulation(model, data, q_init, "pw", robot_params, circular_path_sim_params, circular_path_obj, max_acceleration=max_acceleration, save_operational_space=True)

# Extract logs for joint angles, torques, and end-effector positions
q_log_circular_all = np.array([sim_result.q for sim_result in circular_simulation_log])
q1_log_circular = q_log_circular_all[:, 0]
q2_log_circular = q_log_circular_all[:, 1]

torque_log_circular_all = np.array([sim_result.tau for sim_result in circular_simulation_log])
torque1_log_circular = torque_log_circular_all[:, 0]
torque2_log_circular = torque_log_circular_all[:, 1]

ee_pos_log_circular_all = np.array([sim_result.x for sim_result in circular_simulation_log])
ee_x_log_circular = ee_pos_log_circular_all[:, 0]
ee_z_log_circular = ee_pos_log_circular_all[:, 1]

print("Circular Path Tracking simulation complete.")

desired_speed_circular = circular_path_obj.cruise_angular_speed * circular_path_obj.radius

# Generate the full desired circular path for plotting
theta = np.linspace(0, 2 * np.pi, 500) # More points for a smooth circle
full_circular_path_x = circular_path_obj.center[0] + circular_path_obj.radius * np.cos(theta)
full_circular_path_z = circular_path_obj.center[1] + circular_path_obj.radius * np.sin(theta)

# Call the GIF generation function for the circular path
generate_robot_arm_full_trajectory_gif(
    q1_log_circular,
    q2_log_circular,
    circular_path_sim_params,
    robot_params,
    output_gif_filename='experiment/results/robot_arm_circular_path.gif',
    desired_total_gif_frames=200,
    full_desired_path_x=full_circular_path_x,
    full_desired_path_z=full_circular_path_z
)