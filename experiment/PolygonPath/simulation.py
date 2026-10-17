import numpy as np
from src.parameters import RobotParameters, SimulationParameters
from src.mujoco_setup import model, data, dt, L1, L2, m1, m2, c1, c2, g, I1, I2, inverse_kinematics
from src.path.PolygonPath import PolygonPath
from simulation.run_simulation import run_simulation

def star(r, n=5):
    polygon_verticies = []
    for i in range(n):
      polygon_verticies.append([r * np.cos(i * 4 * np.pi / n + np.pi/2), r * np.sin(i * 4 * np.pi / n + np.pi/2)])

    return np.array(polygon_verticies)


def regular_polygon(r, n):
    polygon_verticies = []
    for i in range(n):
      polygon_verticies.append([r * np.cos(2 * np.pi * i / n + np.pi/2), r * np.sin(2 * np.pi * i / n + np.pi/2)])

    return np.array(polygon_verticies)

# Define the vertices of a polygonal path

# polygon_verticies = star(1.5)
polygon_vertices = regular_polygon(1.5, 5)

desired_speed = 3.0
max_acceleration = 45.0

# Create a PolygonalPath object
polygon_path_obj = PolygonPath(
    vertices=polygon_vertices,
    desired_speed=desired_speed,
    stop_at_end = True
)

print(f"\nPolygonal Path Parameters: Vertices=\n{polygon_path_obj.vertices}")

# Update SimulationParameters for the polygonal path
polygon_sim_duration = 20.0 # seconds
polygon_num_steps = int(polygon_sim_duration / dt)
polygon_sim_params = SimulationParameters(dt=dt, num_steps=polygon_num_steps)

robot_params = RobotParameters(L1=L1, L2=L2, m1=m1, m2=m2, c1=c1, c2=c2, g=g, I1=I1, I2=I2)
sim_params = SimulationParameters(dt=dt, num_steps=int(4000/desired_speed + 2000*desired_speed/max_acceleration))

q_init = inverse_kinematics(np.array([0, 1.5]), L1, L2)

# Run simulation
polygon_simulation_log = \
        run_simulation(model, data, q_init, "pw", robot_params, polygon_sim_params, polygon_path_obj, max_acceleration=max_acceleration, save_operational_space=True)

# Extract data from the simulation_log
q_log_polygon_all = np.array([sim_result.q for sim_result in polygon_simulation_log])
q1_log_polygon = q_log_polygon_all[:, 0]
q2_log_polygon = q_log_polygon_all[:, 1]

torque_log_polygon_all = np.array([sim_result.tau for sim_result in polygon_simulation_log])
torque1_log_polygon = torque_log_polygon_all[:, 0]
torque2_log_polygon = torque_log_polygon_all[:, 1]

ee_pos_log_polygon_all = np.array([sim_result.x for sim_result in polygon_simulation_log])
ee_x_log_polygon = ee_pos_log_polygon_all[:, 0]
ee_z_log_polygon = ee_pos_log_polygon_all[:, 1]

print("Polygonal Path Following simulation complete.")