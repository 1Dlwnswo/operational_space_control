import numpy as np
from typing import Optional

from src.parameters import RobotParameters, SimulationParameters
from src.mujoco_setup import MODEL_PATH, load_model_and_data, get_robot_parameters, get_simulation_parameters
from src.path.PolygonPath import PolygonPath
from src.robot.arm_2dof.kinematics import inverse_kinematics
from src.metrics.metrics import metrics
from src.utils import create_experiment_dir, save_log, save_metrics

from simulation.run_simulation import run_simulation
from experiment.gif_generation import generate_robot_arm_full_trajectory_gif
from experiment.experiment_runner import run_experiment_cli
from experiment.PolygonPath.plot_combined_results import plot_polygonal_path_results

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


# ============================================
# Run Simulation for Polygonal Path Following
# ============================================
def run_experiment(q_init: np.ndarray, polygon_path_obj: PolygonPath, robot_params: Optional[RobotParameters] = None, 
                   max_acceleration = 10.0, verbose = True, use_viewer = False,
                   experiment_dir = None):
    # Load the Mujoco model and data
    model, data = load_model_and_data(MODEL_PATH)
    
    if robot_params is None:
        robot_params = get_robot_parameters(model)

    # Set SimulationParameters for the polygonal path
    polygon_sim_duration = 10.0 # seconds
    sim_params = get_simulation_parameters(model)
    sim_params.num_steps = int(polygon_sim_duration / sim_params.dt)
    
    simulation_log = run_simulation(model, data, q_init, "pw", sim_params, polygon_path_obj, 
                                    max_acceleration=max_acceleration, save_operational_space=True,
                                    use_viewer=use_viewer, experiment_dir=experiment_dir)

    if experiment_dir is not None:
        simulation_metrics = metrics(simulation_log, sim_params, polygon_path_obj, robot_params)
        log_dir = experiment_dir / "logs"
        metrics_dir = experiment_dir / "metrics"
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        save_log(log_dir / "log_0.csv", simulation_log, sim_params)
        save_metrics(metrics_dir / "metric_0.csv", simulation_metrics, sim_params)
    
    if verbose:
        print("Polygonal Path Following simulation complete.\n")
    
    return simulation_log, sim_params


# ============================================
# Plot Results for Polygonal Path Following
# ============================================
def plot_results(
    simulation_log: list,
    sim_params: SimulationParameters,
    polygon_path_obj: PolygonPath,
    robot_params: RobotParameters,
    experiment_dir=None,
    save_individual_figures=False,
):
    
    # Extract logs for joint angles, torques, and end-effector positions
    x_log = np.array([sim_result.x for sim_result in simulation_log])
    q_log = np.array([sim_result.q for sim_result in simulation_log])
    torque_log = np.array([sim_result.tau for sim_result in simulation_log])
    
    # Plot combined results for polygonal path following
    plot_polygonal_path_results(
        x_log, q_log, torque_log, polygon_path_obj, sim_params, robot_params,
        experiment_dir=experiment_dir,
        save_individual_figures=save_individual_figures,
    )


# ============================================
# Generate GIF for Polygonal Path Following
# ============================================
def generate_GIF(simulation_log: list, sim_params: SimulationParameters, polygon_path_obj: PolygonPath, robot_params: RobotParameters):
    # Extract logs for joint angles
    q_log = np.array([sim_result.q for sim_result in simulation_log])
    q1_log_polygon = q_log[:, 0]
    q2_log_polygon = q_log[:, 1]
        
    desired_polygon_path_x_gif = []
    desired_polygon_path_z_gif = []

    # Iterate through each segment of the polygon path to get the full path
    for segment in polygon_path_obj._segments:
        segment_length = segment.length
        # Generate points along the segment for visualization
        if segment_length > 0:
            num_points_per_segment = int(segment_length / (polygon_path_obj.desired_speed * sim_params.dt)) + 2 # Add a few extra points
            # Use np.linspace to get points along the segment
            for k in range(num_points_per_segment):
                alpha = k / (num_points_per_segment - 1)
                point = segment.start_point + alpha * segment.direction
                desired_polygon_path_x_gif.append(point[0])
                desired_polygon_path_z_gif.append(point[1])
        else:
            # For zero-length segments (if start and end are the same), just add the point once
            desired_polygon_path_x_gif.append(segment.start_point[0])
            desired_polygon_path_z_gif.append(segment.start_point[1])

    # Call the GIF generation function for the polygonal path
    generate_robot_arm_full_trajectory_gif(
        q1_log_polygon,
        q2_log_polygon,
        sim_params,
        output_gif_filename='experiment/results/robot_arm_polygonal_path.gif',
        desired_total_gif_frames=200,
        full_desired_path_x=desired_polygon_path_x_gif,
        full_desired_path_z=desired_polygon_path_z_gif
    )

# ============================================
# Main function to run the polygonal path following experiment
# ============================================
def main(use_viewer = True, generate_plot = True, generate_gif = True):
   
     # Define the vertices of a polygonal path
    polygon_vertices = regular_polygon(1.5, 5)
    
    # Initial joint angles
    q_init = inverse_kinematics(polygon_vertices[0], 1.0, 1.0, polygon_vertices[0])

    # Define desired speed and maximum acceleration for the polygonal path
    desired_speed = 3.0
    max_acceleration = 10.0

    # Create a PolygonalPath object
    polygon_path_obj = PolygonPath(vertices = polygon_vertices, desired_speed = desired_speed, 
                                   stop_at_vertices = True, stop_at_end = True)

    print(f"\nPolygonal Path Parameters: Vertices=\n{polygon_path_obj.vertices}")
    
    
    if use_viewer:
        print("Running simulation with viewer...")
    else:
        print("Running simulation without viewer...")
    
    # Run the simulation for polygonal path following
    experiment_dir = create_experiment_dir("PolygonPath")
    simulation_log, sim_params = run_experiment(
        q_init, polygon_path_obj, max_acceleration=max_acceleration,
        verbose=True, use_viewer=use_viewer, experiment_dir=experiment_dir,
    )
    
    # Create RobotParameters object for plotting and GIF generation
    RobotParameters_obj = get_robot_parameters(load_model_and_data(MODEL_PATH)[0])
    
    # Plot results for polygonal path following
    if generate_plot:
        print("Generating plots...")
        plot_results(
            simulation_log, sim_params, polygon_path_obj, RobotParameters_obj,
            experiment_dir=experiment_dir,
        )
    
    # Generate GIF for polygonal path following
    if generate_gif:
        print("Generating GIF...")
        generate_GIF(simulation_log, sim_params, polygon_path_obj, RobotParameters_obj)

if __name__ == "__main__":
    run_experiment_cli(main)
