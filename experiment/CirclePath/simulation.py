# @title
import numpy as np
from src.parameters import RobotParameters, SimulationParameters
from src.mujoco_setup import MODEL_PATH, load_model_and_data, get_robot_parameters, get_simulation_parameters
from src.path.CirclePath import CirclePath
from src.metrics.metrics import metrics
from src.utils import create_experiment_dir, save_log, save_metrics

from simulation.run_simulation import run_simulation

from experiment.gif_generation import generate_robot_arm_full_trajectory_gif
from experiment.experiment_runner import run_experiment_cli
from experiment.CirclePath.plot_combined_results import plot_combined_results

# ============================================
# Run Simulation for Circular Path Tracking
# ============================================
def run_experiment(q_init: np.ndarray, circular_path_obj: CirclePath, 
                   max_acceleration = 10.0, verbose = True, use_viewer = False,
                   experiment_dir = None):
    # Load the Mujoco model and data
    model, data = load_model_and_data(MODEL_PATH)
    
    sim_params = get_simulation_parameters(model)
    sim_params.num_steps = int(20/sim_params.dt)

    simulation_log = run_simulation(model, data, q_init, "pw", sim_params, circular_path_obj, 
                                    max_acceleration = max_acceleration, save_operational_space = True,
                                    use_viewer = use_viewer, experiment_dir = experiment_dir)
    
    if experiment_dir is not None:
        simulation_metrics = metrics(simulation_log, circular_path_obj, experiment_dir)

        log_dir = experiment_dir / "logs"
        metrics_dir = experiment_dir / "metrics"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        save_log(log_dir / "log_0.csv", simulation_log, sim_params)
        save_metrics(metrics_dir / "metric_0.csv", simulation_metrics, sim_params)
    else:
        simulation_metrics = None
    
    if verbose:
        print("Circular Path Tracking simulation complete.\n")
    
    return simulation_log, simulation_metrics, sim_params

# ============================================
# Generate GIF for Circular Path Tracking
# ============================================
def generate_GIF(simulation_log: list, sim_params: SimulationParameters, circular_path_obj: CirclePath, robot_params: RobotParameters):
    # Extract logs for joint angles
    q_log = np.array([sim_result.q for sim_result in simulation_log])

    # Call the GIF generation function for the circular path
    q1_log_circular = q_log[:, 0]
    q2_log_circular = q_log[:, 1]

    # Generate the full desired circular path for plotting
    theta = np.linspace(0, 2 * np.pi, 500) # More points for a smooth circle
    full_circular_path_x = circular_path_obj.center[0] + circular_path_obj.radius * np.cos(theta)
    full_circular_path_z = circular_path_obj.center[1] + circular_path_obj.radius * np.sin(theta)


    generate_robot_arm_full_trajectory_gif(
        q1_log_circular,
        q2_log_circular,
        sim_params,
        output_gif_filename='experiment/results/robot_arm_circular_path.gif',
        desired_total_gif_frames=200,
        full_desired_path_x=full_circular_path_x,
        full_desired_path_z=full_circular_path_z
    )


def main(use_viewer = True, generate_plot = True, generate_gif = True):
    
    # Define initial joint angles (q1, q2)
    q_init = np.array([np.pi/6, 2*np.pi/3 ])
    
    # Define the circular path parameters
    circular_path_obj = CirclePath(center=np.array([0.3, 0.0]), radius=0.7, cruise_angular_speed=1.0)
    print(f"Circular Path Parameters: Center={circular_path_obj.center}, Radius={circular_path_obj.radius}")
    
    # Set a maximum acceleration
    max_acceleration = 15.0
    
    if use_viewer:
        print("Running simulation with viewer...")
    else:
        print("Running simulation without viewer...")
    
    # Run the simulation for circular path tracking
    experiment_dir = create_experiment_dir("CirclePath")
    simulation_log, simulation_metrics, sim_params = run_experiment(
        q_init, circular_path_obj, max_acceleration=max_acceleration,
        use_viewer=use_viewer, experiment_dir=experiment_dir,
    )
    
    # Create RobotParameters object for plotting and GIF generation
    RobotParameters_obj = get_robot_parameters(load_model_and_data(MODEL_PATH)[0])
    
    # Plot results for circular path tracking
    if generate_plot:
        print("Generating plots...")
        plot_combined_results(
            simulation_log=simulation_log,
            simulation_metrics=simulation_metrics,
            q_init=q_init,
            circular_path_obj=circular_path_obj,
            sim_params=sim_params,
            robot_params=RobotParameters_obj,
            experiment_dir=experiment_dir,
        )
    
    # Generate GIF for circular path tracking
    if generate_gif:
        print("Generating GIF...")
        generate_GIF(simulation_log, sim_params, circular_path_obj, RobotParameters_obj)

if __name__ == "__main__":
    run_experiment_cli(main)
