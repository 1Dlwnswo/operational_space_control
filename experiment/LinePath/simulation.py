from typing import Optional

import numpy as np

from experiment.experiment_runner import run_experiment_cli
from experiment.gif_generation import generate_robot_arm_full_trajectory_gif
from experiment.LinePath.generate_grid import uniform_grid
from experiment.LinePath.plot_combined_results import plot_combined_results
from simulation.run_simulation import run_simulation
from src.metrics.metrics import SimulationMetrics
from src.mujoco_setup import (
    MODEL_PATH,
    get_robot_parameters,
    get_simulation_parameters,
    load_model_and_data,
)
from src.parameters import RobotParameters, SimulationParameters
from src.path.LinePath import LinePath
from src.robot.arm_2dof.kinematics import get_end_effector_position
from src.utils import create_experiment_dir, save_log, save_metrics


# ============================================
# Run simulation for multiple target positions
# ============================================
def run_experiment(
    q_init: np.ndarray,
    target_positions: np.ndarray,
    robot_params: Optional[RobotParameters] = None,
    cruise_speed=2.0,
    max_acceleration=10.0,
    verbose=False,
    use_viewer=False,
    experiment_dir=None,
):
    # Load the Mujoco model and data
    model, data = load_model_and_data(MODEL_PATH)

    if robot_params is None:
        robot_params = get_robot_parameters(model)

    # Define simulation parameters based on cruise speed and max acceleration
    sim_params = get_simulation_parameters(model)
    sim_params.num_steps = int(4000 / cruise_speed + 3000 * cruise_speed / max_acceleration)

    # Initialize logs for all simulations
    simulation_logs = []
    simulation_metrics = []

    if verbose:
        print(f"Number of OSC simulations: {len(target_positions)}")
        simulation_time = sim_params.dt * sim_params.num_steps
        print(f"Simulation time: {simulation_time:.4f} s")

    # Run simulation for each target position
    for idx, (target_x, target_z) in enumerate(target_positions):
        ee_start_pos = get_end_effector_position(q_init[0], q_init[1], robot_params.L1, robot_params.L2)

        path_obj = LinePath(
            start_point=ee_start_pos,
            end_point=np.array([target_x, target_z]),
            cruise_speed=cruise_speed,
        )

        sim_log = run_simulation(
            model,
            data,
            q_init,
            "pw",
            sim_params,
            path_obj,
            max_acceleration=max_acceleration,
            save_operational_space=True,
            use_viewer=use_viewer,
            experiment_dir=experiment_dir,
        )

        if experiment_dir is not None:
            simulation_metric = SimulationMetrics(sim_log, path_obj, experiment_dir)

            log_dir = experiment_dir / "logs"
            metrics_dir = experiment_dir / "metrics"

            log_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            log_path = log_dir / f"log_{idx}.csv"
            metrics_path = metrics_dir / f"metric_{idx}.csv"

            save_log(log_path=log_path, simulation_log=sim_log, sim_params=sim_params)
            save_metrics(
                metrics_path=metrics_path,
                simulation_metrics=simulation_metric,
                sim_params=sim_params,
            )
        else:
            simulation_metric = None

        simulation_logs.append(sim_log)
        simulation_metrics.append(simulation_metric)

        print(f"Simulation {idx + 1}/{len(target_positions)}: RMSE = {simulation_metric.rmse:.6f}")

    if verbose:
        print("Simulation completed for all target positions.")

    return simulation_logs, simulation_metrics, sim_params


# ============================================
# Generate GIF for the full trajectory across all target positions
# ============================================
def generate_GIF(
    simulation_logs: list,
    sim_params: SimulationParameters,
    target_positions: np.ndarray,
):

    concatenated_q1_log = []
    concatenated_q2_log = []
    concatenated_target_x_log = []
    concatenated_target_z_log = []

    for log, (target_x, target_z) in zip(simulation_logs, target_positions):
        q_log = np.array([sim_result.q for sim_result in log])

        # Extend with the actual logged joint positions for this target
        concatenated_q2_log.extend(q_log[:, 1])
        concatenated_q1_log.extend(q_log[:, 0])

        # Repeat the current target for the duration of this simulation segment
        concatenated_target_x_log.extend([target_x] * len(q_log))
        concatenated_target_z_log.extend([target_z] * len(q_log))

    # Call the new function to generate the GIF
    generate_robot_arm_full_trajectory_gif(
        concatenated_q1_log,
        concatenated_q2_log,
        sim_params,
        output_gif_filename="experiment/results/robot_arm_full_trajectory.gif",
        desired_total_gif_frames=200,
        target_trajectory_x=concatenated_target_x_log,
        target_trajectory_z=concatenated_target_z_log,
    )


def main(use_viewer=True, generate_plot=True, generate_gif=True):
    # Define initial joint angles (q1, q2)
    q_init = np.array([np.pi * 5 / 12, np.pi * (1 - 10 / 12)])

    # Generate a uniform grid of target positions
    target_positions = uniform_grid(2.0, 0.8)

    if use_viewer:
        print("Running simulation with viewer...")
    else:
        print("Running simulation without viewer...")

    experiment_dir = create_experiment_dir("LinePath")

    # Run simulations for all target positions
    simulation_logs, simulation_metrics, sim_params = run_experiment(
        q_init,
        target_positions,
        cruise_speed=2.0,
        max_acceleration=10.0,
        verbose=True,
        use_viewer=use_viewer,
        experiment_dir=experiment_dir,
    )

    # Plot results for all target positions
    if generate_plot:
        print("Generating plots...")
        plot_combined_results(
            simulation_logs,
            simulation_metrics,
            q_init,
            target_positions,
            sim_params,
            get_robot_parameters(load_model_and_data(MODEL_PATH)[0]),
            experiment_dir=experiment_dir,
        )

    # Generate GIF for the full trajectory across all target positions
    if generate_gif:
        print("Generating GIF...")
        generate_GIF(simulation_logs, sim_params, target_positions)


if __name__ == "__main__":
    run_experiment_cli(main)
