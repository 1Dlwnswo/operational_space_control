from typing import Optional

import numpy as np

from experiment.LinePath.generate_grid import uniform_grid
from experiment.LinePath.simulation import plot_combined_results as plot_linepath
from experiment.LinePath.simulation import run_experiment as run_linepath_sim
from src.mujoco_setup import MODEL_PATH, get_robot_parameters, load_model_and_data
from src.utils import create_experiment_dir


def linepath_cruise_speed_sweep(
    q_init: np.ndarray,
    target_positions: np.ndarray,
    cruise_speeds: Optional[np.ndarray],
    max_accelerations: Optional[np.ndarray],
    generate_plot: Optional[np.ndarray],
):
    generate_plots = np.asarray(generate_plot)
    
    if generate_plots.ndim != 2:
        raise ValueError("generate_plot must be a 2D array.")
    
    if len(cruise_speeds) != generate_plots.shape[0]:
        raise ValueError("The number of rows in generate_plot must match the number of cruise_speeds.")
    
    if len(max_accelerations) != generate_plots.shape[1]:
        raise ValueError("The number of columns in generate_plot must match the number of max_accelerations values.")
    
    # Don't save experiment logs and metrics
    experiment_dir = create_experiment_dir("sweeps")
    
    for speed_index, cruise_speed in enumerate(cruise_speeds):
        for acceleration_index, max_acceleration in enumerate(max_accelerations):
            current_sweep_dir = experiment_dir / f"LinePath/cs_{cruise_speed:g}_ma_{max_acceleration:g}"
            current_sweep_dir.mkdir(parents=True, exist_ok=True)
            
            simulation_logs, simulation_metrics, sim_params = run_linepath_sim(
                q_init,
                target_positions,
                cruise_speed=cruise_speed,
                max_acceleration=max_acceleration,
                experiment_dir=current_sweep_dir,
            )
            
            average_rmse = np.mean([metric.rmse for metric in simulation_metrics])
            print(f"Average RMSE: {average_rmse:.6f} m")
            
            for target_position, metric in zip(target_positions, simulation_metrics):
                if not metric.success:
                    print(
                        f"Unsuccessful target position: {target_position}, "
                        f"cruise_speed: {cruise_speed}, "
                        f"max_acceleration: {max_acceleration}"
                    )
            

            if generate_plots[speed_index, acceleration_index]:
                plot_linepath(
                    simulation_logs,
                    simulation_metrics,
                    q_init,
                    target_positions,
                    sim_params,
                    get_robot_parameters(load_model_and_data(MODEL_PATH)[0]),
                    experiment_dir=current_sweep_dir,
                )

def main():
    # Define initial joint angles (q1, q2)
    q_init = np.array([np.pi * 5 / 12, np.pi * (1 - 10 / 12)])

    # Generate a uniform grid of target positions
    target_positions = uniform_grid(2.0, 1.2)
    
    cruise_speeds = np.linspace(0.8, 4.0, 5)
    max_accelerations = np.array([10.0])
    generate_plot=np.zeros((len(cruise_speeds), len(max_accelerations)))
    
    print("target positions:", target_positions)
    print("cruise_speeds:", cruise_speeds)
    print("max_accelerations:", max_accelerations)
    
    linepath_cruise_speed_sweep(
        q_init=q_init,
        target_positions=target_positions,
        cruise_speeds=cruise_speeds,
        max_accelerations=max_accelerations,
        generate_plot=generate_plot,
    )


if __name__ == "__main__":
    main()
    
