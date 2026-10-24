import matplotlib.pyplot as plt
import numpy as np

from src.parameters import RobotParameters, SimulationParameters
from src.path.PolygonPath import PolygonPath
from src.robot.arm_2dof.kinematics import get_end_effector_position


def process_simulation_log(simulation_log: list, simulation_metrics):
    if not simulation_log:
        raise ValueError("simulation_log must contain at least one result.")

    q_log = np.asarray([result.q for result in simulation_log], dtype=float)
    torque_log = np.asarray([result.tau for result in simulation_log], dtype=float)
    a_cmd_log = np.asarray([result.a_cmd for result in simulation_log], dtype=float)
    x_log = np.asarray([result.x for result in simulation_log], dtype=float)
    xdot_log = np.asarray([result.xdot for result in simulation_log], dtype=float)
    xddot_log = np.asarray([result.xddot for result in simulation_log], dtype=float)

    return (
        q_log,
        torque_log,
        a_cmd_log,
        x_log,
        xdot_log,
        xddot_log,
        np.asarray(simulation_metrics.errors),
        np.asarray(simulation_metrics.distance_errors),
        float(simulation_metrics.rmse),
        float(simulation_metrics.convergence_time),
        np.asarray(simulation_metrics.manipulability),
    )


def plot_combined_results(simulation_log: list, simulation_metrics, q_init: np.ndarray, polygon_path_obj: PolygonPath, 
                          sim_params: SimulationParameters, robot_params: RobotParameters, 
                          experiment_dir=None, save_individual_figures: bool = False):
    q_log, torque_log, a_cmd_log, \
        x_log, xdot_log, xddot_log, \
            error_log, distance_error_log, rmse, \
                convergence_time, manipulability_log = process_simulation_log(simulation_log, simulation_metrics)

    log_length = len(simulation_log)
    time_log = np.arange(log_length) * sim_params.dt
    if not all(
        len(values) == log_length
        for values in (
            q_log,
            torque_log,
            a_cmd_log,
            x_log,
            xdot_log,
            xddot_log,
            error_log,
            distance_error_log,
            manipulability_log,
        )
    ):
        raise ValueError("Simulation logs and metrics must have matching lengths.")

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle("Polygonal Path Tracking Results", fontsize=18)
    (
        (ax_q, ax_position, ax_torque),
        (ax_trajectory, ax_velocity, ax_error),
        (ax_manipulability, ax_a_cmd, ax_acceleration),
        (ax_summary, ax_empty_1, ax_empty_2),
    ) = axes

    sample = slice(None, None, 10)

    ax_q.plot(time_log[sample], q_log[sample, 0], label="Joint 1")
    ax_q.plot(time_log[sample], q_log[sample, 1], label="Joint 2")
    ax_q.set(title="Joint Angles", xlabel="Time [s]", ylabel="Angle [rad]")
    ax_q.grid(True)
    ax_q.legend()

    ax_position.plot(time_log[sample], x_log[sample, 0], label="End-effector X")
    ax_position.plot(time_log[sample], x_log[sample, 1], label="End-effector Z")
    ax_position.set(title="End-effector Position", xlabel="Time [s]", ylabel="Position [m]")
    ax_position.grid(True)
    ax_position.legend()

    ax_torque.plot(time_log[sample], torque_log[sample, 0], label="Joint 1")
    ax_torque.plot(time_log[sample], torque_log[sample, 1], label="Joint 2")
    ax_torque.set(title="Joint Torques", xlabel="Time [s]", ylabel="Torque [Nm]")
    ax_torque.grid(True)
    ax_torque.legend()

    vertices = np.asarray(polygon_path_obj.vertices, dtype=float)
    desired_path = vertices if polygon_path_obj.stop_at_end else np.vstack((vertices, vertices[0]))
    ax_trajectory.plot(x_log[sample, 0], x_log[sample, 1], label="Actual path")
    ax_trajectory.plot(desired_path[:, 0], desired_path[:, 1], "r--", label="Desired polygon")
    ax_trajectory.add_patch(
        plt.Circle(
            (0.0, 0.0), robot_params.L1 + robot_params.L2,
            facecolor="gray", alpha=0.1, edgecolor="black", linestyle="--",
        )
    )
    ax_trajectory.set(title="End-effector Trajectory", xlabel="X Position [m]", ylabel="Z Position [m]")
    ax_trajectory.set_aspect("equal", adjustable="box")
    ax_trajectory.grid(True)
    ax_trajectory.legend()

    speed = np.linalg.norm(xdot_log, axis=1)
    ax_velocity.plot(time_log[sample], xdot_log[sample, 0], label="X Velocity")
    ax_velocity.plot(time_log[sample], xdot_log[sample, 1], label="Z Velocity")
    ax_velocity.plot(time_log[sample], speed[sample], "--", label="Speed")
    ax_velocity.axhline(polygon_path_obj.desired_speed, color="orange", linestyle=":", label="Desired speed")
    ax_velocity.set(title="End-effector Velocity", xlabel="Time [s]", ylabel="Velocity [m/s]")
    ax_velocity.grid(True)
    ax_velocity.legend(fontsize="small")

    ax_error.plot(time_log[sample], error_log[sample], label="Path error")
    ax_error.plot(time_log[sample], distance_error_log[sample], label="Distance error", alpha=0.75)
    ax_error.set(title="Polygon Path Tracking Error", xlabel="Time [s]", ylabel="Error [m]")
    ax_error.grid(True)
    ax_error.legend(fontsize="small")

    ax_manipulability.plot(time_log[sample], manipulability_log[sample])
    ax_manipulability.set(title="Manipulability √det(JJᵀ)", xlabel="Time [s]", ylabel="Manipulability")
    ax_manipulability.grid(True)

    acceleration_command_magnitude = np.linalg.norm(a_cmd_log.reshape(log_length, -1), axis=1)
    ax_a_cmd.plot(time_log[sample], acceleration_command_magnitude[sample])
    ax_a_cmd.set(title="Acceleration Command", xlabel="Time [s]", ylabel="Acceleration [m/s²]")
    ax_a_cmd.grid(True)

    ee_acceleration_magnitude = np.linalg.norm(xddot_log.reshape(log_length, -1), axis=1)
    ax_acceleration.plot(time_log[sample], ee_acceleration_magnitude[sample])
    ax_acceleration.set(title="EE Acceleration Magnitude", xlabel="Time [s]", ylabel="Acceleration [m/s²]")
    ax_acceleration.grid(True)

    initial_ee_position = get_end_effector_position(
        q_init[0], q_init[1], robot_params.L1, robot_params.L2
    )
    ax_summary.axis("off")
    ax_summary.set_title("Metric Summary")
    ax_summary.text(
        0.03, 0.95,
        f"Path tracking RMSE: {rmse:.4f} m\n"
        f"EE convergence time (±0.01 m): {convergence_time:.3f} s\n"
        f"Initial EE position: ({initial_ee_position[0]:.3f}, {initial_ee_position[1]:.3f}) m",
        va="top", transform=ax_summary.transAxes,
    )
    ax_empty_1.axis("off")
    ax_empty_2.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if experiment_dir is None:
        fig.savefig("experiment/results/polygonal_path_simulation_results.png", dpi=300, bbox_inches="tight")
    else:
        figures_dir = experiment_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / "combined_figures.png", dpi=300, bbox_inches="tight")
        if save_individual_figures:
            individual_figures = [
                ("joint_angles.png", ax_q),
                ("ee_position.png", ax_position),
                ("joint_torques.png", ax_torque),
                ("ee_trajectory.png", ax_trajectory),
                ("ee_velocity_and_speed.png", ax_velocity),
                ("path_tracking_error.png", ax_error),
                ("manipulability.png", ax_manipulability),
                ("acceleration_command.png", ax_a_cmd),
                ("ee_acceleration_magnitude.png", ax_acceleration),
                ("metric_summary.png", ax_summary),
            ]
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for filename, axis in individual_figures:
                bbox = axis.get_tightbbox(renderer)
                if bbox is None:
                    continue
                bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(0.1)
                fig.savefig(figures_dir / filename, dpi=300, bbox_inches=bbox)

    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)
