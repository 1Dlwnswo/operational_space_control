import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

from src.parameters import RobotParameters, SimulationParameters
from src.path.CirclePath import CirclePath
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
        float(simulation_metrics.saturation_torque_ratio),
        np.asarray(simulation_metrics.manipulability),
    )


def plot_combined_results(simulation_log: list, simulation_metrics, q_init: np.ndarray, circular_path_obj: CirclePath, 
                          sim_params: SimulationParameters, robot_params: RobotParameters,
                          experiment_dir=None, save_individual_figures: bool = False,):
    q_log, torque_log, a_cmd_log, \
        x_log, xdot_log, xddot_log, \
            error_log, distance_error_log, rmse, \
                convergence_time, saturation_torque_ratio, manipulability_log = process_simulation_log(simulation_log, simulation_metrics)

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

    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    fig.suptitle("Circular Path Tracking Results", fontsize=18)

    ax_q, ax_position, ax_torque = axes[0]
    ax_trajectory, ax_x_fit, ax_z_fit = axes[1]
    ax_x_fft, ax_z_fft, ax_velocity = axes[2]
    ax_error, ax_manipulability, ax_a_cmd = axes[3]
    ax_acceleration, ax_summary, ax_empty = axes[4]

    # --- Plot 1: Joint Angles ---
    ax_q.plot(time_log[::10], q_log[::10, 0], label="Joint 1")
    ax_q.plot(time_log[::10], q_log[::10, 1], label="Joint 2")
    ax_q.set(title="Joint Angles", xlabel="Time [s]", ylabel="Angle [rad]")
    ax_q.grid(True)
    ax_q.legend()

    # --- Plot 2: End-effector Position ---
    ax_position.plot(time_log[::10], x_log[::10, 0], label="End-effector X")
    ax_position.plot(time_log[::10], x_log[::10, 1], label="End-effector Z")
    ax_position.set(title="End-effector Position", xlabel="Time [s]", ylabel="Position [m]")
    ax_position.grid(True)
    ax_position.legend()

    # --- Plot 3: Joint Torques ---
    ax_torque.plot(time_log[::10], torque_log[::10, 0], label="Joint 1")
    ax_torque.plot(time_log[::10], torque_log[::10, 1], label="Joint 2")
    ax_torque.set(title="Joint Torques", xlabel="Time [s]", ylabel="Torque [Nm]")
    ax_torque.grid(True)
    ax_torque.legend()

    # --- Plot 4: End-effector Trajectory ---
    center = circular_path_obj.center
    radius = circular_path_obj.radius
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    ax_trajectory.plot(x_log[::10, 0], x_log[::10, 1], label="Actual path")
    ax_trajectory.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        "r--",
        label="Desired circle",
    )
    workspace = plt.Circle(
        (0.0, 0.0), robot_params.L1 + robot_params.L2,
        facecolor="gray", alpha=0.1, edgecolor="black", linestyle="--",
    )
    ax_trajectory.add_patch(workspace)
    ax_trajectory.set(
        title="End-effector Trajectory", xlabel="X Position [m]", ylabel="Z Position [m]"
    )
    ax_trajectory.set_aspect("equal", adjustable="box")
    ax_trajectory.grid(True)
    ax_trajectory.legend()

    # --- Plot 5, 6: End-effector Sine Fitting ---
    def sine_function(t, amplitude, omega, phase, offset):
        return amplitude * np.sin(omega * t + phase) + offset

    for axis, coordinate, label, phase in (
        (ax_x_fit, 0, "X", np.pi / 2),
        (ax_z_fit, 1, "Z", 0.0),
    ):
        axis.plot(time_log[::10], x_log[::10, coordinate], label=f"Measured {label}")
        if log_length >= 4:
            guess = [radius, circular_path_obj.cruise_angular_speed, phase, center[coordinate]]
            try:
                parameters, _ = curve_fit(
                    sine_function, time_log[::10], x_log[::10, coordinate], p0=guess, maxfev=10000
                )
                axis.plot(
                    time_log[::10],
                    sine_function(time_log[::10], *parameters),
                    "--",
                    label="Sine fit",
                )
                axis.text(
                    0.98,
                    0.98,
                    f"{label}(t) = {parameters[0]:.3f} sin("
                    f"{parameters[1]:.3f}t + {parameters[2]:.3f}) "
                    f"+ {parameters[3]:.3f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.8},
                )
            except (RuntimeError, ValueError):
                pass
        axis.set(
            title=f"End-effector {label} Sine Fit",
            xlabel="Time [s]",
            ylabel=f"{label} Position [m]",
        )
        axis.grid(True)
        axis.legend(loc='right', fontsize='x-small')


    # --- Plot 7, 8: X, Z Frequency Spectrum ---    
    if log_length > 1:
        frequencies = fftfreq(log_length, sim_params.dt)[: log_length // 2]
        x_spectrum = 2.0 / log_length * np.abs(fft(x_log[:, 0])[: log_length // 2])
        z_spectrum = 2.0 / log_length * np.abs(fft(x_log[:, 1])[: log_length // 2])
        for axis, spectrum, label in (
            (ax_x_fft, x_spectrum, "X"),
            (ax_z_fft, z_spectrum, "Z"),
        ):
            axis.plot(frequencies, spectrum)
            upper_frequency = frequencies[-1]
            threshold = 0.005
            peak_candidates = np.flatnonzero(
                (frequencies > 0.0) & (spectrum >= threshold)
            )
            if peak_candidates.size:
                peak_index = peak_candidates[
                    np.argmax(spectrum[peak_candidates])
                ]
                above_after_peak = np.flatnonzero(
                    spectrum[peak_index:] > threshold
                )
                if above_after_peak.size:
                    last_above_index = peak_index + above_after_peak[-1]
                    cutoff_index = min(last_above_index + 1, len(frequencies) - 1)
                    upper_frequency = frequencies[cutoff_index]
                axis.scatter(frequencies[peak_index], spectrum[peak_index], color="red")
                axis.text(
                    0.98,
                    0.98,
                    f"Peak frequency: {frequencies[peak_index]:.3f} Hz",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.8},
                )
            axis.set(
                title=f"End-effector {label} Frequency Spectrum",
                xlabel="Frequency [Hz]",
                ylabel="Magnitude",
            )
            axis.grid(True)
            axis.set_xlim(0.0, upper_frequency)
    else:
        ax_x_fft.set_title("X Frequency Spectrum (requires multiple samples)")
        ax_z_fft.set_title("Z Frequency Spectrum (requires multiple samples)")

    # --- Plot 9: End-effector Velocity ---
    speed = np.linalg.norm(xdot_log, axis=1)
    ax_velocity.plot(time_log[::10], xdot_log[::10, 0], label="X Velocity")
    ax_velocity.plot(time_log[::10], xdot_log[::10, 1], label="Z Velocity")
    ax_velocity.plot(time_log[::10], speed[::10], "--", label="Speed")
    ax_velocity.axhline(
        circular_path_obj.cruise_angular_speed * radius,
        color="orange",
        linestyle=":",
        label="Desired cruise speed",
    )
    ax_velocity.set(title="End-effector Velocity", xlabel="Time [s]", ylabel="Velocity [m/s]")
    ax_velocity.grid(True)
    ax_velocity.legend(fontsize="small")

    # --- Plot 10: Circle Path Tracking Error ---
    ax_error.plot(time_log[::10], error_log[::10], label="Signed radial error")
    ax_error.plot(time_log[::10], distance_error_log[::10], label="Distance error", alpha=0.75)
    ax_error.set(title="Circle Path Tracking Error", xlabel="Time [s]", ylabel="Error [m]")
    ax_error.grid(True)
    ax_error.legend(fontsize="small")

    # --- Plot 11: Manipulability det(JJᵀ) ---
    ax_manipulability.plot(time_log[::10], manipulability_log[::10])
    ax_manipulability.set(
        title="Manipulability det(JJᵀ)", xlabel="Time [s]", ylabel="Manipulability"
    )
    ax_manipulability.grid(True)

    # --- Plot 12: Acceleration Command ---
    a_cmd_magnitude = np.linalg.norm(
        a_cmd_log.reshape(log_length, -1), axis=1
    )
    ax_a_cmd.plot(time_log[::10], a_cmd_magnitude[::10])
    ax_a_cmd.set(
        title="Acceleration Command",
        xlabel="Time [s]",
        ylabel="Acceleration [m/s²]",
    )
    ax_a_cmd.grid(True)

    # --- Plot 13: EE Acceleration Magnitude ---
    ee_acceleration_magnitude = np.linalg.norm(
        xddot_log.reshape(log_length, -1), axis=1
    )
    ax_acceleration.plot(time_log[::10], ee_acceleration_magnitude[::10])
    ax_acceleration.set(
        title="EE Acceleration Magnitude",
        xlabel="Time [s]",
        ylabel="Acceleration [m/s²]",
    )
    ax_acceleration.grid(True)

    # Placeholder for additional plots; add new plots here as needed.
    initial_ee_position = get_end_effector_position(
        q_init[0], q_init[1], robot_params.L1, robot_params.L2
    )
    ax_summary.axis("off")
    ax_summary.set_title("Metric Summary")
    ax_summary.text(
        0.03,
        0.95,
        f"Path tracking RMSE: {rmse:.4f} m\n"
        f"EE convergence time (±0.01 m): {convergence_time:.3f} s\n"
        f"Initial EE position: ({initial_ee_position[0]:.3f}, "
        f"{initial_ee_position[1]:.3f}) m\n"
        f"Torque saturation: {saturation_torque_ratio:.1%}",
        va="top",
        transform=ax_summary.transAxes,
    )
    ax_empty.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if experiment_dir is None:
        fig.savefig(
            "experiment/results/circular_path_simulation_results.png",
            dpi=300,
            bbox_inches="tight",
        )
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
                ("ee_x_sine_fit.png", ax_x_fit),
                ("ee_z_sine_fit.png", ax_z_fit),
                ("ee_x_frequency_spectrum.png", ax_x_fft),
                ("ee_z_frequency_spectrum.png", ax_z_fft),
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
                fig.savefig(
                    figures_dir / filename,
                    dpi=300,
                    bbox_inches=bbox,
                )

    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)
