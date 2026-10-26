# Operational Space Control for a 2-DOF Manipulator

A Python and MuJoCo simulation project for implementing **operational space control (OSC)** and evaluating Cartesian path-following behavior on a planar two-link robot arm.

The project is a model-based control baseline and an experimental framework. The immediate focus is to make the controller, reference generation, simulation, and evaluation reproducible before studying model mismatch, actuator constraints, and configurations near kinematic singularities. Learning-based compensation is a possible future extension, not part of the current controller.

## Current Features

- Planar 2-DOF manipulator modeled in MuJoCo.
- Forward kinematics, Jacobian, Jacobian derivative, and a separately implemented manipulator dynamics model.
- OSC torque computation with operational-space inertia and compensation terms.
- Cartesian acceleration-command generation with feedback for path following.
- Line, circular, and polygonal path experiments.
- Optional MuJoCo viewer, result plots, and GIF generation.
- Timestamped experiment directories containing configuration, logs, metrics, and figures.
- Unit tests for core kinematics, control, path, metrics, simulation, utilities, and experiment behavior.

The project is under active development. A successful simulation in one configuration does not establish robustness, stability guarantees, or superiority over another controller. Quantitative claims should be tied to repeatable experiments and documented settings.

## Control Pipeline

1. The path object generates a Cartesian reference.
2. The acceleration-command module combines reference acceleration with position and velocity feedback.
3. The OSC controller maps the Cartesian command to joint torques using the robot state and model parameters.
4. MuJoCo advances the plant and the simulation records state and control signals.
5. Metrics and plots summarize path error, torque usage, saturation, and manipulability.

The operational-space dynamics use the standard structure

\[
\Lambda = (J M^{-1}J^T)^{-1}, \qquad
F = \Lambda a_{\mathrm{cmd}} + \mu + p, \qquad
\tau = J^T F,
\]

where \(M\) is the joint-space inertia matrix, \(J\) is the end-effector Jacobian, \(a_{\mathrm{cmd}}\) is the commanded Cartesian acceleration, and \(\mu\) and \(p\) denote the implemented velocity-dependent and gravity-compensation terms.

The implementation regularizes the operational-space inertia calculation numerically. Regularization by itself is not a complete singularity-handling strategy; behavior near low-manipulability configurations remains an evaluation target.

## Repository Layout

```text
.
├── model/arm_2dof.xml
├── src/
│   ├── control/                 # OSC, acceleration command, torque limiting
│   ├── metrics/                 # Path and simulation metrics
│   ├── path/                    # Line, circle, and polygon references
│   ├── robot/arm_2dof/          # Kinematics, dynamics, sensors
│   ├── mujoco_setup.py
│   ├── parameters.py
│   └── utils.py
├── simulation/                  # Simulation-step and run orchestration
├── experiment/
│   ├── LinePath/
│   ├── CirclePath/
│   ├── PolygonPath/
│   ├── experiment_runner.py
│   └── gif_generation.py
├── tests/
├── run.py                       # Interactive experiment selector
├── pyproject.toml
├── uv.lock
└── TODO.md
```

## Setup

The project uses `uv` to manage Python and project dependencies. Python 3.8.10 is currently pinned in `.python-version` and the CI workflow.

Install `uv` using the instructions in the [official uv documentation](https://docs.astral.sh/uv/), then run the following commands from the repository root:

```bash
uv sync --locked
```

Run the test suite:

```bash
uv run --locked pytest
```

A graphical environment is required for the MuJoCo viewer. Use `--no-viewer` for headless runs.

## Running Experiments

### Interactive selector

```bash
uv run python run.py
```

Select the circle, line, or polygon experiment from the menu. Default viewer, plot, and GIF settings are defined near the top of `run.py`.

### Run a specific experiment

Run commands from the repository root:

```bash
# Circle path
uv run python -m experiment.CirclePath.simulation --no-viewer --plot --no-gif

# Line path
uv run python -m experiment.LinePath.simulation --no-viewer --plot --no-gif

# Polygon path
uv run python -m experiment.PolygonPath.simulation --no-viewer --plot --no-gif
```

Available options are `--viewer` / `--no-viewer`, `--plot` / `--no-plot`, and `--gif` / `--no-gif`. For example, to run without a viewer or generated outputs, use:

```bash
uv run python -m experiment.CirclePath.simulation --no-viewer --no-plot --no-gif
```

Each run writes to a timestamped directory under `experiment/<PathName>/results/`. Outputs can include `config.json`, logs, metric data, figures, and an optional GIF. Generated results are intended to remain outside source control unless selected deliberately as reference artifacts.

## Evaluation Metrics

The current metrics implementation includes:

- **RMSE** and **maximum tracking/path error** for the path-error definition used by the path object.
- **Threshold-violation ratio** and a corresponding success flag based on the current configured criterion.
- **Convergence time** for the line-path endpoint metric.
- **RMS joint torque**, **RMS torque rate**, and **torque-saturation ratio**.
- **Manipulability**, computed from the end-effector Jacobian as \(\sqrt{\det(JJ^T)}\).

Metric interpretation depends on the path type. Geometric path error is not the same as time-indexed trajectory-tracking error, and the current convergence-time metric is not implemented uniformly for every path type. Consult the implementation before comparing values across experiments. These definitions and edge cases remain part of the ongoing validation work.

## Research Questions

1. Under what conditions does OSC fail to follow a Cartesian path?
2. How do path geometry and reference velocity/acceleration affect tracking?
3. How do torque limits affect tracking performance and control behavior?
4. How does performance change near low-manipulability configurations?
5. How sensitive is the controller to errors in mass, inertia, or center-of-mass parameters?
6. What changes under external disturbances or sensor noise?
7. Can adaptive or learning-based residual compensation improve a measured model-based limitation enough to justify its added complexity?

## Reproducibility and Limitations

For meaningful comparisons, record and hold constant the initial joint state, path parameters, simulation timestep and duration, controller gains, robot parameters, and actuator constraints. Preserve each run's configuration alongside its logs and metrics.

The current project is a simulation-based implementation, not a validated hardware controller. Dynamics consistency, torque-limiter behavior, metric definitions, reproducibility, and performance near singularities and actuator limits require further verification. See [TODO.md](TODO.md) for the current validation plan.

## License

No license has been selected yet. Until a license is added, do not assume that the code is available for unrestricted reuse, modification, or redistribution.