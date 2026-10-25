# Operational Space Control for a 2-DOF Manipulator

A Python and MuJoCo project for implementing **Operational Space Control
(OSC)** and studying Cartesian path-following behavior in a planar
two-link robotic arm.

The project is being developed as an experimental control framework:
begin with a model-based controller, establish reproducible baselines,
and then investigate where model mismatch, actuator constraints, and
kinematic singularities affect performance. Learning-based compensation
is a possible later extension, not part of the current controller.

## Goals

-   Implement the kinematics and dynamics used by a 2-DOF manipulator
    controller.
-   Track Cartesian line, circular, and polygonal paths in MuJoCo.
-   Separate path/reference generation, control, simulation, and
    evaluation code.
-   Record simulation metrics and visualizations for repeatable
    experiments.
-   Investigate torque saturation, singularity handling, and robustness
    to model mismatch, disturbances, and sensor noise.
-   Eventually compare model-based control with learning-based methods
    under clearly defined experimental conditions.

## Current Scope

The current project includes:

-   A planar 2-DOF MuJoCo robot model.
-   Forward-kinematics/Jacobian utilities and a separately implemented
    manipulator dynamics model.
-   An OSC torque calculation using operational-space inertia and
    compensation terms.
-   Cartesian acceleration-command generation for path following.
-   Line, circle, and polygon path experiments.
-   Simulation logging, metrics, plots, and optional GIF generation.
-   Unit tests for core utilities, paths, control, metrics, simulation,
    and experiment modules.

This is an actively developing research/learning project. The existence
of a working simulation or a low path-error value under one
configuration does **not** by itself establish robustness, stability
guarantees, or superiority over another controller. Quantitative claims
should be tied to documented settings and repeatable comparisons.

## Repository Structure

``` text
operational_space_control/
├── model/
│   └── arm_2dof.xml              # MuJoCo model
├── src/
│   ├── control/
│   │   ├── osc.py                # OSC parameter and torque calculations
│   │   ├── acceleration_command.py
│   │   └── torque_limiter.py
│   ├── robot/
│   │   └── arm_2dof/
│   │       ├── kinematics.py     # Forward kinematics, Jacobian, J-dot
│   │       ├── dynamics.py       # Inertia, Coriolis/centrifugal, gravity terms
│   │       └── sensors.py
│   ├── path/
│   │   ├── LinePath.py
│   │   ├── CirclePath.py
│   │   └── PolygonPath.py
│   ├── metrics/
│   │   └── metrics.py            # Path error, RMSE, convergence, manipulability
│   ├── mujoco_setup.py
│   ├── parameters.py
│   └── utils.py                  # Shared simulation/experiment utilities
├── simulation/
│   ├── perform_simulation_step.py
│   └── run_simulation.py
├── experiment/
│   ├── LinePath/
│   │   ├── simulation.py
│   │   └── plot_combined_results.py
│   ├── CirclePath/
│   │   ├── simulation.py
│   │   └── plot_combined_results.py
│   ├── PolygonPath/
│   │   ├── simulation.py
│   │   └── plot_combined_results.py
│   ├── experiment_runner.py
│   └── gif_generation.py
├── tests/
├── run.py                        # Interactive experiment selector
├── pyproject.toml
├── uv.lock
└── TODO.md
```

Experiment outputs are written beneath the relevant
`experiment/<PathName>/results/` directory, organized by run timestamp.
Depending on the experiment and enabled outputs, a run may contain
configuration data, metric CSV files, and figures. Some generated
results are excluded from Git by `.gitignore`.

## Controller Overview

The control pipeline is organized around the following flow:

1.  The path object generates a desired Cartesian position, velocity,
    and acceleration.
2.  The acceleration-command module combines feedforward acceleration
    with proportional and derivative feedback.
3.  The OSC module uses the robot state and model parameters to
    calculate an operational-space force and map it to joint torques.
4.  The simulation applies the actuator commands in MuJoCo and records
    the resulting state.
5.  Metrics and plotting utilities analyze the recorded run.

At a high level, the operational-space calculation follows the structure

\[ `\Lambda `{=tex}= (J M\^{-1} J^T)^{-1}, \]

\[ F = `\Lambda `{=tex}a\_{`\mathrm{cmd}`{=tex}} + `\mu `{=tex}+ p, \]

\[ `\tau `{=tex}= J\^T F, \]

where (M) is the joint-space inertia matrix, (J) is the end-effector
Jacobian, (a\_{`\mathrm{cmd}`{=tex}}) is the commanded Cartesian
acceleration, and (`\mu`{=tex}) and (p) represent implemented
velocity-dependent and gravity compensation terms.

The implementation includes numerical regularization in the
operational-space inertia calculation. Its behavior near singular
configurations must be evaluated experimentally; regularization alone
should not be interpreted as a complete singularity-handling strategy.

## Robot and Simulation

The current MuJoCo model is a planar arm with two revolute joints and
motor actuation. The model file is `model/arm_2dof.xml`; model geometry
and actuator constraints are defined there. The simulation setup and
controller parameters are maintained separately in `src/`.

Important consistency checks for future experiments include:

-   Matching the dynamics parameters used by the controller to the
    MuJoCo model.
-   Distinguishing computed torque commands from the control input
    actually applied after actuator limits.
-   Recording the exact initial state, path parameters, simulation
    timestep, controller parameters, and actuator limits for each run.

## Setup

The project uses `uv` to manage the Python environment and locked
dependencies.

Install [uv](https://docs.astral.sh/uv/) if it is not already available,
then run these commands from the repository root:

``` bash
uv sync
```

To run the test suite:

``` bash
uv run pytest
```

The project dependencies and Python version requirement are declared in
`pyproject.toml`; `uv.lock` records the resolved environment. A MuJoCo
viewer requires a graphical environment. Headless runs should disable
the viewer.

## Running Experiments

### Interactive selector

Run the menu in `run.py`:

``` bash
uv run python run.py
```

The menu lets you select the circle, line, or polygon experiment. The
default viewer/plot/GIF settings are configured near the top of
`run.py`.

### Run one experiment directly

Run commands from the repository root:

``` bash
# Circular path
uv run python -m experiment.CirclePath.simulation --no-viewer --no-plot --no-gif

# Straight-line path
uv run python -m experiment.LinePath.simulation --no-viewer --no-plot --no-gif

# Polygonal path
uv run python -m experiment.PolygonPath.simulation --no-viewer --no-plot --no-gif
```

To enable the MuJoCo viewer, replace `--no-viewer` with `--viewer`. To
generate plots or a GIF, replace `--no-plot` or `--no-gif` with `--plot`
or `--gif`, respectively.

For example, to run circular path tracking with the viewer but without
generating plots or a GIF:

``` bash
uv run python -m experiment.CirclePath.simulation --viewer --no-plot --no-gif
```

The exact options accepted by each experiment are provided by the shared
experiment CLI utility. Run commands from the repository root so that
Python module imports and model paths resolve correctly.

## Evaluation Metrics

The current metrics module includes:

-   **RMSE:** root-mean-square error for the error definition used by
    the path type.
-   **Distance error:** distance to the endpoint for line-path
    evaluation, radial deviation for circle paths, or path deviation for
    polygon paths.
-   **Convergence time:** time associated with the last error sample
    outside a specified threshold.
-   **Manipulability:** a Jacobian-based measure derived from
    (`\sqrt{\det(JJ^T)}`{=tex}).

Metric definitions depend on the path type. In particular, path
deviation is not equivalent to time-indexed trajectory-tracking error.
For rigorous comparisons, report the metric definition and units, and
consider adding time-indexed position/velocity errors, maximum error,
endpoint error, torque usage, and the fraction of time spent at actuator
limits.

## Testing

Run all available tests with:

``` bash
uv run pytest
```

The test suite is intended to check core calculations and
simulation/experiment behavior. A passing test suite does not replace
validation against MuJoCo's dynamics or controlled experiments near
singularities and actuator limits.

## Research Questions

The project is organized around these questions:

1.  Under what conditions does OSC fail to track a Cartesian path or
    trajectory?
2.  How do path geometry and acceleration/velocity reference design
    affect tracking?
3.  How do torque limits change tracking performance and controller
    behavior?
4.  How does performance change near kinematic singularities?
5.  How sensitive is the controller to errors in mass, inertia, or other
    model parameters?
6.  What changes when disturbances or sensor noise are introduced?
7.  Can adaptive or learning-based residual compensation improve a
    model-based baseline enough to justify its added complexity?

## Planned Work

Recommended order of work:

1.  **Validate the implementation:** test torque-limiter behavior and
    compare controller dynamics calculations against MuJoCo under
    matched states and parameters.
2.  **Establish a reproducible baseline:** standardize initial
    conditions, path definitions, timestep, actuator constraints, and
    metric definitions.
3.  **Improve evaluation:** distinguish geometric path error from
    time-indexed trajectory error; record torque saturation and endpoint
    behavior.
4.  **Study limitations:** vary model parameters, introduce
    disturbances/noise, and test configurations with low manipulability.
5.  **Compare controller designs:** change one factor at a time and
    preserve configurations and outputs for each run.
6.  **Explore learning-based compensation:** only after a clear
    model-based baseline and measurable limitation have been
    established.

See [`TODO.md`](TODO.md) for current implementation tasks.

## Reproducibility and Reporting

For each experiment, preserve at least:

-   Path type and all path parameters.
-   Initial joint positions and velocities.
-   Simulation timestep and duration.
-   Controller gains and acceleration limits.
-   Controller model parameters and MuJoCo model version.
-   Actuator constraints.
-   Metric definitions and output files.

A plot or summary number should be accompanied by the configuration that
produced it. When comparing methods, use the same test conditions and
report both performance and constraint violations.

## License

No license has been specified yet. Add a license file before
distributing or reusing the project as an open-source package.