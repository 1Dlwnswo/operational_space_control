# Operational Space Control

A MuJoCo-based project for implementing Operational Space Control (OSC) on a 2-DOF robotic manipulator and investigating Cartesian path-tracking behavior through simulation.

## Overview

The goal is to build and evaluate model-based controllers from first principles, using simulation to study manipulator trajectory tracking and controller limitations.

Current areas of interest include:

- Cartesian-space path tracking
- Operational Space Control
- Trajectory and velocity-profile design
- Interaction between torque commands and robot dynamics
- Kinematic singularities and inverse-kinematics branch changes
- Future robustness evaluation under model mismatch, external disturbances, and sensor noise

The project is currently focused on developing the controller and simulation workflow. Quantitative tracking errors and comparative performance results have not yet been documented.

## Environment

- Python
- MuJoCo
- NumPy
- Matplotlib
- `uv` for Python environment and dependency management

Dependency versions are managed through `pyproject.toml` and `uv.lock`.

## Robot Model

The current simulation uses a planar robotic arm with two revolute joints.

- Physics engine: MuJoCo
- Degrees of freedom: 2
- Actuation: Joint motor torque
- Gravity: `-9.81 m/s²`
- Simulation timestep: `0.001 s`
- End-effector workspace: X-Z plane

The end-effector position is computed from the joint configuration. Link dimensions and physical parameters are defined by the MuJoCo model.

## Experiments

### Circular Path Tracking

The CirclePath experiment tracks a circular reference path.

Current progress:
- CirclePath experiment module is configured.
- The MuJoCo simulation runs successfully.
- Plot and GIF generation can be controlled through command-line options.
- Real-time MuJoCo viewer integration is being developed.

```bash
uv run python -m experiment.CirclePath.simulation --no-plot --no-gif
```

### Line Path Tracking

An experiment module is available for tracking a straight-line reference path.

```bash
uv run python -m experiment.LinePath.simulation --no-plot --no-gif
```

### Polygon Path Tracking

An experiment module is available for tracking a polygonal reference path.

```bash
uv run python -m experiment.PolygonPath.simulation --no-plot --no-gif
```

Tracking performance across these paths should be compared under consistent initial conditions and evaluation metrics.

## Simulation and Visualization

The simulation workflow separates execution from result visualization to make experiments easier to repeat.

Implemented or in-progress features include:

- MuJoCo physics simulation
- Optional real-time viewer
- Simulation-state logging
- Plot generation
- GIF generation
- Viewer camera configuration
- Custom world-coordinate-axis visualization

Use `--no-plot --no-gif` when plots and GIFs are not needed.

The `--viewer` flag must be registered in the relevant experiment module's command-line parser and passed to the simulation runner before it can be used. If it is not registered, the CLI will reject the option.

## Running Experiments

Run commands from the project root:

```bash
# Circular path
uv run python -m experiment.CirclePath.simulation --no-plot --no-gif

# Line path
uv run python -m experiment.LinePath.simulation --no-plot --no-gif

# Polygon path
uv run python -m experiment.PolygonPath.simulation --no-plot --no-gif
```

If viewer support is enabled in the experiment module, launch the circular-path viewer with:

```bash
uv run python -m experiment.CirclePath.simulation --viewer --no-plot --no-gif
```

## Current Status

| Item | Status |
|---|---|
| 2-DOF MuJoCo model | Implemented |
| CirclePath experiment | Runs successfully |
| LinePath experiment module | Available |
| PolygonPath experiment module | Available |
| Plot/GIF CLI options | Available |
| Real-time viewer integration | In progress / dependent on CLI integration |
| Camera viewpoint configuration | Implemented in viewer configuration |
| Custom coordinate axes | Visualization work in progress |
| Quantitative tracking metrics | Not yet evaluated |
| Torque saturation study | Planned |
| Singularity-handling evaluation | Planned |
| Robustness tests under model mismatch and disturbances | Planned |

## Planned Work

### Quantitative trajectory-tracking evaluation

- Use consistent initial conditions and trajectory parameters.
- Record Cartesian tracking error over time.
- Compare RMS error, maximum error, and settling behavior.
- Inspect joint positions, joint velocities, and actuator torques.

### Controller robustness

- Evaluate the effects of torque saturation.
- Investigate behavior near kinematic singularities.
- Introduce model-parameter mismatch and external disturbances.
- Test sensitivity to measurement noise.

### Model-based and learning-based methods

The long-term direction is to use a model-based controller as the primary control structure and investigate learning-based methods where they address specific limitations.

Possible future directions include adaptive compensation and residual learning, evaluated against a clearly defined model-based baseline.

## Research Questions

1. Under what conditions does OSC fail to track a desired Cartesian trajectory?
2. How do trajectory design and velocity/acceleration commands affect tracking?
3. How should torque saturation and singular configurations be handled?
4. How sensitive is the controller to model mismatch, disturbances, and measurement noise?
5. Where can learning-based methods provide measurable benefits over a model-based baseline?

## Results and Reproducibility

A successful simulation run confirms that the experiment pipeline executes; it does not, by itself, demonstrate tracking accuracy or robustness. Quantitative claims should be added after collecting and analyzing metrics under reproducible conditions.