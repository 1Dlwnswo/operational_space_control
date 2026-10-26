# TODO — Operational Space Control

This file separates release-readiness checks from the next validation tasks and longer-term research directions. Items should be marked complete only after the corresponding behavior has been checked or tested.

## 1. Public Release Checklist

- [ ] Replace machine-specific paths with repository-relative paths.
- [ ] Run the full test suite from a clean environment with the locked dependencies.
- [ ] Run Ruff and resolve actionable lint errors.
- [ ] Verify `uv sync --locked` and all README run commands from the repository root.
- [ ] Confirm `.gitignore` excludes virtual environments, caches, and generated experiment outputs without excluding source files needed to reproduce results.
- [ ] Search tracked files for credentials, private data, and unintended machine-specific paths.
- [ ] Choose and add a license if the intention is to permit reuse or redistribution.
- [ ] Review the README and TODO against the implementation immediately before publishing.

## 2. Validate Metrics and Result Reporting

- [ ] Document each metric's mathematical definition, units, and interpretation.
- [ ] Keep geometric path-following error distinct from time-indexed trajectory-tracking error.
- [ ] Decide whether to add Cartesian position-tracking RMSE against a time-indexed desired reference.
- [ ] Validate maximum error, threshold-violation ratio, and the success criterion at boundary cases.
- [ ] Make convergence-time definitions explicit for each path type; avoid interpreting an initial state inside a geometric threshold as successful endpoint convergence.
- [ ] Check short runs, invalid or empty logs, zero-length paths, and thresholds that are never reached.
- [ ] Verify that saved configuration, logs, metrics, figures, and optional GIFs correspond to the same run.
- [ ] Confirm the LinePath, CirclePath, and PolygonPath plotters work with their saved timestamped result directories.

## 3. Validate Controller and Simulation

### Torque limiting and actuator constraints

- [ ] Test the torque limiter for positive/negative torque, one or both joints exceeding the limit, and zero torque.
- [ ] Confirm whether the intended policy clips each joint independently or scales the full torque vector, and document that policy.
- [ ] Align the configured torque limit with MuJoCo actuator constraints.
- [ ] Log requested torque separately from the torque actually applied after limiting.
- [ ] Verify the saturation metric against the actual applied control input.

### Dynamics consistency

- [ ] Verify controller parameters against the MuJoCo model: mass, center of mass, inertia, geometry, joint axes, and gravity.
- [ ] Compare the manually computed inertia matrix with MuJoCo's mass matrix at selected states.
- [ ] Compare gravity and velocity-dependent bias terms against MuJoCo at matched states.
- [ ] Check sign conventions and coordinate frames in forward kinematics, Jacobian, and Jacobian derivative.
- [ ] Add numerical tests across multiple non-singular joint configurations.

### OSC numerical behavior

- [ ] Review the operational-space inertia calculation and regularization near singular configurations.
- [ ] Test sensitivity to the regularization parameter.
- [ ] Record manipulability, tracking error, and torque magnitude near low-manipulability configurations.
- [ ] Document controller assumptions and known limitations; do not equate numerical regularization with complete singularity handling.

## 4. Establish Reproducible Baselines

- [ ] Define baseline initial joint positions and velocities.
- [ ] Fix path parameters, desired speed, acceleration limits, timestep, and simulation duration for each baseline.
- [ ] Confirm metric definitions are consistent wherever cross-path comparison is intended.
- [ ] Repeat baseline runs and verify reproducibility.
- [ ] Summarize line, circle, and polygon results using documented configurations and metrics.
- [ ] Report both tracking performance and constraint violations rather than relying on one scalar metric.

## 5. Robustness Experiments

### Model mismatch

- [ ] Perturb controller-side mass, inertia, and center-of-mass parameters while keeping the MuJoCo plant fixed.
- [ ] Sweep mismatch magnitude and direction.
- [ ] Measure tracking/path error, endpoint error where applicable, torque usage, saturation, and failure cases.
- [ ] Identify and document conditions where nominal OSC performance degrades significantly.

### Disturbances and sensor noise

- [ ] Add configurable external disturbances.
- [ ] Add configurable noise to measured position and velocity.
- [ ] Sweep disturbance/noise levels on the same baseline paths.
- [ ] Compare results under identical conditions.

### Singularity and path geometry

- [ ] Design paths that approach low-manipulability configurations.
- [ ] Track manipulability alongside errors and torque.
- [ ] Evaluate reference modification or path planning to avoid problematic configurations.
- [ ] Consider hybrid joint-space/task-space handling after baseline behavior is understood.

## 6. Controller Comparison and Learning-Based Extensions

- [ ] Implement a simple joint-space PD or computed-torque baseline.
- [ ] Compare it with OSC under matched initial conditions, paths, actuator limits, and simulation duration.
- [ ] Change one experimental factor at a time.
- [ ] Document assumptions and implementation complexity for each controller.
- [ ] Begin learning-based compensation only after a specific model-based limitation has been measured and documented.
- [ ] Prototype a bounded residual term rather than replacing the model-based controller immediately.
- [ ] Compare OSC and OSC-plus-learning under the same conditions, including generalization and constraint violations.

## 7. Maintenance

- [ ] Keep tests passing with `uv run --locked pytest`.
- [ ] Update the README when commands, metrics, or output formats change.
- [ ] Keep generated results out of source control unless intentionally selected as reference artifacts.
- [ ] Record important design decisions and metric definitions.
- [ ] Remove obsolete code or duplicated plotting logic only after a tested replacement exists.