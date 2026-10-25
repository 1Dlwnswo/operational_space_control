# TODO — Operational Space Control

This checklist tracks the current implementation work and the longer-term research direction.
Items are ordered by priority; complete and validate the current metrics/visualization work before expanding into robustness experiments.

## 1. Current Branch: Metrics and Result Visualization

### 1.1 Expand metrics
- [ ] Review existing metric definitions and document their units and interpretation.
- [ ] Keep geometric path-following error distinct from time-indexed trajectory-tracking error.
- [ ] Add Cartesian position-tracking RMSE against the desired reference at each simulation timestep.
- [ ] Add maximum position/path error.
- [ ] Add final endpoint error.
- [ ] Add velocity-tracking error when desired velocity is available.
- [ ] Record peak and mean absolute joint torque.
- [ ] Record the fraction of simulation steps where actuator torque limits are active.
- [ ] Make convergence-time definitions explicit for each path type; avoid reporting zero merely because the initial state satisfies a geometric error threshold.
- [ ] Check metric behavior on edge cases: short runs, empty/invalid logs, zero-length paths, and thresholds never reached.
- [ ] Add or update unit tests for each metric, including expected units and boundary conditions.

### 1.2 Polygon path metric plots
- [ ] Implement plotting for metric results produced by PolygonPath experiments.
- [ ] Reuse the existing plotting conventions used by LinePath and CirclePath where practical.
- [ ] Plot the polygon path and end-effector trajectory together, with a clear legend and equal axis scaling.
- [ ] Plot available scalar metrics (for example, tracking/path error over time and joint torque over time).
- [ ] Make the plotter work with the experiment's saved result format and timestamped output directories.
- [ ] Verify that plots are generated correctly for one polygon and for paths with multiple segments/vertices.
- [ ] Add tests for result loading and plotting-data preparation; keep rendering-dependent tests optional/headless-safe.
- [ ] Update experiment documentation and README examples after the plotting workflow is stable.

## 2. Validate Existing Controller and Simulation

### 2.1 Torque limiting and actuator constraints
- [ ] Review and fix the torque-limiting function so its behavior matches the intended policy.
- [ ] Add tests for one joint exceeding the limit, both joints exceeding the limit, zero torque, and negative torque.
- [ ] Align the controller's configured torque limit with the MuJoCo actuator `ctrlrange`/force constraints.
- [ ] Log the requested torque separately from the control input actually applied by MuJoCo.
- [ ] Record when and how often actuator saturation occurs.

### 2.2 Dynamics consistency
- [ ] Verify that controller dynamics parameters match the MuJoCo model: link mass, center of mass, inertia, link geometry, joint axes, and gravity.
- [ ] Compare the manually computed inertia matrix with MuJoCo's mass matrix at selected states.
- [ ] Compare gravity and velocity-dependent bias terms against MuJoCo at matched states.
- [ ] Check sign conventions and coordinate-frame assumptions in forward kinematics, Jacobian, and J-dot.
- [ ] Add numerical tests across multiple non-singular joint configurations.

### 2.3 OSC numerical behavior
- [ ] Review the operational-space inertia calculation and its regularization near singular configurations.
- [ ] Test sensitivity to the regularization parameter.
- [ ] Record manipulability, torque magnitude, and tracking error as the arm approaches low-manipulability configurations.
- [ ] Distinguish numerical regularization from a complete singularity-handling strategy.
- [ ] Document the current controller's assumptions and known limitations.

## 3. Establish Reproducible Baselines

- [ ] Fix and document baseline initial joint positions and velocities.
- [ ] Fix path parameters, desired speed, acceleration limits, and simulation duration for each baseline.
- [ ] Record timestep, controller gains, model parameters, and actuator constraints with every experiment.
- [ ] Ensure metric definitions are consistent across runs and path types where comparison is intended.
- [ ] Run repeated baseline experiments and confirm results are reproducible.
- [ ] Save configuration and metric outputs alongside plots.
- [ ] Summarize baseline results for line, circle, and polygon paths.

## 4. Robustness Experiments

### 4.1 Model mismatch
- [ ] Introduce controlled errors in mass, inertia, and center-of-mass parameters used by the controller while keeping the MuJoCo plant fixed.
- [ ] Sweep mismatch magnitude and direction.
- [ ] Measure tracking error, endpoint error, torque usage, saturation frequency, and failure cases.
- [ ] Identify conditions under which nominal OSC performance degrades significantly.

### 4.2 Disturbances and sensor noise
- [ ] Add controlled external disturbances.
- [ ] Add configurable sensor noise to measured position and velocity.
- [ ] Sweep disturbance/noise levels using the same baseline trajectories.
- [ ] Compare performance and failure behavior under identical conditions.

### 4.3 Singularity and path geometry
- [ ] Design test paths that approach low-manipulability configurations.
- [ ] Track manipulability alongside tracking error and torque.
- [ ] Test whether path planning or reference modification can avoid problematic configurations.
- [ ] Evaluate hybrid joint-space/task-space handling only after baseline behavior is understood.

## 5. Controller Comparisons

- [ ] Establish a simple joint-space PD or computed-torque baseline.
- [ ] Compare the baseline with OSC under matched initial conditions, paths, actuator limits, and simulation duration.
- [ ] Change one experimental factor at a time.
- [ ] Report both tracking performance and constraint violations; do not compare controllers using a single metric alone.
- [ ] Document controller assumptions and computational/implementation complexity.

## 6. Future Learning-Based Compensation

- [ ] Start only after the model-based baseline and its failure conditions are clearly documented.
- [ ] Select one measured limitation (for example, residual error under model mismatch) as the learning target.
- [ ] Prototype a bounded residual compensation term rather than replacing the model-based controller immediately.
- [ ] Compare OSC and OSC-plus-learning using the same test conditions.
- [ ] Evaluate sample efficiency, generalization, safety/constraint violations, and performance outside training conditions.
- [ ] Keep the model-based controller as a transparent baseline.

## 7. Project Maintenance

- [ ] Keep tests passing with `uv run pytest`.
- [ ] Update README instructions when CLI options, metrics, or output formats change.
- [ ] Keep generated results out of source control unless they are intentionally selected as reference artifacts.
- [ ] Record important design decisions and metric definitions.
- [ ] Remove obsolete code and duplicated plotting logic when the replacement is tested.