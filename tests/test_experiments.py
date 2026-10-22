"""Short end-to-end checks for each experiment's path configuration."""

import numpy as np

from experiment.CirclePath import simulation as circle_experiment
from experiment.LinePath import simulation as line_experiment
from experiment.PolygonPath import simulation as polygon_experiment
from simulation.run_simulation import run_simulation as run_mujoco_simulation
from src.path.CirclePath import CirclePath
from src.path.PolygonPath import PolygonPath
from src.robot.arm_2dof.kinematics import (
    get_end_effector_position,
    inverse_kinematics,
)


SHORT_RUN_STEPS = 12


def _shorten_simulation(monkeypatch, experiment_module):
    """Keep each experiment's real setup while capping its MuJoCo steps."""

    def run_short(model, data, q_init, control_type, sim_params, path_obj, **kwargs):
        sim_params.num_steps = SHORT_RUN_STEPS
        return run_mujoco_simulation(
            model, data, q_init, control_type, sim_params, path_obj, **kwargs
        )

    monkeypatch.setattr(experiment_module, "run_simulation", run_short)


def _assert_operational_space_log(log, expected_steps):
    assert len(log) == expected_steps
    for result in log:
        assert result.save_os is True
        assert result.q.shape == result.qdot.shape == result.tau.shape == (2,)
        assert result.a_cmd.shape == (2, 1)
        for value in (result.q, result.qdot, result.tau, result.a_cmd, result.x):
            assert np.all(np.isfinite(value))


def test_circle_path_experiment_runs(monkeypatch):
    _shorten_simulation(monkeypatch, circle_experiment)
    q_init = np.array([np.pi * 5 / 12, np.pi * (1 - 10 / 12)])
    path = CirclePath(center=np.array([0.3, 0.0]), radius=0.7)

    log, sim_params = circle_experiment.run_experiment(
        q_init, path, max_acceleration=15.0, use_viewer=False
    )

    _assert_operational_space_log(log, SHORT_RUN_STEPS)
    assert sim_params.dt > 0


def test_line_path_experiment_runs_for_one_target(monkeypatch):
    _shorten_simulation(monkeypatch, line_experiment)
    q_init = np.array([np.pi * 5 / 12, np.pi * (1 - 10 / 12)])
    start = get_end_effector_position(q_init[0], q_init[1], 1.0, 1.0)
    target_positions = np.array([start + np.array([0.05, 0.05])])

    logs, simulation_metrics, sim_params = line_experiment.run_experiment(
        q_init,
        target_positions,
        cruise_speed=0.5,
        max_acceleration=5.0,
        use_viewer=False,
    )

    assert len(logs) == 1
    assert len(simulation_metrics) == 1
    assert np.isfinite(simulation_metrics[0].rmse)
    _assert_operational_space_log(logs[0], SHORT_RUN_STEPS)
    assert sim_params.dt > 0


def test_polygon_path_experiment_runs(monkeypatch):
    _shorten_simulation(monkeypatch, polygon_experiment)
    vertices = polygon_experiment.regular_polygon(1.0, 5)
    q_init = inverse_kinematics(vertices[0], 1.0, 1.0, vertices[0])
    path = PolygonPath(vertices, desired_speed=0.5, stop_at_end=False)

    log, sim_params = polygon_experiment.run_experiment(
        q_init, path, max_acceleration=5.0, use_viewer=False
    )

    _assert_operational_space_log(log, SHORT_RUN_STEPS)
    assert sim_params.dt > 0
