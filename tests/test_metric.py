import numpy as np
import pytest

from src.metrics.metrics import (
    SimulationMetrics,
    get_ee_convergence_time,
)
from src.parameters import (
    ImpedanceControlParameters,
    RobotParameters,
    SimulationParameters,
    SimulationResult,
)
from src.path.LinePath import LinePath
from src.utils import save_config


def test_convergence_time_is_after_last_sample_outside_threshold():
    positions = np.array([[0.0, 0.0], [0.5, 0.0], [0.95, 0.0], [1.02, 0.0], [1.0, 0.0]])

    convergence_time = get_ee_convergence_time(positions, target_x=np.array([1.0, 0.0]), dt=0.01, error_threshold=0.1)

    assert convergence_time == pytest.approx(0.02)


def test_convergence_time_is_zero_when_all_samples_are_within_threshold():
    positions = np.array([[1.0, 0.0], [1.005, 0.0]])

    assert get_ee_convergence_time(positions, np.array([1.0, 0.0]), 0.01) == 0.0


def test_metrics_computes_path_error_and_manipulability(tmp_path):
    line = LinePath(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    robot = RobotParameters(
        m1=1.0,
        m2=1.0,
        I1=1.0,
        I2=1.0,
        L1=1.0,
        L2=1.0,
        c1=0.5,
        c2=0.5,
        g=9.81,
    )
    logs = [
        SimulationResult(
            q=np.array([0.0, np.pi / 2]),
            qdot=np.zeros(2),
            tau=np.zeros(2),
            a_cmd=np.zeros(2),
            x=np.array([0.5, 0.1]),
        ),
        SimulationResult(
            q=np.array([0.0, np.pi / 2]),
            qdot=np.zeros(2),
            tau=np.zeros(2),
            a_cmd=np.zeros(2),
            x=np.array([0.5, -0.1]),
        ),
    ]

    sim_params = SimulationParameters(dt=0.01, num_steps=2)
    save_config(
        tmp_path,
        sim_params,
        ImpedanceControlParameters(max_acceleration=10.0, max_torque=100.0),
        robot,
    )

    result = SimulationMetrics(
        simulation_logs=logs,
        path_obj=line,
        experiment_dir=tmp_path,
    )

    assert result.rmse == pytest.approx(0.1)
    np.testing.assert_allclose(result.errors, [0.1, 0.1])
    assert result.distance_errors.shape == (2,)
    assert result.convergence_time == pytest.approx(0.02)
    assert result.manipulability.shape == (2,)
    assert result.saturation_torque_ratio == pytest.approx(0.0)


def test_metrics_rejects_empty_simulation_logs(tmp_path):
    with pytest.raises(ValueError, match="at least one result"):
        SimulationMetrics(
            simulation_logs=[],
            path_obj=LinePath(np.zeros(2), np.ones(2)),
            experiment_dir=tmp_path,
        )
