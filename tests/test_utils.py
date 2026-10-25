import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.parameters import (
    ImpedanceControlParameters,
    RobotParameters,
    SimulationParameters,
    SimulationResult,
)
from src.utils import (
    create_experiment_dir,
    read_config,
    read_log,
    read_metrics,
    save_config,
    save_log,
    save_metrics,
    wrap_to_pi,
)


def test_wrap_to_pi_returns_angles_in_expected_interval():
    wrapped = wrap_to_pi(np.array([-3 * np.pi, -np.pi / 2, 3 * np.pi]))

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    np.testing.assert_allclose(wrapped, [-np.pi, -np.pi / 2, -np.pi])


def test_create_experiment_dir_creates_timestamped_directory(tmp_path, monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now():
            class FixedNow:
                @staticmethod
                def strftime(_format):
                    return "2026-10-22_120000"

            return FixedNow()

    monkeypatch.setattr("src.utils.datetime", FixedDateTime)
    monkeypatch.setattr("src.utils.Path", lambda _path: tmp_path)

    experiment_dir = create_experiment_dir("line")

    assert (
        experiment_dir
        == tmp_path / "experiment" / "line" / "results" / "2026-10-22_120000"
    )
    assert experiment_dir.is_dir()


def test_create_experiment_dir_raises_if_timestamped_directory_exists(
    tmp_path, monkeypatch
):
    class FixedDateTime:
        @staticmethod
        def now():
            class FixedNow:
                @staticmethod
                def strftime(_format):
                    return "2026-10-22_120000"

            return FixedNow()

    monkeypatch.setattr("src.utils.datetime", FixedDateTime)
    monkeypatch.setattr("src.utils.Path", lambda _path: tmp_path)
    create_experiment_dir("line")

    with pytest.raises(FileExistsError):
        create_experiment_dir("line")


def test_save_config_writes_parameters_and_compact_matrices(tmp_path):
    sim_params = SimulationParameters(dt=0.001, num_steps=2600)
    control_params = ImpedanceControlParameters(max_acceleration=10.0)
    robot_params = RobotParameters(
        m1=1.0,
        m2=0.8,
        I1=0.1,
        I2=0.08,
        L1=1.0,
        L2=0.8,
        c1=0.5,
        c2=0.4,
        g=9.81,
    )

    config_path = save_config(tmp_path, sim_params, control_params, robot_params)
    contents = config_path.read_text(encoding="utf-8")
    config = json.loads(contents)

    assert config["simulation_parameters"] == {"dt": 0.001, "num_steps": 2600}
    assert config["robot_parameters"] == {
        "m1": 1.0,
        "m2": 0.8,
        "I1": 0.1,
        "I2": 0.08,
        "L1": 1.0,
        "L2": 0.8,
        "c1": 0.5,
        "c2": 0.4,
        "g": 9.81,
    }
    assert config["impedance_control_parameters"]["Kp"] == [[25.0, 0.0], [0.0, 25.0]]
    assert config["impedance_control_parameters"]["Kd"] == [[10.0, 0.0], [0.0, 10.0]]
    assert '"Kp": [[25.0, 0.0], [0.0, 25.0]]' in contents
    assert '"Kd": [[10.0, 0.0], [0.0, 10.0]]' in contents


def test_save_log_and_read_log_round_trip(tmp_path):
    sim_params = SimulationParameters(dt=0.01, num_steps=2)
    simulation_log = [
        SimulationResult(
            q=np.array([1.0, 2.0]),
            qdot=np.array([3.0, 4.0]),
            tau=np.array([5.0, 6.0]),
            a_cmd=np.array([7.0, 8.0]),
        )
    ]
    log_path = tmp_path / "raw_log.csv"

    save_log(log_path, simulation_log, sim_params)
    rows = read_log(log_path)

    assert rows == [
        {
            "time": 0.0,
            "q_1": 1.0,
            "q_2": 2.0,
            "qdot_1": 3.0,
            "qdot_2": 4.0,
            "tau_1": 5.0,
            "tau_2": 6.0,
            "a_cmd_1": 7.0,
            "a_cmd_2": 8.0,
        }
    ]


def test_save_config_and_read_config_round_trip(tmp_path):
    sim_params = SimulationParameters(dt=0.001, num_steps=2600)
    control_params = ImpedanceControlParameters(max_acceleration=10.0)
    robot_params = RobotParameters(
        m1=1.0,
        m2=0.8,
        I1=0.1,
        I2=0.08,
        L1=1.0,
        L2=0.8,
        c1=0.5,
        c2=0.4,
        g=9.81,
    )

    config_path = save_config(tmp_path, sim_params, control_params, robot_params)

    assert read_config(config_path) == {
        "simulation_parameters": {"dt": 0.001, "num_steps": 2600},
        "robot_parameters": {
            "m1": 1.0,
            "m2": 0.8,
            "I1": 0.1,
            "I2": 0.08,
            "L1": 1.0,
            "L2": 0.8,
            "c1": 0.5,
            "c2": 0.4,
            "g": 9.81,
        },
        "impedance_control_parameters": {
            "max_acceleration": 10.0,
            "Kp": [[25.0, 0.0], [0.0, 25.0]],
            "Kd": [[10.0, 0.0], [0.0, 10.0]],
            "max_torque": 100,
        },
    }


def test_save_metrics_and_read_metrics_round_trip(tmp_path):
    sim_params = SimulationParameters(dt=0.01, num_steps=2)
    simulation_metric = SimpleNamespace(
        simulation_logs=["excluded log"],
        path_obj="excluded path",
        experiment_dir="excluded directory",
        sim_params="excluded simulation parameters",
        robot_params="excluded robot parameters",
        control_params="excluded control parameters",
        errors=np.array([0.1, 0.2]),
        distance_errors=np.array([0.3, 0.4]),
        manipulability=np.array([0.5, 0.6]),
        rmse=0.15,
        convergence_time=0.02,
        saturation_torque_ratio=0.25,
    )
    metrics_path = tmp_path / "metrics.csv"

    save_metrics(metrics_path, simulation_metric, sim_params)
    save_metrics(metrics_path, simulation_metric, sim_params)

    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        csv_rows = list(csv.reader(f))
    assert csv_rows[0][0] == "rmse"
    assert not any(
        row
        and row[0]
        in {
            "simulation_logs",
            "path_obj",
            "experiment_dir",
            "sim_params",
            "robot_params",
            "control_params",
        }
        for row in csv_rows
    )
    assert csv_rows[:6] == [
        ["rmse", "0.15", "", ""],
        ["convergence_time", "0.02", "", ""],
        ["saturation_torque_ratio", "0.25", "", ""],
        ["time", "errors", "distance_errors", "manipulability"],
        ["0.0", "0.1", "0.3", "0.5"],
        ["0.01", "0.2", "0.4", "0.6"],
    ]

    assert read_metrics(metrics_path) == [
        {
            "rmse": 0.15,
            "convergence_time": 0.02,
            "saturation_torque_ratio": 0.25,
            "time": [0.0, 0.01],
            "errors": [0.1, 0.2],
            "distance_errors": [0.3, 0.4],
            "manipulability": [0.5, 0.6],
        },
        {
            "rmse": 0.15,
            "convergence_time": 0.02,
            "saturation_torque_ratio": 0.25,
            "time": [0.0, 0.01],
            "errors": [0.1, 0.2],
            "distance_errors": [0.3, 0.4],
            "manipulability": [0.5, 0.6],
        },
    ]
