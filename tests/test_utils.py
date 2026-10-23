import json
import numpy as np
import pytest

from src.parameters import ImpedanceControlParameters, SimulationParameters
from src.utils import create_experiment_dir, save_config, wrap_to_pi


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

    assert experiment_dir == tmp_path / "experiment" / "line" / "results" / "2026-10-22_120000"
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

    config_path = save_config(tmp_path, sim_params, control_params)
    contents = config_path.read_text(encoding="utf-8")
    config = json.loads(contents)

    assert config["simulation_parameters"] == {"dt": 0.001, "num_steps": 2600}
    assert config["impedance_control_parameters"]["Kp"] == [[25.0, 0.0], [0.0, 25.0]]
    assert config["impedance_control_parameters"]["Kd"] == [[10.0, 0.0], [0.0, 10.0]]
    assert '"Kp": [[25.0, 0.0], [0.0, 25.0]]' in contents
    assert '"Kd": [[10.0, 0.0], [0.0, 10.0]]' in contents
