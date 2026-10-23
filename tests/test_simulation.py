"""Short integration checks for the MuJoCo simulation pipeline."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from simulation.perform_simulation_step import perform_simulation_step
from simulation.run_simulation import run_simulation
from src.mujoco_setup import get_robot_parameters, load_model_and_data
from src.parameters import ImpedanceControlParameters, SimulationParameters
from src.path.CirclePath import CirclePath


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "arm_2dof.xml"


def test_short_mujoco_run_advances_and_records_finite_state():
    model, data = load_model_and_data(str(MODEL_PATH))
    steps = 8
    sim_params = SimulationParameters(dt=model.opt.timestep, num_steps=steps)
    path = CirclePath(center=np.array([0.3, 0.0]), radius=0.5)

    log = run_simulation(
        model,
        data,
        q_init=np.array([0.4, 0.5]),
        control_type="pw",
        sim_params=sim_params,
        path_obj=path,
        max_acceleration=5.0,
        save_operational_space=True,
        use_viewer=False,
    )

    assert len(log) == steps
    assert data.time == pytest.approx(steps * sim_params.dt)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))

    for result in log:
        assert result.save_os is True
        for value in (
            result.q,
            result.qdot,
            result.tau,
            result.a_cmd,
            result.qddot,
            result.x,
            result.xdot,
            result.xddot,
        ):
            assert np.all(np.isfinite(value))


def test_single_simulation_step_applies_torque_and_advances_time():
    model, data = load_model_and_data(str(MODEL_PATH))
    data.qpos[:] = [0.4, 0.5]
    mujoco.mj_forward(model, data)
    path = CirclePath(center=np.array([0.3, 0.0]), radius=0.5)
    robot_params = get_robot_parameters(model)
    control_params = ImpedanceControlParameters(
        max_acceleration=4.0, max_torque=100.0
    )

    result = perform_simulation_step(
        model,
        data,
        robot_params,
        control_params,
        control_type="pw",
        path_obj=path,
        max_acceleration=5.0,
        save_operational_space=True,
    )

    assert data.time == pytest.approx(model.opt.timestep)
    assert np.any(data.ctrl != 0.0)
    assert result.tau.shape == (2,)
    assert np.all(np.isfinite(result.tau))


def test_simulation_step_rejects_unknown_control_type():
    model, data = load_model_and_data(str(MODEL_PATH))
    robot_params = get_robot_parameters(model)
    control_params = ImpedanceControlParameters(
        max_acceleration=2.0, max_torque=100.0
    )
    path = CirclePath(center=np.zeros(2), radius=0.5)

    with pytest.raises(ValueError, match="Unknown control_type"):
        perform_simulation_step(
            model,
            data,
            robot_params,
            control_params,
            control_type="invalid",
            path_obj=path,
            max_acceleration=5.0,
        )
