from dataclasses import dataclass
import numpy as np

@dataclass
class RobotParameters:
    L1: float
    L2: float
    m1: float
    m2: float
    c1: float
    c2: float
    g: float
    I1: float
    I2: float


@dataclass
class SimulationParameters:
    dt: float
    num_steps: int


@dataclass
class ImpedanceControlParameters:
    Kp: np.ndarray
    Kd: np.ndarray
    max_torque: float


@dataclass
class SimulationResult:
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray

    tau: np.ndarray
    a_cmd: np.ndarray

    save_os: bool = False
    x: np.ndarray | None = None
    xdot: np.ndarray | None = None
    xddot: np.ndarray | None = None