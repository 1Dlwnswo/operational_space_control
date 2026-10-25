from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RobotParameters:
    m1: float
    m2: float
    I1: float
    I2: float
    L1: float
    L2: float
    c1: float
    c2: float
    g: float


@dataclass
class SimulationParameters:
    dt: float
    num_steps: int


@dataclass
class ImpedanceControlParameters:
    max_acceleration: float

    Kp: np.ndarray = field(init=False)
    Kd: np.ndarray = field(init=False)

    max_torque: float = 100

    def __post_init__(self):
        # Operational Space Parameters
        self.Kp = (0.25 * self.max_acceleration**2) * np.eye(2)
        self.Kd = self.max_acceleration * np.eye(2)


@dataclass
class SimulationResult:
    # all ndarray need to be (2,) vector
    q: np.ndarray
    qdot: np.ndarray

    tau: np.ndarray
    a_cmd: np.ndarray

    save_os: Optional[bool] = None
    qddot: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    xdot: Optional[np.ndarray] = None
    xddot: Optional[np.ndarray] = None
