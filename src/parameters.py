from dataclasses import dataclass
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
    Kp: np.ndarray
    Kd: np.ndarray
    max_torque: float


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
    