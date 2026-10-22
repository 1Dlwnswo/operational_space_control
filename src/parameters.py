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
    
"""
fucking 개버그를 발견했어요. 현재 
experiment <= run_simulation <= perform_simulation_step
으로 시뮬레이션 로그를 넘기는데 이때, perform_simulation_step에서 SimulationResults로 데이터를 넘겨요.
근데 fucking run_simulation에서 이걸 리스트에 쳐 담에세 experiment로 보내서 여기서 또 염병을 하는 상황

아 아니네
ㅈㅅ
"""