import numpy as np
from src.utils import wrap_to_pi

def get_end_effector_position(q1, q2, L1, L2):
    x = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    z = - L1 * np.cos(q1) - L2 * np.cos(q1 + q2)
    return np.array([x, z])

def get_jacobian(q1, q2, L1, L2):
    J = np.array([
        [L1 * np.cos(q1) + L2 * np.cos(q1 + q2), L2 * np.cos(q1 + q2)],
        [L1 * np.sin(q1) + L2 * np.sin(q1 + q2), L2 * np.sin(q1 + q2)]
    ])
    return J

def get_J_dot(q1, q2, q1_dot, q2_dot, L1, L2):
    Jdot = np.array([
        [-L1 * q1_dot * np.sin(q1) - L2 * (q1_dot + q2_dot) * np.sin(q1 + q2), -L2 * (q1_dot + q2_dot) * np.sin(q1 + q2)],
        [L1 * q1_dot * np.cos(q1) + L2 * (q1_dot + q2_dot) * np.cos(q1 + q2), L2 * (q1_dot + q2_dot) * np.cos(q1 + q2)]
    ])
    return Jdot

def inverse_kinematics(pos, L1, L2, q_current = None):
    x = pos[0]
    z = pos[1]

    r = np.sqrt(x**2 + z**2)
    theta = np.arctan2(x, -z)

    cos_q2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    # solution 1
    q2_1 = wrap_to_pi(np.arccos(cos_q2))
    beta_1 = np.arctan2(L2 * np.sin(q2_1), L1 + L2 * np.cos(q2_1))
    q1_1 = wrap_to_pi(theta - beta_1)

    # solution 2
    q2_2 = wrap_to_pi(-np.arccos(cos_q2))
    beta_2 = np.arctan2(L2 * np.sin(q2_2), L1 + L2 * np.cos(q2_2))
    q1_2 = wrap_to_pi(theta - beta_2)

    q1 = np.array([q1_1, q2_1])
    q2 = np.array([q1_2, q2_2])

    if q_current is None: return q1 if abs(q1_1) < abs(q1_2) else q2

    d1 = np.linalg.norm(wrap_to_pi(q1 - q_current))
    d2 = np.linalg.norm(wrap_to_pi(q2 - q_current))

    return q1 if d1 < d2 else q2
