import numpy as np


def calcuate_inertia_matrix(q1, q2, m1, m2, I1, I2, c1, c2, L1, L2):
    M11 = I1 + I2 + m1 * c1**2 + m2 * (L1**2 + c2**2 + 2 * L1 * c2 * np.cos(q2))
    M12 = I2 + m2 * (c2**2 + L1 * c2 * np.cos(q2))
    M21 = M12
    M22 = I2 + m2 * (c2**2)

    return np.array([[M11, M12], [M21, M22]])


def calculate_gravity_torques(q1, q2, m1, m2, c1, c2, L1, g):
    # Torque for joint 1
    tau_g1 = m1 * g * c1 * np.sin(q1) + m2 * g * (L1 * np.sin(q1) + c2 * np.sin(q1 + q2))

    # Torque for joint 2
    tau_g2 = m2 * g * c2 * np.sin(q1 + q2)

    return np.array([tau_g1, tau_g2])


def calculate_coriolis_centrifugal_torques(q1, q2, q1_dot, q2_dot, m2, c2, L1):
    # These terms come from the Christoffel symbols / C(q, q_dot) * q_dot
    h = m2 * L1 * c2 * np.sin(q2)

    tau_cc1 = -h * q2_dot * q1_dot - h * (q1_dot + q2_dot) * q2_dot
    tau_cc2 = h * q1_dot * q1_dot

    return np.array([tau_cc1, tau_cc2])
