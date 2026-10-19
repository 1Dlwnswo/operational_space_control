import mujoco
import numpy as np

xml = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.001"/>

  <worldbody>
    <body name="link1" pos="0 0 0">
      <!-- MuJoCo will calculate mass and inertia from geom properties and density -->
      <joint name="j1" type="hinge" axis="0 -1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" density="160"/> <!-- Density added -->
      <site name="link1_end" pos="0 0 -1"/> <!-- Site at end of link 1 -->

      <body name="link2" pos="0 0 -1">
        <!-- MuJoCo will calculate mass and inertia from geom properties and density -->
        <joint name="j2" type="hinge" axis="0 -1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" density="120"/> <!-- Density added -->
        <site name="end_effector" pos="0 0 -1"/> <!-- Site at end of link 2 (end-effector) -->
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="j1"/>
    <motor joint="j2"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

dt = model.opt.timestep
num_simulation_steps = 6000
time_log = [i * dt for i in range(num_simulation_steps)]

# Define link lengths based on XML geom tags (fromto = [x1 y1 z1 x2 y2 z2], so length is 1.0 for each)
L1 = 1.0
L2 = 1.0

# Extract physics parameters from the model
g = abs(model.opt.gravity[2]) # Magnitude of gravity (assuming gravity is along -Z)
m1 = model.body_mass[model.body('link1').id]
m2 = model.body_mass[model.body('link2').id]
c1 = 0.5 # Center of mass for link1 is at L1/2 relative to its body frame
c2 = 0.5 # Center of mass for link2 is at L2/2 relative to its body frame
I1 = model.body_inertia[model.body('link1').id][1] # Moment of inertia for link1 about its joint axis
I2 = model.body_inertia[model.body('link2').id][1] # Moment of inertia for link2 about its joint axis

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

def calcuate_inertia_matrix(q1, q2, m1, m2, I1, I2, c1, c2, L1, L2):
    M11 = I1 + I2 + m1*c1**2 + m2*(L1**2 + c2**2 + 2*L1*c2*np.cos(q2))
    M12 = I2 + m2*(c2**2 + L1*c2*np.cos(q2))
    M21 = M12
    M22 = I2 + m2*(c2**2)

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

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi # (-pi, pi]

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

# Define convergence time function for end-effector position
def get_ee_convergence_time(ee_x_log: np.ndarray, target_x: np.ndarray, dt, error_threshold=0.1):

    dist_errors = np.linalg.norm(np.array(ee_x_log) - target_x, axis=1)

    last_exceeding_idx = -1
    for i in range(len(dist_errors) - 1, -1, -1):
        if dist_errors[i] >= error_threshold:
            last_exceeding_idx = i
            break

    return (last_exceeding_idx + 1) * dt