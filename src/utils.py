import numpy as np

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi # (-pi, pi]

# Define convergence time function for end-effector position
def get_ee_convergence_time(ee_x_log: np.ndarray, target_x: np.ndarray, dt, error_threshold=0.1):

    dist_errors = np.linalg.norm(np.array(ee_x_log) - target_x, axis=1)

    last_exceeding_idx = -1
    for i in range(len(dist_errors) - 1, -1, -1):
        if dist_errors[i] >= error_threshold:
            last_exceeding_idx = i
            break

    return (last_exceeding_idx + 1) * dt