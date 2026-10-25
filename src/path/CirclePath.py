# @title
from dataclasses import dataclass
import numpy as np

@dataclass
class CirclePath:
    center: np.ndarray        # [x, z] numpy array
    radius: float
    cruise_angular_speed: float = 1.0 # Radians per second
    start_angle: float = 0.0  # Angle from positive x-axis for start of circle (in radians)
    direction: int = 1        # 1 for counter-clockwise, -1 for clockwise

    def compute_reference(self, current_position: np.ndarray, current_velocity, max_aceleration, step_index: int = 0):
        # Vector from center to current_position
        vec_from_center = current_position - self.center
        current_radius = np.linalg.norm(vec_from_center)

        unit_vec_from_center = np.array([np.cos(self.start_angle), np.sin(self.start_angle)])

        if current_radius < 1e-6: # Avoid division by zero
            reference_point = self.center + self.radius * np.array([np.cos(self.start_angle), np.sin(self.start_angle)])

            distance = self.radius
            tangent_vector = np.array([-np.sin(self.start_angle), np.cos(self.start_angle)]) * self.direction

            reference_velocity = np.array([0.0, 0.0])
        else:
            unit_vec_from_center = vec_from_center / current_radius
            reference_point = self.center + self.radius * unit_vec_from_center

            distance = np.abs(current_radius - self.radius)
            tangent_vector = np.array([-unit_vec_from_center[1], unit_vec_from_center[0]]) * self.direction
            reference_velocity = tangent_vector * self.radius * self.cruise_angular_speed

        reference_acceleration = - self.cruise_angular_speed**2 * self.radius * unit_vec_from_center

        return reference_point, reference_velocity, reference_acceleration
    
    def calculate_path_errors(self, x_log: np.ndarray):

        x_log = np.asarray(x_log, dtype=float)
        
        if x_log.ndim != 2 or x_log.shape[1] != 2:
            raise ValueError("x_log must have shape (N, 2).")
        if x_log.shape[0] == 0:
            raise ValueError("x_log must contain at least one position.")

        errors = (
                np.linalg.norm(x_log - self.center, axis=1) - self.radius
            )
        
        return errors