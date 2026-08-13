# @title
import numpy as np
from dataclasses import dataclass

@dataclass
class LinePath:
    start_point: np.ndarray    # [x, z] numpy array
    end_point: np.ndarray      # [x, z] numpy array
    cruise_speed: float = 1.0  # Constant desired speed along the path
    stop_at_end: bool = True   # New parameter to control deceleration

    def __post_init__(self):
        self.direction = self.end_point - self.start_point
        self.length = np.linalg.norm(self.direction)
        if self.length > 0:
            self.unit_direction = self.direction / self.length
        else:
            self.unit_direction = np.array([0.0, 0.0])

    def compute_reference(self, current_position: np.ndarray, current_velocity: np.ndarray, max_acceleration, step_index: int = 0):

        vec_to_point = current_position - self.start_point

        if self.length > 0:
            projection_length = np.dot(vec_to_point, self.unit_direction)
        else:
            projection_length = 0.0

        projection_length_clamped = np.clip(projection_length, 0, self.length)
        projected_point = self.start_point + projection_length_clamped * self.unit_direction

        dist = self.length - projection_length_clamped
        brake_distance = max((self.cruise_speed**2)/(2*max_acceleration),0.05)

        # Adjust brake_distance if the segment is very short
        if self.length < brake_distance:
            brake_distance = 0.5 * self.length

        # If stop_at_end is True and close to the end, decelerate to stop. Otherwise, maintain velocity and acceleration
        if self.stop_at_end and dist <= brake_distance:
            reference_point = self.end_point
            reference_velocity = np.zeros(2)
            reference_acceleration = np.zeros(2)
        else:
            reference_point = projected_point
            reference_velocity = self.cruise_speed * self.unit_direction
            reference_acceleration = np.zeros(2)

        return reference_point, reference_velocity, reference_acceleration