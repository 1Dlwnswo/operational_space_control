# @title
import numpy as np
from dataclasses import dataclass, field

@dataclass
class LinePath:
    start_point: np.ndarray    # [x, z] numpy array
    end_point: np.ndarray      # [x, z] numpy array
    cruise_speed: float = 1.0  # Constant desired speed along the path
    stop_at_end: bool = True   # New parameter to control deceleration

    state: int = field(init=False)
    direction: np.ndarray = field(init=False)
    length: float = field(init=False)
    unit_direction: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.state = 0
        
        direction = self.end_point - self.start_point
        
        self.length = np.linalg.norm(direction)
        
        if self.length > 0:
            self.unit_direction = direction / self.length
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

        speed_along_path = np.dot(current_velocity, self.unit_direction)
        at_endpoint = dist <= 1e-2
        stopped = abs(speed_along_path) <= 1e-2
        
        if self.stop_at_end and at_endpoint and stopped:
            candidate_state = 3
        elif self.stop_at_end and dist <= brake_distance:
            candidate_state = 2
        else:
            if speed_along_path < self.cruise_speed - 1e-2:
                candidate_state = 0
            else:
                candidate_state = 1
        self.state = max(self.state, candidate_state)
        
        reference_point = projected_point
        reference_velocity = self.cruise_speed * self.unit_direction
        reference_acceleration = np.zeros(2)
        
        if self.state == 2:
            reference_point = self.end_point
            reference_velocity = np.zeros(2)
            reference_acceleration = - max_acceleration * self.unit_direction
        elif self.state == 3:
            reference_point = self.end_point
            reference_velocity = np.zeros(2)
            reference_acceleration = np.zeros(2)
            

            
        return reference_point, reference_velocity, reference_acceleration
    
    def calculate_path_errors(self, x_log: np.ndarray):

        x_log = np.asarray(x_log, dtype=float)
        
        if x_log.ndim != 2 or x_log.shape[1] != 2:
            raise ValueError("x_log must have shape (N, 2).")
        if x_log.shape[0] == 0:
            raise ValueError("x_log must contain at least one position.")

        if self.length == 0:
            closest_points = np.broadcast_to(self.start_point, x_log.shape)
        else:
            offsets = x_log - self.start_point
            projections = offsets @ self.unit_direction
            projections = np.clip(projections, 0.0, self.length)
            closest_points = self.start_point + projections[:, None] * self.unit_direction

        errors = np.linalg.norm(x_log - closest_points, axis=1)
        
        return errors
