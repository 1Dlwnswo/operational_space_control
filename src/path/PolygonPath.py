from dataclasses import dataclass, field
import numpy as np
from src.path.LinePath import LinePath

@dataclass
class PolygonPath:
    vertices: np.ndarray                    # (num_vertices, 2) array of [x, z] vertex positions
    desired_speed: float = 1.0              # Constant desired speed along the Path
    current_segment_idx: int = field(init=False, default=0)    # Internal state to track the current segment being traversed
    _segments: list = field(init=False)     # List of LinePath objects
    stop_at_vertices: bool = False          # Parameter to control deceleration at each verticies
    stop_at_end: bool = False               # Parameter to control deceleration at the absolute end of the path

    def __post_init__(self):
        if len(self.vertices) < 2:
            raise ValueError("PolygonPath requires at least two vertices to form segments.")
        self._segments = []

        if self.stop_at_end: # Non-looping path, stops at the last vertex
            # Create segments from vertex[0] to vertex[1], ..., vertex[N-2] to vertex[N-1]
            for i in range(len(self.vertices) - 1):
                start = self.vertices[i]
                end = self.vertices[i + 1]
                # For the very last segment of the overall path, ensure it stops.
                # For intermediate segments, use stop_at_vertices.
                segment_stop_at_end_param = self.stop_at_vertices # Default for intermediate
                if i == len(self.vertices) - 2: # This is the final segment of the entire PolygonPath
                    segment_stop_at_end_param = True # Ensure robot stops at the last vertex

                self._segments.append(LinePath(start_point=start, end_point=end, cruise_speed=self.desired_speed, stop_at_end=segment_stop_at_end_param))
        else: # Closed looping path
            # Create segments from vertex[0] to vertex[1], ..., vertex[N-1] to vertex[0]
            for i in range(len(self.vertices)):
                start = self.vertices[i]
                end = self.vertices[(i + 1) % len(self.vertices)] # Loop back to the first vertex
                self._segments.append(LinePath(start_point=start, end_point=end, cruise_speed=self.desired_speed, stop_at_end=self.stop_at_vertices)) # Use stop_at_vertices for all segments in a loop

        for idx, segment in enumerate(self._segments):
            print(f"Segment {idx}: Start {segment.start_point}, End {segment.end_point}, Stop at end: {segment.stop_at_end}")

    def compute_reference(self, current_position: np.ndarray, current_velocity: np.ndarray, max_acceleration, step_index: int = 0):
        
        if self.current_segment_idx >= len(self._segments):
            if self.stop_at_end: # If at the end of a non-looping path, maintain the last position and zero velocity/acceleration
                last_segment_end_point = self._segments[-1].end_point
                return last_segment_end_point, np.zeros(2), np.zeros(2)
            for segment in self._segments:
                segment.state = 0
            self.current_segment_idx = 0

        current_segment_path = self._segments[self.current_segment_idx]

        # Get reference from the current LinePath segment
        x_des, x_des_dot, x_des_ddot = current_segment_path.compute_reference(
            current_position, current_velocity, max_acceleration, step_index
        )
        
        if current_segment_path.stop_at_end:
            if current_segment_path.state == 3:
                print(f"Segment {self.current_segment_idx} completed, moving to the next segment. stop_at_end: {current_segment_path.stop_at_end}")
                self.current_segment_idx += 1
        elif current_segment_path.at_endpoint:
            print(f"Segment {self.current_segment_idx} completed, moving to the next segment. stop_at_end: {current_segment_path.stop_at_end}")
            self.current_segment_idx += 1                

        return x_des, x_des_dot, x_des_ddot

    def calculate_path_errors(self, x_log: np.ndarray) -> np.ndarray:
        x_log = np.asarray(x_log, dtype=float)
        if x_log.ndim != 2 or x_log.shape[1] != 2:
            raise ValueError("x_log must have shape (N, 2).")
        if x_log.shape[0] == 0:
            raise ValueError("x_log must contain at least one position.")

        segment_errors = np.vstack([
            segment.calculate_path_errors(x_log)
            for segment in self._segments
        ])
        return np.min(segment_errors, axis=0)
