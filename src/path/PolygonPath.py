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


    def compute_reference(self, current_position: np.ndarray, current_velocity: np.ndarray, max_acceleration, step_index: int = 0):
        # Handle the case where the path has completed (only applicable for non-looping paths)
        if self.stop_at_end and self.current_segment_idx >= len(self._segments):
            # If at the end of a non-looping path, maintain the last position and zero velocity/acceleration
            last_segment_end_point = self._segments[-1].end_point
            return last_segment_end_point, np.zeros(2), np.zeros(2)

        # Ensure current_segment_idx loops around if it exceeds the number of segments for looping paths
        if not self.stop_at_end: # Only wrap around for looping paths
            self.current_segment_idx = self.current_segment_idx % len(self._segments)

        current_segment_path = self._segments[self.current_segment_idx]

        # Get reference from the current LinePath segment
        x_des, x_des_dot, x_des_ddot = current_segment_path.compute_reference(
            current_position, current_velocity, max_acceleration, step_index
        )

        # Check for segment completion
        distance_to_segment_end = np.linalg.norm(current_position - current_segment_path.end_point)
        brake_distance = max((self.desired_speed**2)/(2*max_acceleration),0.05)

        # If the current segment's LinePath object has stop_at_end = True, it will decelerate.
        # We only advance to the next segment if we are close to the end AND (it's a looping path OR we have stopped).
        # If the LinePath's stop_at_end is False, we just advance based on distance.
        if distance_to_segment_end < brake_distance: # distance threshold
            if current_segment_path.stop_at_end == True: # This segment is set to stop
                if np.linalg.norm(current_velocity) < 0.01: # velocity threshold
                    self.current_segment_idx += 1
            else: # This segment does not stop, just transition when close
                self.current_segment_idx += 1

            # If there are more segments (or it's a looping path), compute reference from the new current segment
            if self.current_segment_idx < len(self._segments): # More segments remaining in a non-looping path, or any segment in a looping path
                next_segment_path = self._segments[self.current_segment_idx]
                # Re-calculate reference with the new segment's logic for the current simulation step
                return next_segment_path.compute_reference(current_position, current_velocity, max_acceleration, step_index)
            elif not self.stop_at_end: # If it's a looping path and we've reached the end of _segments
                self.current_segment_idx = 0 # Loop back to the first segment
                next_segment_path = self._segments[self.current_segment_idx]
                return next_segment_path.compute_reference(current_position, current_velocity, max_acceleration, step_index)
            # else: if stop_at_end is True and self.current_segment_idx >= len(self._segments),
            # the robot has completed its path and the initial check handles it.

        return x_des, x_des_dot, x_des_ddot