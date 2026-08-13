import numpy as np

def uniform_grid(workspace_radius, step_size):
    # Define target end-effector positions (grid distribution)

    # Generate grid points
    x_range = np.arange(-workspace_radius, workspace_radius + step_size, step_size)
    z_range = np.arange(-workspace_radius, workspace_radius + step_size, step_size)

    target_positions = []
    for x in x_range:
      for z in z_range:
        # Check if the point is within the circular workspace
          if np.sqrt(x**2 + z**2) <= workspace_radius:
            target_positions.append((x, z))

    # Remove duplicates (though grid generation typically doesn't create them unless floats are tricky)
    target_positions = list(set(target_positions))

    # Sort for consistent plotting/ordering: Z-value first (smallest), then X-value (smallest)
    target_positions.sort(key=lambda p: (p[1], p[0]))

    return target_positions

def linear_grid(initial_ee_pos, final_ee_pos, n):
    target_positions = []
    for i in range(n):
        t = i / (n - 1)
        x = initial_ee_pos[0] + t * (final_ee_pos[0] - initial_ee_pos[0])
        z = initial_ee_pos[1] + t * (final_ee_pos[1] - initial_ee_pos[1])
        target_positions.append((x, z))
    return target_positions