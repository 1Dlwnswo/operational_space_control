import numpy as np


def uniform_grid(workspace_radius, step_size):
    x_indices = np.arange(
        np.ceil(-workspace_radius / step_size),
        np.floor(workspace_radius / step_size) + 1,
    )
    z_indices = np.arange(
        np.ceil(-workspace_radius / step_size - 0.5),
        np.floor(workspace_radius / step_size - 0.5) + 1,
    )
    x_range = z_indices * step_size
    z_range = (x_indices + 0.5) * step_size

    target_positions = []
    for x in x_range:
        for z in z_range:
            if np.hypot(x, z) < workspace_radius:
                target_positions.append((x, z))

    target_positions.sort(key=lambda p: (p[0] + p[1], p[0]))

    return target_positions


def linear_grid(initial_ee_pos, final_ee_pos, n):
    target_positions = []
    for i in range(n):
        t = i / (n - 1)
        x = initial_ee_pos[0] + t * (final_ee_pos[0] - initial_ee_pos[0])
        z = initial_ee_pos[1] + t * (final_ee_pos[1] - initial_ee_pos[1])
        target_positions.append((x, z))
    return target_positions


def main():
    grid = uniform_grid(2.0, 0.8)

    import matplotlib.pyplot as plt

    circle = plt.Circle(
        (0, 0),
        2.0,
        facecolor="gray",
        alpha=0.1,
        fill=True,
        linestyle="--",
        edgecolor="black",
    )
    plt.gca().add_patch(circle)

    plt.scatter(*zip(*grid))
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.title("Uniform Grid Points")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
