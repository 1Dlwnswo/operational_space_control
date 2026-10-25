import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SIMULATIONS = {
    "1": ("Circle Path", "experiment.CirclePath.simulation"),
    "2": ("Line Path", "experiment.LinePath.simulation"),
    "3": ("Polygon Path", "experiment.PolygonPath.simulation"),
}

USE_VIEWER = False
GENERATE_PLOT = True
GENERATE_GIF = False


def main():
    while True:
        print("\n========== Robot Arm Simulation ==========\n")

        for key, (name, _) in SIMULATIONS.items():
            print(f"{key}. {name}")

        print("0. Exit")

        choice = input("\nSelect simulation: ").strip()

        if choice == "0":
            print("Exiting.")
            break

        if choice not in SIMULATIONS:
            print("Invalid selection. Try again.")
            continue

        name, module = SIMULATIONS[choice]

        print(f"\nRunning {name}...\n")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                module,
                "--viewer" if USE_VIEWER else "--no-viewer",
                "--plot" if GENERATE_PLOT else "--no-plot",
                "--gif" if GENERATE_GIF else "--no-gif",
            ],
            cwd=PROJECT_ROOT,
        )

        if result.returncode == 0:
            print(f"\n{name} completed successfully.")
        else:
            print(f"\n{name} failed with exit code {result.returncode}.")


if __name__ == "__main__":
    main()
