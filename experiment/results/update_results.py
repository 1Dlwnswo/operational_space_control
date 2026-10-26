import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"

PATH_GIFS = {
    "LinePath": "robot_arm_full_trajectory.gif",
    "CirclePath": "robot_arm_circular_path.gif",
    "PolygonPath": "robot_arm_polygonal_path.gif",
}


def latest_experiment(path_results_dir: Path):
    experiments = sorted(
        (path for path in path_results_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    return experiments[-1] if experiments else None


def update_results() -> None:
    for path_name, gif_name in PATH_GIFS.items():
        latest = latest_experiment(EXPERIMENT_ROOT / path_name / "results")
        if latest is None:
            print(f"[skip] {path_name}: 실험 결과 폴더가 없습니다.")
            continue

        destination = RESULTS_ROOT / path_name
        destination.mkdir(parents=True, exist_ok=True)

        plot = latest / "figures" / "combined_figures.png"
        gif = RESULTS_ROOT / gif_name

        for source, output_name in (
            (plot, "combined_figures.png"),
            (gif, "simulation.gif"),
        ):
            if source.is_file():
                shutil.copy2(source, destination / output_name)
                print(f"[copy] {source.relative_to(EXPERIMENT_ROOT)}")
            else:
                print(f"[skip] {source.relative_to(EXPERIMENT_ROOT)} 없음")


if __name__ == "__main__":
    update_results()
