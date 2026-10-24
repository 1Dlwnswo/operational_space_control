import csv
import json
from dataclasses import asdict
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.parameters import ImpedanceControlParameters, RobotParameters, SimulationParameters

def save_log(log_path: Path, simulation_log: list, sim_params: SimulationParameters) -> Path:
    
    if simulation_log[0].save_os == None:
        vector_fields = ("q", "qdot", "tau", "a_cmd")
    else:
        vector_fields = ("q", "qdot", "tau", "a_cmd", "qddot", "x", "xdot", "xddot")
    
    fieldnames = ["time"] + [
        f"{field}_{index + 1}"
        for field in vector_fields
        for index in range(2)
    ]

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step, result in enumerate(simulation_log):
            row = {"time": step * sim_params.dt}
            for field in vector_fields:
                values = getattr(result, field, None)
                if values is not None:
                    row.update({
                        f"{field}_{index + 1}": value
                        for index, value in enumerate(values)
                    })
            writer.writerow(row)

    return log_path


def read_log(log_path: Path) -> List[Dict[str, Optional[float]]]:
    with log_path.open("r", newline="", encoding="utf-8") as f:
        return [
            {
                key: float(value) if value else None
                for key, value in row.items()
            }
            for row in csv.DictReader(f)
        ]


def save_metrics(metrics_path: Path, simulation_metrics: list, sim_params: SimulationParameters) -> Path:
    metric = simulation_metrics
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    rows = [
        [
            step * sim_params.dt,
            error,
            metric.distance_errors[step],
            metric.manipulability[step],
        ]
        for step, error in enumerate(metric.errors)
    ]

    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not write_header:
            writer.writerow([])
        writer.writerow(["rmse", metric.rmse, 0, 0])
        writer.writerow(["convergence_time", metric.convergence_time, 0, 0])
        writer.writerow([
            "saturation_torque_ratio", metric.saturation_torque_ratio, 0, 0
        ])
        writer.writerow(["time", "errors", "distance_errors", "manipulability"])
        writer.writerows(rows)

    return metrics_path


def read_metrics(metrics_path: Path) -> List[Dict[str, object]]:
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    simulations = []
    row_index = 0
    while row_index < len(rows):
        if not rows[row_index]:
            row_index += 1
            continue

        metric = {"rmse": float(rows[row_index][1])}
        row_index += 1
        metric["convergence_time"] = float(rows[row_index][1])
        row_index += 1
        if (
            row_index < len(rows)
            and rows[row_index]
            and rows[row_index][0] == "saturation_torque_ratio"
        ):
            metric["saturation_torque_ratio"] = float(rows[row_index][1])
            row_index += 1
        row_index += 1  # Skip the time-series column header.

        times = []
        errors = []
        distance_errors = []
        manipulability = []
        while row_index < len(rows) and rows[row_index]:
            times.append(float(rows[row_index][0]))
            errors.append(float(rows[row_index][1]))
            distance_errors.append(float(rows[row_index][2]))
            manipulability.append(float(rows[row_index][3]))
            row_index += 1

        metric.update({
            "time": times,
            "errors": errors,
            "distance_errors": distance_errors,
            "manipulability": manipulability,
        })
        simulations.append(metric)

    return simulations


def save_config(
    experiment_dir: Path,
    sim_params: SimulationParameters,
    control_params: ImpedanceControlParameters,
    robot_params: RobotParameters,
) -> Path:
    
    config = {
        "simulation_parameters": asdict(sim_params),
        "robot_parameters": asdict(robot_params),
        "impedance_control_parameters": asdict(control_params),
    }

    def format_value(value, indent_level=0):
        if isinstance(value, dict):
            indent = "  " * indent_level
            child_indent = "  " * (indent_level + 1)
            items = [
                f'{child_indent}{json.dumps(key)}: '
                f'{format_value(item, indent_level + 1)}'
                for key, item in value.items()
            ]
            return "{\n" + ",\n".join(items) + f"\n{indent}}}"
        if isinstance(value, np.ndarray):
            return json.dumps(
                value.tolist(),
                separators=(", ", ": "),
                ensure_ascii=False,
            )
        return json.dumps(value, ensure_ascii=False)
    
    config_path = experiment_dir / "config.json"
    
    with config_path.open("w", encoding="utf-8") as f:
        f.write(format_value(config) + "\n")
    return config_path


def read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_experiment_dir(experiment_name: str) -> Path:
    
    project_root = Path("/home/junjae/Documents/work/operational_space_control")
    results_root = project_root / "experiment"
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    experiment_dir = results_root / experiment_name / "results" / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)
    
    return experiment_dir

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi # (-pi, pi]

def main():
    create_experiment_dir(experiment_name = "LinePath")
    
if __name__ == "__main__":
    main()
