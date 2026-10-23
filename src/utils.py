import json
from dataclasses import asdict
import numpy as np
from datetime import datetime
from pathlib import Path

from src.parameters import ImpedanceControlParameters, SimulationParameters


def save_config(experiment_dir: Path, sim_params: SimulationParameters, control_params: ImpedanceControlParameters) -> Path:
    
    config = {
        "simulation_parameters": asdict(sim_params),
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
