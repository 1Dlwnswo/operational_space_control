import numpy as np
from datetime import datetime
from pathlib import Path

def create_experiment_dir(experiment_name: str, results_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = results_root / experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi # (-pi, pi]