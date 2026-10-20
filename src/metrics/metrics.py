import numpy as np
from src.path.LinePath import LinePath

def calculate_linepath_rmse(x_log: np.ndarray, linepath_obj: LinePath):
    
    errors = linepath_obj.calculate_path_errors(x_log)
    
    return np.sqrt(np.mean(errors**2))