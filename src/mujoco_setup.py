import mujoco
import numpy as np

MODEL_PATH = "model/arm_2dof.xml"

def load_model_and_data(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data

def get_robot_parameters(model):
    from src.parameters import RobotParameters

    m1 = model.body_mass[model.body('link1').id]
    m2 = model.body_mass[model.body('link2').id]
    I1 = model.body_inertia[model.body('link1').id][1] # Moment of inertia for link1 about its joint axis
    I2 = model.body_inertia[model.body('link2').id][1] # Moment of inertia for link2 about its joint axis
    L1 = 1.0
    L2 = 1.0
    c1 = 0.5 # Center of mass for link1 is at L1/2 relative to its body frame
    c2 = 0.5 # Center of mass for link2 is at L2/2 relative to its body frame
    g = abs(model.opt.gravity[2]) # Magnitude of gravity (assuming gravity is along -Z)

    return RobotParameters(m1=m1, m2=m2, I1=I1, I2=I2, L1=L1, L2=L2, c1=c1, c2=c2, g=g)

def get_simulation_parameters(model):
    from src.parameters import SimulationParameters
    dt = model.opt.timestep
    num_steps = 6000
    return SimulationParameters(dt=dt, num_steps=num_steps)