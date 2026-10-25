import numpy as np


def get_sensor_addresses(model):
    sensor_names = [
        "joint1_pos",
        "joint2_pos",
        "joint1_vel",
        "joint2_vel",
    ]

    return {name: model.sensor(name).adr for name in sensor_names}


def read_sensors(model, data):

    sensor_addresses = get_sensor_addresses(model)

    q = np.array(
        [
            data.sensordata[sensor_addresses["joint1_pos"]],
            data.sensordata[sensor_addresses["joint2_pos"]],
        ]
    )

    qdot = np.array(
        [
            data.sensordata[sensor_addresses["joint1_vel"]],
            data.sensordata[sensor_addresses["joint2_vel"]],
        ]
    )

    return q.reshape(-1), qdot.reshape(-1)
