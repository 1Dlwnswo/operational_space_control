def limit_torque(tau, max_torque):
    for i in range(len(tau)):
        if abs(tau[i]) > max_torque:
            scale = max_torque / abs(tau[i])
            for j in range(len(tau)):
                tau[j] *= scale
    return tau