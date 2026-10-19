import matplotlib.pyplot as plt
import numpy as np
from src.parameters import RobotParameters, SimulationParameters
from src.path.CirclePath import CirclePath

from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


def plot_circular_path_results(x_log, q_log, torque_log, circular_path_obj: CirclePath ,sim_params: SimulationParameters, robot_params: RobotParameters):
    _, axes_circular = plt.subplots(3, 3, figsize=(18, 12))
    
    ee_x_log_circular = x_log[:, 0]
    ee_z_log_circular = x_log[:, 1]
    q1_log_circular = q_log[:, 0]
    q2_log_circular = q_log[:, 1]
    torque1_log_circular = torque_log[:, 0]
    torque2_log_circular = torque_log[:, 1]
    
    dt = sim_params.dt
    
    circular_path_center = circular_path_obj.center
    circular_path_radius = circular_path_obj.radius
    
    # Create time_log for circular trajectory based on its simulation parameters
    time_log_circular = [i * dt for i in range(sim_params.num_steps)]

    # --- End-effector X, Z Position over Time ---
    axes_circular[0,0].plot(time_log_circular, ee_x_log_circular, label='End-effector X')
    axes_circular[0,0].plot(time_log_circular, ee_z_log_circular, label='End-effector Z')
    axes_circular[0,0].set_title('End-effector X, Z Position for Circular Path')
    axes_circular[0,0].set_xlabel('Time [s]')
    axes_circular[0,0].set_ylabel('Position [m]')
    axes_circular[0,0].grid(True)
    axes_circular[0,0].legend()

    # --- Joint Angles (q1, q2) over Time ---
    axes_circular[0,1].plot(time_log_circular, q1_log_circular, label='Joint 1 Angle')
    axes_circular[0,1].plot(time_log_circular, q2_log_circular, label='Joint 2 Angle')
    axes_circular[0,1].set_title('Joint Angles for Circular Path')
    axes_circular[0,1].set_xlabel('Time [s]')
    axes_circular[0,1].set_ylabel('Angle [rad]')
    axes_circular[0,1].grid(True)
    axes_circular[0,1].legend()

    # --- Joint Torques (torque1, torque2) over Time ---
    axes_circular[0,2].plot(time_log_circular, torque1_log_circular, label='Joint 1 Torque')
    axes_circular[0,2].plot(time_log_circular, torque2_log_circular, label='Joint 2 Torque')
    axes_circular[0,2].set_title('Joint Torques for Circular Path')
    axes_circular[0,2].set_xlabel('Time [s]')
    axes_circular[0,2].set_ylabel('Torque [Nm]')
    axes_circular[0,2].grid(True)
    axes_circular[0,2].legend()

    # --- End-effector XY Trajectory Plot for Circular Trajectory ---
    axes_circular[1,0].plot(ee_x_log_circular, ee_z_log_circular, label='End-effector Trajectory', color='blue')

    # Plot the desired circular path
    theta = np.linspace(0, 2 * np.pi, 100) # Generate angles for a full circle
    desired_x = circular_path_center[0] + circular_path_radius * np.cos(theta)
    desired_z = circular_path_center[1] + circular_path_radius * np.sin(theta)
    axes_circular[1,0].plot(desired_x, desired_z, 'r--', label='Desired Circular Path')

    # Plot workspace boundary
    workspace_radius = robot_params.L1 + robot_params.L2
    circle_osc = plt.Circle((0, 0), workspace_radius, color='gray', alpha=0.1, fill=True, linestyle='--', edgecolor='black')
    axes_circular[1,0].add_patch(circle_osc)

    axes_circular[1,0].set_title('End-effector Trajectory for Circular Path')
    axes_circular[1,0].set_xlabel('X Position [m]')
    axes_circular[1,0].set_ylabel('Z Position [m]')
    axes_circular[1,0].set_aspect('equal', adjustable='box')
    axes_circular[1,0].grid(True)
    axes_circular[1,0].legend()
    
    
    # --- Fit Sine Function to End-effector X and Z Positions ---
    def sine_function(t, A, omega, phi, C):
        return A * np.sin(omega * t + phi) + C

    # Convert logs to numpy arrays for curve fitting
    np_time_log_circular = np.array(time_log_circular)
    np_ee_x_log_circular = np.array(ee_x_log_circular)
    np_ee_z_log_circular = np.array(ee_z_log_circular)

    # Initial guess for parameters: [Amplitude, Angular Frequency, Phase, Offset]
    x_initial_guess = [circular_path_radius, circular_path_obj.cruise_angular_speed, np.pi/2, circular_path_center[0]]
    z_initial_guess = [circular_path_radius, circular_path_obj.cruise_angular_speed, 0, circular_path_center[1]]

    # Fit the sine function to ee_x_log_circular
    params_x, covariance_x = curve_fit(sine_function, np_time_log_circular, np_ee_x_log_circular, p0=x_initial_guess)

    # Fit the sine function to ee_z_log_circular
    params_z, covariance_z = curve_fit(sine_function, np_time_log_circular, np_ee_z_log_circular, p0=z_initial_guess)

    # Extract fitted parameters
    A_x, omega_x, phi_x, C_x = params_x
    A_z, omega_z, phi_z, C_z = params_z

    print(f"Fitted parameters for X: Amplitude={A_x:.4f}, Angular Freq={omega_x:.4f}, Phase={phi_x:.4f}, Offset={C_x:.4f}")
    print(f"Fitted parameters for Z: Amplitude={A_z:.4f}, Angular Freq={omega_z:.4f}, Phase={phi_z:.4f}, Offset={C_z:.4f}")

    # Generate fitted data
    x_fitted = sine_function(np_time_log_circular, A_x, omega_x, phi_x, C_x)
    z_fitted = sine_function(np_time_log_circular, A_z, omega_z, phi_z, C_z)

    # Plot X-coordinate
    axes_circular[1,1].plot(np_time_log_circular, np_ee_x_log_circular, label='Original End-effector X', color='blue', alpha=0.7)
    axes_circular[1,1].plot(np_time_log_circular, x_fitted, \
        label = fr'y = Asin(ωt+φ)+C' + '\n' + fr' (A={A_x:.4f}, ω={omega_x:.4f}, φ={phi_x:.4f}, C={C_x:.4f})', color='red', linestyle='--')
    axes_circular[1,1].set_title('End-effector X Position and Fitted Sine Function Over Time')
    axes_circular[1,1].set_xlabel('Time [s]')
    axes_circular[1,1].set_ylabel('X Position [m]')
    axes_circular[1,1].grid(True)
    axes_circular[1,1].legend(loc='upper left')

    # Plot Z-coordinate
    axes_circular[1,2].plot(np_time_log_circular, np_ee_z_log_circular, label='Original End-effector Z', color='green', alpha=0.7)
    axes_circular[1,2].plot(np_time_log_circular, z_fitted, \
        label = fr'y = Asin(ωt+φ)+C' + '\n' + fr' (A={A_z:.4f}, ω={omega_z:.4f}, φ={phi_z:.4f}, C={C_z:.4f})', color='purple', linestyle='--')
    axes_circular[1,2].set_title('End-effector Z Position and Fitted Sine Function Over Time')
    axes_circular[1,2].set_xlabel('Time [s]')
    axes_circular[1,2].set_ylabel('Z Position [m]')
    axes_circular[1,2].grid(True)
    axes_circular[1,2].legend(loc='upper left')



    # --- Frequency Domain Analysis ---
    # Calculate FFT for X position
    N_circular = len(np_ee_x_log_circular) # Number of sample points
    T_circular = dt # Sample spacing

    x_fft = fft(np_ee_x_log_circular)
    z_fft = fft(np_ee_z_log_circular)

    # Calculate frequencies
    xf = fftfreq(N_circular, T_circular)[:N_circular//2]

    # Get magnitude spectrum (take only the positive frequencies)
    yf_x = 2.0/N_circular * np.abs(x_fft[0:N_circular//2])
    yf_z = 2.0/N_circular * np.abs(z_fft[0:N_circular//2])
    

    # Find peaks in X and Z magnitude spectrum
    # We set a minimum height threshold to ignore minor noise/sidelobes
    peaks_x, properties_x = find_peaks(yf_x, height=0.05)
    peaks_z, properties_z = find_peaks(yf_z, height=0.05)

    # Get peak frequencies and corresponding magnitudes
    peak_freqs_x = xf[peaks_x]
    peak_mags_x = yf_x[peaks_x]

    peak_freqs_z = xf[peaks_z]
    peak_mags_z = yf_z[peaks_z]
    
    max_peak_idx_x = np.argmax(peak_mags_x)
    max_peak_freq_x = peak_freqs_x[max_peak_idx_x]
    max_peak_mag_x = peak_mags_x[max_peak_idx_x]
    
    max_peak_idx_z = np.argmax(peak_mags_z)
    max_peak_freq_z = peak_freqs_z[max_peak_idx_z]
    max_peak_mag_z = peak_mags_z[max_peak_idx_z]

    # Plot X-coordinate frequency spectrum
    axes_circular[2,0].plot(xf, yf_x, color='blue')
    axes_circular[2,0].set_title('Frequency Spectrum of End-effector X Position')
    axes_circular[2,0].set_xlabel('Frequency [Hz]')
    axes_circular[2,0].set_ylabel('Magnitude')
    axes_circular[2,0].grid(True)
    axes_circular[2,0].scatter(max_peak_freq_x, max_peak_mag_x, color='red', label= f'Max Peak: {max_peak_freq_x:.4f} Hz')
    axes_circular[2,0].set_xlim(0, 2) # Limit x-axis to relevant frequencies        
    axes_circular[2,0].legend()

    # Plot Z-coordinate frequency spectrum
    axes_circular[2,1].plot(xf, yf_z, color='green')
    axes_circular[2,1].set_title('Frequency Spectrum of End-effector Z Position')
    axes_circular[2,1].set_xlabel('Frequency [Hz]')
    axes_circular[2,1].set_ylabel('Magnitude')
    axes_circular[2,1].grid(True)
    axes_circular[2,1].scatter(max_peak_freq_z, max_peak_mag_z, color='red', label= f'Max Peak: {max_peak_freq_z:.4f} Hz')
    axes_circular[2,1].set_xlim(0, 2) # Limit x-axis to relevant frequencies
    axes_circular[2,1].legend()
    
    
    # Calculate end-effector velocities by numerical differentiation
    ee_x_vel_circular = np.gradient(np_ee_x_log_circular, np_time_log_circular)
    ee_z_vel_circular = np.gradient(np_ee_z_log_circular, np_time_log_circular)

    # Calculate the total speed (magnitude of the velocity vector)
    ee_speed_circular = np.sqrt(ee_x_vel_circular**2 + ee_z_vel_circular**2)

    # Plot X-coordinate velocity
    axes_circular[2,2].plot(np_time_log_circular, ee_x_vel_circular, label='End-effector X Velocity', color='blue')

    # Plot Z-coordinate velocity
    axes_circular[2,2].plot(np_time_log_circular, ee_z_vel_circular, label='End-effector Z Velocity', color='green')

    # Plot total speed
    axes_circular[2,2].plot(np_time_log_circular, ee_speed_circular, label='End-effector Total Speed', color='red', linestyle='--')
    
    cruise_speed = circular_path_obj.cruise_angular_speed * circular_path_obj.radius
    axes_circular[2,2].axhline(y=cruise_speed, color='orange', linestyle=':', label=f'Desired Cruise Speed: {cruise_speed:.2f} m/s')

    axes_circular[2,2].set_title('End-effector X, Z Velocities and Total Speed Over Time')
    axes_circular[2,2].set_xlabel('Time [s]')
    axes_circular[2,2].set_ylabel('Speed [m/s]')
    axes_circular[2,2].grid(True)
    axes_circular[2,2].legend()
    
    plt.tight_layout()
    plt.suptitle('Circular Path Tracking Simulation Results', y=1.02, fontsize=16)    
    plt.savefig('experiment/results/circular_path_simulation_results.png', dpi=300, bbox_inches='tight') # Save the figure as a high-resolution PNG
    
    plt.show(block=False)
    plt.pause(5)
    plt.close()