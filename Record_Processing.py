def read_peer_nga2(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract event name and station name
    #event_name = lines[0].strip()
    station_name = lines[0].strip()

    # Extract metadata
    metadata = lines[1].strip().split(",")
    dt = float(metadata[1].split("=")[1].strip().split()[0])
    num_points = int(metadata[0].split("=")[1].strip())

    # Extract acceleration data
    acceleration = []
    for line in lines[2:]:
        acceleration.extend([float(value.strip()) for value in line.split()])

    return dt, num_points, station_name, acceleration


import os

def read_peer_nga2_folder(folder):
    # Initialize lists to store data
    all_dt = []
    all_num_points = []
    all_station_names = []
    all_accelerations = []

    # Get all filenames in the folder
    filenames = [filename for filename in os.listdir(folder) if filename.endswith(".AT2")]

    # Sort filenames in ascending order
    filenames.sort(key=lambda x: int(x.split(".")[0]))

    # Iterate over all sorted filenames
    for filename in filenames:
        # Construct the full path to the file
        filepath = os.path.join(folder, filename)
        
        # Read data from the file
        dt, num_points, station_name, acceleration = read_peer_nga2(filepath)
        
        # Append data to the lists
        all_dt.append(dt)
        all_num_points.append(num_points)
        all_station_names.append(station_name)
        all_accelerations.append(acceleration)

    return all_dt, all_num_points, all_station_names, all_accelerations

# Example usage:
folder = "record_without_info"
dt_list, num_points_list, station_names_list, accelerations_list = read_peer_nga2_folder(folder)




import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz

g = 9.81 # m/s2
    
def plot_accelerogram(dt, acceleration, station_name, ax):
    num_points = len(acceleration)
    time_length = num_points * dt
    time = np.linspace(0, time_length, num_points)

    # Check if the size of time array is consistent with acceleration array
    if len(time) != len(acceleration):
        if len(time) == len(acceleration) + 1:
            time = time[:-1]  # Reduce the size of the time array by one
        elif len(time) == len(acceleration) - 1:
            time = np.append(time, time[-1] + dt)  # Increase the size of the time array by one
        else:
            raise ValueError("Inconsistent sizes of time and acceleration arrays.")

    ax.plot(time, acceleration, color='blue', linewidth=1.0)
    ax.set_title(f"{station_name}", fontsize=20)
    ax.set_xlabel("Time (s)",fontsize = 14)
    ax.set_ylabel("Acceleration (g)",fontsize = 14)
    ax.grid(True)
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlim(0, time[-1])


def plot_velocity(dt, acceleration, station_name, ax):
    num_points = len(acceleration)
    time_length = num_points * dt
    time = np.linspace(0, time_length, num_points)

    # Check if the size of time array is consistent with acceleration array
    if len(time) != len(acceleration):
        if len(time) == len(acceleration) + 1:
            time = time[:-1]  # Reduce the size of the time array by one
        elif len(time) == len(acceleration) - 1:
            time = np.append(time, time[-1] + dt)  # Increase the size of the time array by one
        else:
            raise ValueError("Inconsistent sizes of time and acceleration arrays.")
    acceleration = [a * g * 100 for a in acceleration]
    velocity = cumtrapz(acceleration, dx=dt, initial=0)
    ax.plot(time, velocity, color='red', linewidth=1.5)
    ax.set_title(f"{station_name}", fontsize=20)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Velocity (cm/s)", fontsize=14)
    ax.grid(True)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylim(-70, 70)
    ax.set_xlim(0, time[-1])

def plot_cumulative_energy(dt, acceleration, station_name, ax):
    num_points = len(acceleration)
    time_length = num_points * dt
    time = np.linspace(0, time_length, num_points)

    # Check if the size of time array is consistent with acceleration array
    if len(time) != len(acceleration):
        if len(time) == len(acceleration) + 1:
            time = time[:-1]  # Reduce the size of the time array by one
        elif len(time) == len(acceleration) - 1:
            time = np.append(time, time[-1] + dt)  # Increase the size of the time array by one
        else:
            raise ValueError("Inconsistent sizes of time and acceleration arrays.")
    acceleration = [a * g * 100 for a in acceleration]
    velocity = cumtrapz(acceleration, dx=dt, initial=0)
    energy = np.cumsum(velocity ** 2 * dt)
    ax.plot(time, energy, color='green', linewidth=2.0)
    ax.set_title(f"{station_name}", fontsize=20)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Cumulative Energy (m^2/s^2)", fontsize=14)
    ax.grid(True)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylim(0, 4000) # energy[-1]
    ax.set_xlim(0, time[-1])


# Plotting accelerograms, velocity time-histories, and cumulative energy diagrams
fig, ((ax1_acc, ax2_acc), (ax1_vel, ax2_vel), (ax1_energy, ax2_energy)) = plt.subplots(3, 2, figsize=(12, 20))

rec_index = 3
file_index_a = rec_index*2-2
file_index_b = rec_index*2-1

# Plot for File 1
plot_accelerogram(dt_list[file_index_a], accelerations_list[file_index_a], station_names_list[file_index_a], ax1_acc)
plot_velocity(dt_list[file_index_a], accelerations_list[file_index_a], station_names_list[file_index_a], ax1_vel)
plot_cumulative_energy(dt_list[file_index_a], accelerations_list[file_index_a], station_names_list[file_index_a], ax1_energy)

# Plot for File 2
plot_accelerogram(dt_list[file_index_b], accelerations_list[file_index_b], station_names_list[file_index_b], ax2_acc)
plot_velocity(dt_list[file_index_b], accelerations_list[file_index_b], station_names_list[file_index_b], ax2_vel)
plot_cumulative_energy(dt_list[file_index_b], accelerations_list[file_index_b], station_names_list[file_index_b], ax2_energy)

plt.tight_layout()
plt.show()




def calculate_cumulative_energy(acceleration, dt):
    acceleration = np.array(acceleration)  # Convert to NumPy array
    velocity = cumtrapz(acceleration * g * 100.0, dx=dt, initial=0)  # Convert acceleration to cm/s
    energy = cumtrapz(velocity ** 2, dx=dt, initial=0)
    return energy[-1]

def compare_energy_pairs(accelerations_list, dt_list):
    assert len(accelerations_list) % 2 == 0, "The accelerations list must contain even number of elements."

    higher_energy_indices = []
    for i in range(0, len(accelerations_list), 2):
        energy1 = calculate_cumulative_energy(accelerations_list[i], dt_list[i])
        energy2 = calculate_cumulative_energy(accelerations_list[i + 1], dt_list[i + 1])
        higher_energy_indices.append(i if energy1 > energy2 else i + 1)

    return higher_energy_indices


higher_energy_indices = compare_energy_pairs(accelerations_list, dt_list)

high_energy_accelerations = [accelerations_list[i] for i in higher_energy_indices]
high_energy_dt = [dt_list[i] for i in higher_energy_indices]
high_energy_num_points = [num_points_list[i] for i in higher_energy_indices]



#|||||||||||||||||||||||||||||||||||||||||||||||| PGA scaling ||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

high_energy_PGAs = []

for acceleration in high_energy_accelerations:
    max_acceleration = max(np.abs(acceleration))
    high_energy_PGAs.append(max_acceleration)

high_energy_SFs = []
for PGA in high_energy_PGAs:
    Scale_Factor = 1 / PGA
    high_energy_SFs.append(Scale_Factor)

# scale to 1g
high_energy_scaled_accelerations = []

# Iterate over both high_energy_accelerations and high_energy_SFs simultaneously
for acceleration, scale_factor in zip(high_energy_accelerations, high_energy_SFs):
    # Multiply each acceleration record by its corresponding scale factor
    scaled_acceleration = [a * scale_factor for a in acceleration]
    high_energy_scaled_accelerations.append(scaled_acceleration)


num_points = len(high_energy_accelerations[rec_index])
time_length = num_points * high_energy_dt[rec_index]
time = np.linspace(0, time_length, num_points)

# Check if the size of time array is consistent with acceleration array
if len(time) != len(acceleration):
    if len(time) == len(acceleration) + 1:
        time = time[:-1]  # Reduce the size of the time array by one
    elif len(time) == len(acceleration) - 1:
        time = np.append(time, time[-1] + high_energy_dt[rec_index]) 
   
plt.figure(figsize=(18, 5))      
plt.subplot(1,2,1)
plt.plot(time, high_energy_accelerations[rec_index], color = 'b')
plt.title(f'Free Field Record: {station_names_list[rec_index*2-2]}', fontsize = 14 , fontweight = 'bold')
plt.ylim(-1, 1)
plt.xlim(0,time[-1])
plt.xlabel("Time (s)",fontsize = 14)
plt.ylabel("Acceleration (g)",fontsize = 14)
plt.grid(True)
plt.axhline(y=0, color='black', linewidth=1.5)


plt.subplot(1,2,2)
plt.plot(time, high_energy_scaled_accelerations[rec_index], color = 'b')
plt.title(f'PGA Scaled to 1g: {station_names_list[rec_index*2-2]}', fontsize = 14 , fontweight = 'bold')
plt.ylim(-1, 1)
plt.xlim(0,time[-1])
plt.xlabel("Time (s)",fontsize = 14)
plt.ylabel("Acceleration (g)",fontsize = 14)
plt.grid(True)
plt.axhline(y=0, color='black', linewidth=1.5)
plt.show()

    
#||||||||||||||||||||||||||||||||||||||||| Response Spectrum plot |||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


from statespace import statespace

# Generate a range of periods from 0 to 4 seconds.
periods = np.zeros((201, 1))
periods[:, 0] = np.linspace(0.02, 4, 201)  # Example of periods array
damping_ratio = 0.05  # 5% damping
omega_n = 2 * np.pi / periods
m = 1
K = m * omega_n**2 
dt = high_energy_dt[rec_index]
acceleration_time_history = high_energy_scaled_accelerations[rec_index]

spectral_displacement = statespace(damping_ratio, K, omega_n, dt, acceleration_time_history)
spectral_velocity = omega_n * spectral_displacement
spectral_acceleration = omega_n**2 * spectral_displacement / g
# Plotting the pseudo-acceleration spectrum


plt.figure(figsize=(8, 12))
plt.subplot(3,1,1)
plt.plot(periods, spectral_displacement, color='b', label=f'Damping Ratio = {damping_ratio * 100}%')
plt.ylabel('Spectral Displacement (m)', fontsize = 11)
plt.title(f'Response Spectra: {station_names_list[rec_index*2-2]}', fontsize = 18 , fontweight = 'bold')
plt.grid(True)
plt.xlim(0,4)
plt.legend()
plt.show()

plt.subplot(3,1,2)
plt.plot(periods, spectral_velocity, color='b', label=f'Damping Ratio = {damping_ratio * 100}%')
plt.ylabel('Pseudo Spectral Velocity (m/s)', fontsize = 11)
plt.grid(True)
plt.xlim(0,4)
plt.legend()
plt.show()

plt.subplot(3,1,3)
plt.plot(periods, spectral_acceleration, color='b', label=f'Damping Ratio = {damping_ratio * 100}%')
plt.xlabel('Period (seconds)', fontsize = 11)
plt.ylabel('Pseudo Spectral Acceleration (g)', fontsize = 12)
plt.grid(True)
plt.xlim(0,4)
plt.legend()
plt.show()




# Read seismic spectrum file

# file_path = "2800-type2-group1-spectrum.txt"
# data = np.loadtxt(file_path, skiprows=1)  # Skip the first row (header)

# period = data[:, 0]  # First column
# Standard_Spectrum = data[:, 1]      # Second column

# # Plot seismic spectrum
# plt.figure(figsize=(10, 6))
# plt.plot(period, Standard_Spectrum, color='blue', linewidth=2)
# plt.title('2800 Type II Spectrum for Highest Seismicity', fontsize=16)
# plt.xlabel('Period (sec)', fontsize=14)
# plt.ylabel('Spectral Acceleration (g)', fontsize=14)
# plt.xlim(0, 4)
# plt.grid(True)
# plt.show()

#||||||||||||||||||||||||||||||||||||||||||||||||| Save 1g Records |||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def create_records_folder(records_folder, high_energy_accelerations, high_energy_dt, high_energy_num_points):
    # Create a folder named Records_Scaled_1g if it doesn't exist
    records_scaled_folder = os.path.join(records_folder, 'Records_Scaled_1g')
    os.makedirs(records_scaled_folder, exist_ok=True)
    
    # Write each acceleration record to a separate file
    for i, acceleration in enumerate(high_energy_scaled_accelerations, start=1):
        filename = os.path.join(records_scaled_folder, f'{i}.txt')
        with open(filename, 'w') as file:
            for value in acceleration:
                file.write(f'{value}\n')
    
    # Write dt values to a file named time_intervals
    time_intervals_file = os.path.join(records_scaled_folder, 'time_intervals.txt')
    with open(time_intervals_file, 'w') as file:
        for dt in high_energy_dt:
            file.write(f'{dt}\n')
    
    # Write number of points for each record to a file
    num_points_file = os.path.join(records_scaled_folder, 'num_points.txt')
    with open(num_points_file, 'w') as file:
        for num_points in high_energy_num_points:
            file.write(f'{num_points}\n')
            
# records_folder = 'OpenSees_Records'
# create_records_folder(records_folder, high_energy_scaled_accelerations, high_energy_dt, high_energy_num_points)



#|||||||||||||||||||||||||||||||||||||||| Response Spectrum Scaling ||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


#---------opensees----------   5-st     10-st      15-st
fundamental_mode_period = [0.867017, 1.253463, 1.520271]


# from scipy.interpolate import interp1d

def get_spectral_acceleration(index):
    g = 9.81  
    m = 1  
    periods = np.zeros((201, 1))
    periods[:, 0] = np.linspace(0.02, 4, 201) 
    damping_ratio = 0.05  # 5% damping
    omega_n = 2 * np.pi / periods
    K = m * omega_n**2
    rec_dt = high_energy_dt[index]
    acceleration_time_history = high_energy_scaled_accelerations[index]
    
    # State space representation to get spectral displacement
    spectral_displacement = statespace(damping_ratio, K, omega_n, rec_dt, acceleration_time_history)

    spectral_acceleration = omega_n**2 * spectral_displacement / g
    
    return spectral_acceleration


data = np.loadtxt('2800-type2-group1-spectrum.txt',  skiprows=1)
period = data[:, 0]  # First column
Standard_Spectrum = data[:, 1]  # Second column



scale_factors_for_standard = [[], [], []]
stepwise_scale_factors = [[], [], []]
g = 9.81


# spectral_accelerations = []
# scaled_spectral_accelerations = []  # To store all the scaled results
# for i in range(len(high_energy_scaled_accelerations)):
    
#     spectral_acceleration = get_spectral_acceleration(i)
#     spectral_accelerations.append(spectral_acceleration)
    
    
    
    
 #------------------------------------------------------------------------------------------------------------
    # Interpolate the standard spectrum to the periods used for spectral acceleration
    
    # interpolation_func = interp1d(period, Standard_Spectrum, kind='linear', fill_value='extrapolate')
    # interpolated_standard_spectrum = interpolation_func(period)

    # Compute scale factors for each period within bounds for each fundamental mode period
    # for j, T in enumerate(fundamental_mode_period):
    #     # Calculate the range
    #     low_bound = 0.2 * T
    #     high_bound = 1.5 * T
    
    #     # Iterate through each period to determine the required scale factor
    #     for k, period_k in enumerate(period):
    #         if low_bound <= period_k <= high_bound:
    #             # Compute scale factor only within the bounds
    #             scale_factor = 1.17 * Standard_Spectrum[k] / spectral_acceleration[k]
    #             stepwise_scale_factors[j].append(scale_factor)
                
    #         else:
    #             stepwise_scale_factors[j].append(1)
    
    #     # max_scale_factor = max(scale_factors_for_standard[j])
    #     # And apply it to the spectral acceleration
    #     # spectral_acceleration *= scale_factor
    #     scale_factors_for_standard[j] = max(stepwise_scale_factors[j])
    #     temp_spectral_acc = spectral_acceleration
    #     temp_spectral_acc *= scale_factors_for_standard[j]
    #     scaled_spectral_accelerations.append(spectral_acceleration)
        


# Create a folder to store the data if it doesn't exist
folder_name = "Scaled_Spectral_Data_1g"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Define the file path for saving the data
file_path = os.path.join(folder_name, "spectral_accelrations_1g.txt")

# Combine spectral accelerations into one array & Save the data to a text file

# data_to_save = np.column_stack(spectral_accelerations)
# np.savetxt(file_path, data_to_save, fmt='%0.6f', delimiter='\t')



periods = np.zeros((201, 1))
periods[:, 0] = np.linspace(0.01, 4, 201)

spectral_accelerations_loaded = np.loadtxt("Scaled_Spectral_Data_1g/spectral_accelrations_1g.txt")
enhanced_standard_spectrum = [1.17 * value for value in Standard_Spectrum]


def calculate_scale_factors(T, periods, spectral_accelerations_loaded, enhanced_standard_spectrum):
    spectral_scale_factors = []

    # Define the bounds based on the fundamental period T
    low_bound = 0.2 * T
    high_bound = 1.5 * T

    indices_in_bounds = np.where((periods >= low_bound) & (periods <= high_bound))[0]  # Ensure array is 1D
    # indices_in_bounds = indices_in_bounds.tolist()

    for spectrum in spectral_accelerations_loaded.T:

        spect_in_bounds = spectrum[indices_in_bounds]

        standard_spectrum_in_bounds = [enhanced_standard_spectrum[i] for i in indices_in_bounds]
        # standard_spectrum_in_bounds = enhanced_standard_spectrum[indices_in_bounds]

        scale_factor = 1
        
        for spect_val, standard_val in zip(spect_in_bounds, standard_spectrum_in_bounds):
            # If the record value is lower than the standard value, compute scale factor
            if spect_val < standard_val:
                scale_factor = max(scale_factor, standard_val / spect_val)

        # Append the scale factor to the list
        spectral_scale_factors.append(scale_factor)

    return spectral_scale_factors


spectral_SF_list = []
for T in fundamental_mode_period:
    spectral_scale_factors = calculate_scale_factors(T, periods, spectral_accelerations_loaded, enhanced_standard_spectrum)
    spectral_SF_list.append(spectral_scale_factors)

def save_to_txt(spectral_SF_list, filename, folder):
    # Make sure the folder exists, create if not
    os.makedirs(folder, exist_ok=True)
    
    # Define the full path for the text file
    filepath = os.path.join(folder, filename)

    # Transpose the spectral_SF_list to get columns
    transposed_data = zip(*spectral_SF_list)
    
    # Write the transposed data to the file
    with open(filepath, 'w') as file:
        for row in transposed_data:
            file.write(' '.join(map(str, row)) + '\n')

save_to_txt(spectral_SF_list, 'scale_factors.txt', 'Scaled_Spectral_Data_1g')



#import SF
def load_from_txt(filename, folder):
    # Define the full path for the text file
    filepath = os.path.join(folder, filename)
    
    # Read the transposed data from the file
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Split each line into a list of values and convert to float
    # Assuming that all data is float
    transposed_data = [list(map(float, line.split())) for line in lines]
    
    # Transpose the data to get original lists
    original_data = list(zip(*transposed_data))
    
    return original_data

# Example usage:
spectral_SF_list = load_from_txt('scale_factors.txt', 'Scaled_Spectral_Data_1g')


# Plot the original spectra and the standard spectrum
plt.figure(figsize=(8, 5))  # Adjust size as needed
for spect in spectral_accelerations_loaded.T:  # Transpose the array to iterate over columns
    plt.plot(periods, spect, color='grey', linestyle='-', linewidth=0.5)
plt.plot(periods, spect, color='grey', linestyle='-', linewidth=0.5, label='PGA-scaled Spectra (1g)')
plt.plot(periods, enhanced_standard_spectrum, color='black', linewidth=2, label='1.17x Standard 2800 Spectrum')
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.legend()
plt.title('Unscaled Spectra', fontsize=14)
plt.xlabel('Period (s)')
plt.ylabel('PSA (g)')
plt.show()


fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Adjust size as needed

building_stories = [5, 10, 15]

# For each subplot
for i, scale_factors in enumerate(spectral_SF_list):
    for idx, spect in enumerate(spectral_accelerations_loaded.T):
        # Scale each record's spectrum by its corresponding scale factor
        scaled_spectra = np.multiply(spect, scale_factors[idx])
        axs[i].plot(periods, scaled_spectra, color='grey', linestyle='-', linewidth=0.5)
        
    axs[i].plot(periods, np.multiply(spect, scale_factors[idx]), color='grey', linestyle='-', linewidth=0.5, label='Scaled Spectra')
    T = fundamental_mode_period[i]
    axs[i].axvline(x=0.2*T, color='red', linestyle='--', linewidth=3, label='Scaling Target Range (0.2T to 1.5T)')
    axs[i].axvline(x=1.5*T, color='red', linestyle='--', linewidth=3)
    axs[i].plot(periods, enhanced_standard_spectrum, color='black', linewidth=2, label='1.17x Standard 2800 Spectrum')
    axs[i].set_xlim(0, 4)
    axs[i].set_ylim(0, 4)
    axs[i].legend()
    axs[i].set_title(f'Scaled Spectra for {building_stories[i]}-story Building', fontsize=14)
    axs[i].set_xlabel('Period (s)')
    axs[i].set_ylabel('PSA (g)')

plt.tight_layout()
plt.show()


