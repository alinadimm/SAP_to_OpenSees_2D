import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

# Constants
story_height = 320.0 # cm
base_directory = 'Results'
results_folder_names = [['5-story_results'], ['10-story_results'], ['15-story_results']]

# Plot for each structure
for folder in results_folder_names:
    structure_name = folder[0]
    num_stories = int(structure_name.split('-')[0])+1
    
    path = os.path.join(base_directory, structure_name)
    drifts_all_records = np.zeros((num_stories, 22)) # +1 for the ground story
    
    for record in range(1, 23): # Loop through each record
        file_name = f'max_disps_record_{record}.txt'
        file_path = os.path.join(path, file_name)
        
        with open(file_path, 'r') as file:
            displacements = [0.0] + [float(line.strip()) for line in file]  # Add zero for ground level
            
        # Calculate drifts
        drifts = [0.0] + [
            (displacements[i] - displacements[i-1]) / story_height
            for i in range(1, len(displacements))
        ]
        
        drifts_all_records[:num_stories, record-1] = drifts  # Store drifts for the current record
    
    # Calculate average drifts across all 22 records
    avg_drifts = np.mean(drifts_all_records, axis=1)
    
    # Smoothing using interpolation
    x = np.arange(num_stories)
    f = interp1d(x, drifts_all_records, kind='cubic', axis=0)
    x_smooth = np.linspace(0, num_stories - 1, 100)
    drifts_all_records_smooth = f(x_smooth)
    
    # Plot
    plot_width = 4
    plot_height =  plot_width * num_stories/5
    plt.figure(figsize=(plot_width, plot_height))
    for record in range(22):
        plt.plot(drifts_all_records_smooth[:, record], x_smooth, color='gray', lw=0.5)  # Plot each record in gray
    plt.plot(drifts_all_records_smooth[:, record], x_smooth, color='gray', lw=0.5, label = 'FEMA Far-Field Records')
    plt.plot(avg_drifts, x, color='black', lw=2, label = 'Average Drift Ratios')  # Plot the average in black with line
    plt.scatter(avg_drifts, x, color='black', marker='o')  # Add circles for average drifts
    
    plt.xlabel('Drift Ratio')
    plt.ylabel('Story Number')
    plt.title(f'Drift Distribution for {num_stories - 1}-story Structure', fontsize = 14)
    plt.grid(True)
    plt.legend()
    plt.show()
    