# units: kgf-cm
# 2D Model

import openseespy.opensees as ops
import vfo.vfo as vfo
import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np
from process_2D_model_database import process_2D_model_database
from successful_run_sound import *
import os
import math
import time

start_time = time.time()


def read_scale_factors(filename):
    with open(filename, 'r') as file:
        scale_factors = np.array([line.split() for line in file], dtype=float)
    return scale_factors

# Path to the scale factors file
scale_factors_file = 'Scaled_Spectral_Data_1g/scale_factors.txt'
num_points_file = 'OpenSees_Records/Records_Scaled_1g/num_points.txt'
time_intervals_file = 'OpenSees_Records/Records_Scaled_1g/time_intervals.txt'

scale_factors = read_scale_factors(scale_factors_file)
# print(scale_factors)
database_name_list = ['5ST_Model_Database.xlsx',
                      '10ST_Model_Database.xlsx',
                      '15ST_Model_Database.xlsx']
#                 story 1    2   3 ...                                  ... 14  15
drift_tracking_tags = [[20, 21, 22, 23, 24],
                       [20, 21, 22, 23, 24, 44, 45, 46, 47, 48],
                       [20, 21, 22, 23, 24, 61, 62, 63, 64, 28, 44, 45, 46, 47, 48]]
results_folder_names = [['5-story_results'],
                        ['10-story_results'],
                        ['15-story_results']]
for idx, folder in enumerate(results_folder_names):
    folder_path = os.path.join('Results', folder[0])
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
# Function to read data from the text file
def read_data_file(filename):
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

# Paths to the files with number of points and time intervals

num_points = read_data_file(num_points_file)
time_intervals = read_data_file(time_intervals_file)


        
# Main loop for the analysis

max_disp_results = []
for idx, database_name in enumerate(database_name_list):
    
    structure_max_disps = []
    node_disps = []
    for record in range(1, 23): 
        print(f'>>>>>>Analysis Started for Model {idx+1}, Record {record}.')
        current_record_max_disp = [0.0] * len(drift_tracking_tags[idx])
        record_file = f"OpenSees_Records/Records_Scaled_1g/{record}.txt"
       
        scale_factor = scale_factors[record - 1, idx]   
        
        
        #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #///////////////////////////////////        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #////////////////////////////////// MODELING \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   PHASE  //////////////////////////////////
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\        ///////////////////////////////////
        #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        
        ops.wipe()
        process_2D_model_database(database_name) #read modeling and loading data + create model
        
                
        #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #///////////////////////////////////        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #////////////////////////////////// ANALYSIS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   PHASE  //////////////////////////////////
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\        ///////////////////////////////////
        #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    

 
        
        numEigenvalues = 12
        # eig = ops.eigen(numEigenvalues)
        # T=[]
        # for i in range(len(eig)):
        #     T.append((2*np.pi)/sqrt(eig[i]))
            
        ops.modalDamping(0.05)
        
        #--------------------------- Gravity Load Case -----------------------------
        ops.wipeAnalysis()

        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('BandGen')
        tol = 1e-8
        iteration = 400
        ops.test('NormDispIncr', tol, iteration)
        ops.algorithm('Newton')
        incr = 0.1
        ops.integrator('LoadControl', incr)
        ops.analysis('Static')
        ops.eigen(numEigenvalues)
        incrNumber = int(1/incr)
        ops.analyze(incrNumber)


        ops.loadConst('-time', 0.0)

        # opsv.plot_defo()
        # opsv.plot_loads_2d()

        #--------------------------- NLTHA -----------------------------
        ops.wipeAnalysis()
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('UmfPack')
        
        tol = 1e-6
        iteration = 400
        ops.test('EnergyIncr', tol, iteration, 5)
        
        ops.algorithm('Newton')
        
        gamma = 0.5
        beta = 0.25
        ops.integrator('Newmark', gamma, beta)
        

        #ops.analyze(numIncr=1, dt=0.0, dtMin=0.0, dtMax=0.0, Jd=0)        

        

        
        dt = time_intervals[record - 1]
        num_step = int(num_points[record - 1])
        
        # Read and apply the ground motion for the particular record
        accel_data = np.genfromtxt(record_file, dtype=None)
        gravity_adjusted_data = accel_data * 981.0
        
        TH_tag = 5
        pattern_tag = 6
        TH_factor = scale_factor * 981.0
        ops.timeSeries('Path', TH_tag,'-dt', dt, '-filePath', record_file, '-factor', TH_factor)
        ops.pattern('UniformExcitation', pattern_tag, 1, '-accel', TH_tag,'-fact', 1)

        ops.analysis('Transient')
        
        # Perform the time history analysis
        algorithm_sequence = ['Newton', 'KrylovNewton', 'RaphsonNewton', 'BFGS', 'NewtonLineSearch']
        previous_step = 0
        # Perform the time history analysis
        disp_plot = [0]
        for step in range(num_step):
            
            record_time = step * dt
            int_part, frac_part = math.modf(record_time)
            if frac_part != previous_step:
                if abs(int_part) > 0.9: # or another suitable tolerance
                    previous_step = frac_part
                    print(f'Model {idx+1}, Record {record}: second {frac_part}')
    
            analysis_result = ops.analyze(1, dt)
        
            # If the initial analysis step fails
            if analysis_result != 0:
                warning_sound()
                print(f"Analysis failed at time {step * dt} seconds for record {record}. Trying with smaller time steps.")
        
                # Try reducing the time step first
                sub_steps = 10
                sub_dt = dt / sub_steps
                for sub_step in range(sub_steps):
                    if ops.analyze(1, sub_dt) == 0:
                        break
                    # If reducing the timestep still fails, attempt different algorithms
                    elif sub_step == sub_steps - 1:  # Only try algorithms at the last sub-step
                        for algorithm in algorithm_sequence:
                            ops.algorithm(algorithm)  # Switch to the next algorithm in the sequence
                            if ops.analyze(1, sub_dt) == 0:
                                print(f"Convergence achieved using {algorithm} algorithm at time {step * dt + sub_step * sub_dt} seconds for record {record}.")
                                break
                        else:
                            # If all algorithm attempts fail at the last sub-step
                            print(f"Analysis ultimately failed at time {step * dt + sub_step * sub_dt} seconds for record {record}. Aborting analysis.")
                            break
        
                if analysis_result != 0:
                    # All attempts have failed; exiting main loop
                    print(f"Unable to converge at time {step * dt} seconds for record {record}. Stopping analysis.")
                    error_sound()
                    break
                
            if analysis_result == 0:
                
                disp_plot.append(ops.nodeDisp(24, 1))
                for i, tag in enumerate(drift_tracking_tags[idx]):
                    
                    node_disp = abs(ops.nodeDisp(tag, 1))  #  drift is in X-direction.
                    node_disps.append(node_disp)
                    # Update max drift for the node if the current drift is larger.
                    current_record_max_disp[i] = max(current_record_max_disp[i], node_disp)
                # Re-apply the load pattern with updated time step
                # ops.remove('loadPattern', pattern_tag)
                # ops.timeSeries('Path', TH_tag,'-dt', dt, '-filePath', record_file, '-factor', TH_factor, '-time', step * dt)
                # ops.pattern('UniformExcitation', pattern_tag, 1, '-accel', TH_tag,'-fact', 1)
            # Continue the loop if the step was successful
        #--------------------plot only for one analysis --------------------    
        # timeline = np.linspace(start=0, stop=num_step * dt , num=num_step+1, endpoint=True)
        # plt.plot(timeline, disp_plot)
        # plt.xlabel('time (second)', fontsize=12)
        # plt.ylabel('roof displacement (cm)', fontsize=12)
        # plt.grid()
        #--------------------------------------------------------------------
        print(f'Analysis Completed for Model {idx+1}, Record {record}.') 
        print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||')
        structure_max_disps.append(current_record_max_disp)
        
        results_file_name = f"max_disps_record_{record}.txt"
        results_file_path = os.path.join('Results', results_folder_names[idx][0], results_file_name)
        with open(results_file_path, 'w') as results_file:
            for disp in current_record_max_disp:
                results_file.write(f"{disp}\n")

    max_disp_results.append(structure_max_disps)
    ops.wipe()
successful_run_sound()

elapsed_time = time.time() - start_time
print("Total time taken by the analysis:", elapsed_time, "seconds")


    
    
    
    
    