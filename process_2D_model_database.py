
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#///////////////////////////////////        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////// MODELING \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   PHASE  //////////////////////////////////
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\        ///////////////////////////////////
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# units: kgf-cm
# 2D Model

import pandas as pd
import openseespy.opensees as ops
import vfo.vfo as vfo
import opsvis as opsv
import matplotlib.pyplot as plt
from Steel_FiberSec import *
from numpy import sqrt
import numpy as np

def process_2D_model_database(model_database):

    ops.wipe()
    DL_factor = 1.2
    LL_factor = 1.0

    ndm = 2
    ndf = 3

    
    ops.model('basic', '-ndm', ndm, '-ndf', ndf)
    
    Model_Database = model_database
    Joint_Coordinates = pd.read_excel(Model_Database, sheet_name="Joint Coordinates",header=[1])
    Joint_Coordinates = Joint_Coordinates.drop(0)
    Joint_Coordinates = Joint_Coordinates.reset_index(drop="True")
    
    
    # Import Joint Coordinates
    for i in range(len(Joint_Coordinates)):
        ops.node(int(Joint_Coordinates.loc[i,"Joint"]),
                  float(Joint_Coordinates.loc[i,"XorR"]),
                  float(Joint_Coordinates.loc[i,"Z"]),
                  float(Joint_Coordinates.loc[i,"Y"]))
        
        
    Assembled_Joint_Masses = pd.read_excel(Model_Database, sheet_name='Assembled Joint Masses',header=[1])
    Assembled_Joint_Masses = Assembled_Joint_Masses.drop(0)
    Assembled_Joint_Masses = Assembled_Joint_Masses.reset_index(drop='True')
        
    # Import Joint Masses    
    for i in range(len(Assembled_Joint_Masses)-3):
        ops.mass(int(Assembled_Joint_Masses.loc[i,'Joint']),
                  float(Assembled_Joint_Masses.loc[i,'U1']),
                  float(Assembled_Joint_Masses.loc[i,'U2']),
                  0.000001)
        
        
    Joint_Restraint_Assignments = pd.read_excel(Model_Database, sheet_name='Joint Restraint Assignments',header=[1])
    Joint_Restraint_Assignments = Joint_Restraint_Assignments.drop(0)
    Joint_Restraint_Assignments = Joint_Restraint_Assignments.reset_index(drop='True')
    constrValues = {'Yes':1, 'No':0}
    # Importing Constraint Values
    for i in range(len(Joint_Restraint_Assignments)):
        ops.fix(int(Joint_Restraint_Assignments.loc[i,'Joint']) ,
                constrValues[Joint_Restraint_Assignments.loc[i,'U1']],
                constrValues[Joint_Restraint_Assignments.loc[i,'U2']],
                constrValues[Joint_Restraint_Assignments.loc[i,'R2']])
    
    # Define Material
    S235JR = 1 ; R0 = 18 ; cR1 = 0.925 ; cR2 = 0.15
    Fy = 3515.3481
    E0 = 2039432.4
    b = 0.03
    params = [R0,cR1,cR2]
    ops.uniaxialMaterial('Steel02', S235JR, Fy, E0, b, *params)
    
                         
    Frame_sec = pd.read_excel(Model_Database, sheet_name="Frame Props 01 - General",header=[1])
    Frame_sec = Frame_sec.drop(0)
    Frame_sec = Frame_sec.reset_index(drop="True")
    Frame_sec = Frame_sec.rename(columns = {'GUID' : 'Tag'})
    
    for i in range(len(Frame_sec)):
        Frame_sec.loc[i , 'Tag'] = i + 1
    sec_list = []
    for i in range(len(Frame_sec)):
        sec_list.append(Frame_sec.loc[i , 'SectionName'])
    
    Frame_sec = Frame_sec.set_index('SectionName')
    
    
    for i in sec_list:
        if str(Frame_sec.loc[i, 'Shape']) == 'I/Wide Flange':
            I_sec(int(Frame_sec.loc[i, 'Tag']), S235JR, E0, 
                  float(Frame_sec.loc[i, 't2']),
                  float(Frame_sec.loc[i, 'tf']),
                  float(Frame_sec.loc[i, 't3']),
                  float(Frame_sec.loc[i, 'tw']))
        elif str(Frame_sec.loc[i, 'Shape']) == 'Box/Tube':
            Box_sec(int(Frame_sec.loc[i, 'Tag']), S235JR, E0,
                    float(Frame_sec.loc[i, 't2']),
                    float(Frame_sec.loc[i, 't3']),
                    float(Frame_sec.loc[i, 'tf']),
                    float(Frame_sec.loc[i, 'tw']))            
    
    
    Connectivity_Frame = pd.read_excel(Model_Database, sheet_name='Connectivity - Frame',header=[1])
    Connectivity_Frame = Connectivity_Frame.drop(0)
    Connectivity_Frame = Connectivity_Frame.reset_index(drop='True')
    # Connectivity_Frame = Connectivity_Frame.set_index('Frame')
    
    
    Frame_Section_Assignments = pd.read_excel(Model_Database, sheet_name='Frame Section Assignments',header=[1])
    Frame_Section_Assignments = Frame_Section_Assignments.drop(0)
    Frame_Section_Assignments = Frame_Section_Assignments.reset_index(drop='True')
    # Frame_Section_Assignments = Frame_Section_Assignments.set_index('Frame')
    
    
    P_delta = 1
    ops.geomTransf('PDelta', P_delta)
    
    Number_of_Integration_Points = 6
    
    
    
    for i in range(len(sec_list)):
        ops.beamIntegration('Lobatto',
                            Frame_sec.loc[Frame_sec.index[i], 'Tag'],
                            Frame_sec.loc[Frame_sec.index[i], 'Tag'],
                            Number_of_Integration_Points)
    
    
    for i in range(len(Connectivity_Frame)):
        
        sect_name = Frame_Section_Assignments.loc[i, 'AnalSect']
        tag = Connectivity_Frame.loc[i, 'Frame']
        i_node = Connectivity_Frame.loc[i, 'JointI']
        j_node = Connectivity_Frame.loc[i, 'JointJ']
        
        if sect_name in ['STIFFBEAM', 'STIFFCOL']:
            
            tag = int(Connectivity_Frame.loc[i, 'Frame'])
            i_node = int(Connectivity_Frame.loc[i, 'JointI'])
            j_node = int(Connectivity_Frame.loc[i, 'JointJ'])
    
            A = 10*(Frame_sec.loc[sect_name, 'Area'] )
            I33 = 10*(Frame_sec.loc[sect_name, 'I33'])
            ops.element('elasticBeamColumn', tag, i_node, j_node,
                        A, E0, I33, P_delta)
    
        elif sect_name not in ['STIFFBEAM', 'STIFFCOL']:
      
            tag = int(Connectivity_Frame.loc[i, 'Frame'])
            i_node = int(Connectivity_Frame.loc[i, 'JointI'])
            j_node = int(Connectivity_Frame.loc[i, 'JointJ'])
            integrationTag = Frame_sec.loc[sect_name, 'Tag']
            A = Frame_sec.loc[sect_name, 'Area']  
    
            ops.element('forceBeamColumn', tag, i_node, j_node,
                        P_delta, integrationTag)
    
    
    
    # opsv.plot_model()                  
    # vfo.plot_model(show_nodetags='yes', line_width=3)
    
    numEigenvalues = 12
    eig = ops.eigen(numEigenvalues)
    # ops.modalProperties('-print')
    
    Modal_Periods_And_Frequencies = pd.read_excel(Model_Database, sheet_name='Modal Periods And Frequencies',header=[1])
    Modal_Periods_And_Frequencies = Modal_Periods_And_Frequencies.drop(0)
    Modal_Periods_And_Frequencies = Modal_Periods_And_Frequencies.reset_index(drop='True')
    Periods_list = []
    for i in range(len(Modal_Periods_And_Frequencies)):
        Periods_list.append(Modal_Periods_And_Frequencies.loc[i, 'Period'])
    # print(Periods_list)
    T=[]
    for i in range(len(eig)):
        T.append((2*3.14)/sqrt(eig[i]))
        
    # print(T)
    data = {'SAP2000': Periods_list, 
            'OpenSees': T}
    df = pd.DataFrame(data)
    df.index = ['Mode ' + str(i+1).zfill(2) for i in range(len(df))]
    # print(df)
    
    # opsv.plot_mode_shape(1)
    # vfo.plot_modeshape(modenumber = 2, scale=1000, line_width=3)
    
    
    Frame_Loads = pd.read_excel(Model_Database, sheet_name='Frame Loads - Distributed',header=[1])
    Frame_Loads = Frame_Loads.drop(0)
    Frame_Loads = Frame_Loads.reset_index(drop='True')
    
    Linear = 1
    ops.timeSeries('Linear', Linear)
    Gr = 1 # gravity
    ops.pattern('Plain', Gr, Linear, '-fact', 1)
    load_type = '-beamUniform'
    
    
    #----------------------------------------loading--------------------------------------------
    
    
    # row object is a tuple from df.iterrows()
    
    for index, row in Frame_Loads.iterrows(): #unpack the tuple into the index and row. 
        eleTag = int(row['Frame'])
        
        if row['LoadPat'] == 'DL':
            ops.eleLoad('-ele', eleTag, '-type', load_type, 
                    -1*DL_factor*row['FOverLA'], 0.0, 0.0)
            
            # print(str(eleTag) + '  DeadLoad')
       
        if row['LoadPat'] == 'LL':
            ops.eleLoad('-ele', eleTag, '-type', load_type, 
                    -1*LL_factor*row['FOverLA'], 0.0, 0.0)
    
            # print(str(eleTag) + '  LiveLoad')
    
    
        if row['LoadPat'] == 'colSelfWeight':
            # If the 'str' conversion didn't work, try using 'int' instead for integer inputs
            # SectRow = Frame_Section_Assignments.loc[Frame_Section_Assignments['Frame'] == str(eleTag)]  
            SectRow = Frame_Section_Assignments.loc[np.where(Frame_Section_Assignments['Frame'].astype(str) == str(eleTag), True, False)]
            AnalSect = SectRow['AnalSect'].iloc[0]
    
            area = Frame_sec.loc[AnalSect, 'Area']
            distributed = 7.849E-03 * area
    
            ops.eleLoad('-ele', eleTag, '-type', load_type, 0.0, -1*DL_factor*distributed, 0.0)
    
            # print(str(eleTag) + '  column self weight')
                  
                  
        if row['LoadPat'] == 'beamSelfWeight':
                  
            # SectRow = Frame_Section_Assignments.loc[Frame_Section_Assignments['Frame'] == str(eleTag)]
            SectRow = Frame_Section_Assignments.loc[np.where(Frame_Section_Assignments['Frame'].astype(str) == str(eleTag), True, False)]
            AnalSect = SectRow['AnalSect'].iloc[0]
            area = Frame_sec.loc[AnalSect, 'Area']
            distributed = 7.849E-03 * area
    
            ops.eleLoad('-ele', eleTag, '-type', load_type, -1*DL_factor*distributed, 0.0, 0.0)