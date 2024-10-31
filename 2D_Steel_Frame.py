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
from successful_run_sound import *

ops.wipe()
DL_factor = 1.2
LL_factor = 1.0

ndm = 2
ndf = 3

ops.model('basic', '-ndm', ndm, '-ndf', ndf)

Model_Database = "15ST_Model_Database.xlsx"
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
 
# Define Sections

# Isection = I_secplot(1,S235JR,E0, 15, 1.5, 30.0, 0.5)
                                                            
# matcolor = ['y', 'lightgrey', 'gold', 'w', 'w', 'w']
# opsv.plot_fiber_section(Isection, matcolor=matcolor)
# plt.axis('equal') 

# doublechannel = DoubleChannel_secplot(1,S235JR,E0, 9.0, 2.0, 26.0, 1.5,2.0)
                                                            
# matcolor = ['y', 'lightgrey', 'gold', 'w', 'w', 'w']
# opsv.plot_fiber_section(doublechannel, matcolor=matcolor)
# plt.axis('equal')      

# cruciformsec = Cruciform_secplot(1,S235JR,E0, 16.0, 2.0, 34.0, 1.0)
                                                            
# matcolor = ['y', 'lightgrey', 'gold', 'w', 'w', 'w']
# opsv.plot_fiber_section(cruciformsec, matcolor=matcolor)
# plt.axis('equal') 

# Box = Box_secplot(1,S235JR,E0, 40.0, 40.0, 2.0, 2.0)
                                                            
# matcolor = ['y', 'lightgrey', 'gold', 'w', 'w', 'w']
# opsv.plot_fiber_section(Box, matcolor=matcolor)
# plt.axis('equal') 
                     
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
print(df)

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
        
        # print(str(eleTag) + '  beam self weight')
                 
# successful_run_sound()

# #-----------------------Analysis Commands-------------------------       
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


ops.loadConst()


opsv.plot_defo()
# opsv.plot_loads_2d()


#-----------------------global forces-------------------------


# forces_df = pd.DataFrame(columns=['Fxi', 'Fyi', 'Mi', 'Fxj', 'Fyj', 'Mj', 'Unit'])   # Empty DataFrame

# for index, row in Frame_Section_Assignments.iterrows():
#     frame_tag = int(row['Frame'])
#     forces = ops.eleForce(frame_tag)
    
#     forces_row = {
#         'Fxi': forces[0],
#         'Fyi': forces[1],
#         'Mi': forces[2],
#         'Fxj': forces[3],
#         'Fyj': forces[4],
#         'Mj': forces[5],
#         'Unit': 'kgf-cm'  
#     }

#     forces_df = forces_df.append(forces_row, ignore_index=True)

# forces_df.index = Frame_Section_Assignments['Frame'].astype(int)

# print("\n Display Frame Global Forces:")
# print(forces_df)

# #-----------------------basic forces-------------------------
# basicforces_df = pd.DataFrame(columns=['P', 'Mi', 'Mj', 'Unit'])

# for index, row in Frame_Section_Assignments.iterrows():
#     frame_tag = int(row['Frame'])
    
#     # Get the basic forces using ops.basicForce
#     basic_forces = ops.basicForce(frame_tag)
    
#     # Create a row for the DataFrame
#     forces_row = {
#         'P': basic_forces[0],
#         'Mi': basic_forces[1],
#         'Mj': basic_forces[2],
#         'Unit': 'kgf-cm'  # Set the unit for all forces
#     }

#     basicforces_df = basicforces_df.append(forces_row, ignore_index=True)

# basicforces_df.index = Frame_Section_Assignments['Frame'].astype(int)

# print("\n Display Frame Basic Forces:")
# print(basicforces_df)


# ele_defo = []
# ele_force = []
# sec_force = [0]
# sec_defo = [0]
# Curve_list = []
# C_node_disp = []
# Base_shear = []

# elementTagToRecord = 56

# num_of_steps = 70
# C_node = 180




# ops.wipeAnalysis()
# ops.constraints('Plain')
# ops.numberer('RCM')
# ops.system('BandGen')
# ops.test('EnergyIncr', 1.e-8, 400)
# ops.algorithm('Newton')
# ops.integrator('DisplacementControl', C_node, 1 , 1)
# ops.analysis('Static')

# aa = []
# for step in range(1, num_of_steps + 1):
#     ops.analyze(1)
#     C_node_disp.append(ops.nodeDisp(C_node, 1))
#     ele_defo.append(ops.basicDeformation(elementTagToRecord))
#     ele_force.append(ops.basicForce(elementTagToRecord))
#     sec_force.append(ops.sectionForce(elementTagToRecord, 1,2))
#     sec_defo.append(ops.sectionDeformation(elementTagToRecord, 1,2))

#     ops.reactions()
#     Reaction_Force = 0
#     for react in range(len(Joint_Restraint_Assignments)):
#         Reaction_Force = Reaction_Force + abs(ops.nodeReaction(Joint_Restraint_Assignments.loc[react,'Joint'], 1))
#     Base_shear.append(Reaction_Force)
    
#  for cur in range(0,7):
#      Curv_list.append(ops.sectionDeformation(56, cur,2))
#  plt.plot(Curv_list)
#  plt.show()


#  ops.wipe()
   








