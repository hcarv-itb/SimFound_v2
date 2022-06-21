#!/usr/bin/env python
# coding: utf-8

# ## **SETD2-ssK36 launcher @HPC**

import os
import sys
if len(sys.argv) > 1:
    replicate=sys.argv[1]
    run_time=sys.argv[2]
else:
    print('Missing inputs: ', replicate, run_time)
    exit(1)
replicate_init=replicate

data_path=os.path.abspath(f'/pfs/work7/workspace/scratch/st_ac131353-SETD2/SETD2/')
base_path=os.path.abspath('/home/st/st_us-030800/st_ac131353/SimFound_v2/source/') #Where source code is (SFv2)

sys.path.append(base_path)
sys.path.append(data_path)

import importlib

import warnings
warnings.filterwarnings('ignore')
import Protocols as P

import main
import tools
import Featurize as F
import Trajectory
import Visualization


from simtk.unit import *


# ## **Define Project**

# In[6]:


importlib.reload(main)
importlib.reload(Trajectory)
importlib.reload(P)
importlib.reload(tools)

#protein=['WT', 'R167Q', 'I602G']
protein=['SETD2']
ligand=['ssK36']
parameters=['1to1']#, '1to2', '1to4']
timestep=20*picoseconds
box_size=11.0


project=main.Project(title='SETD2-ssK36', 
                     hierarchy=('protein', 'ligand', 'parameter'), 
                     workdir=data_path,
                     parameter=parameters, 
                     replicas=replicate, 
                     protein=protein, 
                     ligand=ligand,
                     topology='SETD2_complexed_noSub.pdb',
                     timestep=timestep,
                    initial_replica=replicate)


project.setSystems()


SETD2_variants={('A',1499): 'CYX',
                ('A',1501): 'CYX',
                ('A',1516): 'CYX',
                ('A',1520): 'CYX',
                ('A',1529): 'CYX',
                ('A',1533):'CYX',
                ('A',1539):'CYX',
                ('A',1631):'CYX',
                ('A',1678):'CYX',
                ('A',1680):'CYX',
                ('A',1685):'CYX'}

NPT_equilibration={'ensemble': 'NPT',
                'step': 10*nanoseconds, 
                'report': timestep, 
                'restrained_sets': {'selections': ['protein and backbone', 'resname ZNB', 'resname SAM'],
                                    'forces': [100*kilojoules_per_mole/angstroms, 150*kilojoules_per_mole/angstroms, 150*kilojoules_per_mole/angstroms]}}

NPT_production={'ensemble': 'NPT',
                'step': 1000*nanoseconds, 
                'report': timestep}


# ## Setup simulation protocol


importlib.reload(P)
importlib.reload(tools)
xml_files = ['protein.ff14SB.xml', 'tip4pew.xml', 'ZNB.xml', 'SAM.xml', 'bonded.xml', 'gaff.xml']
extra_pdb = ['SETD2-ssK36_1-1_B_edit.pdb', 
             'SETD2-ssK36_1-1_C_edit.pdb', 
             'SETD2-ssK36_1-1_D.pdb'] #, 
             #'SETD2-ssK36_1-4_E.pdb', 
             #'SETD2-ssK36_1-4_F.pdb', 
             #'SETD2-ssK36_1-4_G.pdb']

simulation=P.Protocols(project, overwrite_mode=False, def_input_struct=None)
simulation.pdb2omm(input_pdb='SETD2-ssK36_1-1_A.pdb',
                    ff_files=xml_files,
                    protonate=True,
                    solvate=True,
                    pH_protein=7.0,
                    box_size=box_size,  
                    residue_variants=SETD2_variants,
                    extra_input_pdb=extra_pdb)

simulation.setSimulations(dt=0.002*picoseconds, 
                                  temperature = 310,
                                  pH=7.0,
                                  friction = 1/picosecond,
                                  equilibrations=[NPT_equilibration],
                                  productions=[NPT_production],
                                  pressure=1*atmospheres)


# ## **WARNING!** Run simulation(s)

#simulation.runSimulations(where='BwUniCluster',  run_time_HPC=run_time_HPC, partition=partition, ws_name=ws_name) 
simulation.runSimulations(where='Local',
                          run_productions=True,
                          run_equilibrations=True,
                          compute_time=run_time,
                          reportFactor=0.005)  




