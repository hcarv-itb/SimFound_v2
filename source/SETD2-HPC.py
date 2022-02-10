#!/usr/bin/env python
# coding: utf-8

# ## **SETD2_complexed_noSub with SAM @HPC**
# ## **Import modules**
import os
import sys

replicate=sys.argv[0]
replicate_init=replicate

data_path=os.path.abspath('/pfs/work7/workspace/scratch/st_ac131353-SETD2-0/SETD2/') #Where your group data is
base_path=os.path.abspath('/home/st/st_us-030800/st_ac131353/SimFound_v2/source/') #Where your source code is (SFv2)

sys.path.append(base_path)
sys.path.append(data_path)

import warnings
warnings.filterwarnings('ignore')
import Protocols as P

import main
import tools

from simtk.unit import *


# ## **Define Project**
workdir=data_path
results=workdir+'/results'
inputs=workdir+'inputs/structures/'

workdir=tools.Functions.fileHandler([workdir], _new=False)
tools.Functions.fileHandler([results, inputs])

protein=['setd2_complexed_noSub']
ligand=['SAM']
parameters=['310K']
timestep=20*picoseconds

project=main.Project(title='SETD2-SAM', 
                     hierarchy=('protein', 'ligand', 'parameter'), 
                     workdir=workdir,
                     parameter=parameters, 
                     replicas=replicate, 
                     protein=protein, 
                     ligand=ligand, 
                     results=results,
                     topology='SETD2_complexed_noSub.pdb',
                     timestep=timestep,
                    initial_replica=replicate_init)


project_systems=project.setSystems()

# ## **Simulation Protocol**
# ## Set defaults
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
                'step': 5*nanoseconds, 
                'report': timestep, 
                'restrained_sets': {'selections': ['protein and backbone', 'resname ZNB', 'resname SAM'],
                                    'forces': [100*kilojoules_per_mole/angstroms, 150*kilojoules_per_mole/angstroms, 150*kilojoules_per_mole/angstroms]}}

NPT_production={'ensemble': 'NPT',
                'step': 500*nanoseconds, 
                'report': timestep}


# ## Setup simulation protocol
extra_pdb = ['ZNB_complexed.pdb', 'SAM.pdb']
xml_files = ['protein.ff14SB.xml', 'tip4pew.xml', 'ZNB.xml', 'SAM.xml', 'bonded.xml', 'gaff.xml']

simulated_systems={}
for name, system in project_systems.items():

    if system.ligand == 'SAM':
    
        sim_tools=P.Protocols(workdir=system.name_folder, 
                              project_dir=project.workdir,
                              def_input_struct=project.def_input_struct,
                              def_input_ff=project.def_input_ff,
                             overwrite_mode=False)
    
        system_omm=sim_tools.pdb2omm(input_pdb=system.input_topology,
                                     extra_input_pdb=extra_pdb,
                                     ff_files=xml_files,
                                     protonate=True,
                                     solvate=True,
                                     pH_protein=7.0,
                                     name=name,
                                     residue_variants=SETD2_variants)
                                    #insert_molecules=True)
                                    #insert_smiles=smiles)
                                      
        simulated_systems[name]=system_omm


# ## **WARNING!** Run simulation(s)
simulations={}
for name, simulation in simulated_systems.items():  
    simulation.setSimulations(dt=0.002*picoseconds, 
                                  temperature = system.scalar,
                                  pH=7.0,
                                  friction = 1/picosecond,
                                  equilibrations=[NPT_equilibration],
                                  productions=[NPT_production],
                                  pressure=1*atmospheres)
    
    
    simulation.runSimulations(where='HPC',
                              run_productions=True,
                              run_equilibrations=True,
                              compute_time=47.9,
                              reportFactor=0.005)        
    
    simulations[name]=simulation


for k, v in simulations.items():
    print(k, v)

