# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:41:44 2020

@author: hcarv
"""



import sys
import os

#Path where program is stored.
sys.path.append('/media/dataHog/hca/proLig/')
base_path='/media/dataHog/hca/proLig/'

import main


workdir='/media/dataHog/hca/calb-MeOH/'

os.chdir(workdir)
results=workdir+"notebook_results"

if not os.path.exists(results):
    os.makedirs(results)
    
    
#concentrations=['300mM', '600mM']
#concentrations=['50mM', '150mM', '300mM', '600mM', '1M', '2.5M', '5.5M']
#concentrations_scalar=[0.3, 0.6]
parameter={"50mM":0.05, "150mM":0.150, "300mM":0.3, "600mM":0.6, "1M":1, "2.5M":2.5, "5.5M":5.5} #   {"Folder": scalar}
stride=2

project=main.Project(workdir=workdir, parent_folder='calb-MeOH', results_folder=results, parameter=parameter,
	replicas=10, trajectory='calb.xtc', topology='calb.pdb', protein='calb', ligand='MeOH', timestep=5)


systems=project.create() 

#print(systems['parent_folder'])


selection='resname MeOH'    


#for parameter, v in systems.items():
#    print(parameter)
#    for k,i in v.items():
#        print(f'{k}: {i}')


main.Trajectory(systems).density(selection=selection, cat_file='trj-cat', unit='Molar')
