# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:41:44 2020

@author: hcarv
"""



import pyemma
import os
import glob
import pandas as pd
import numpy as np
import nglview as nv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter, MaxNLocator

import sys


#Path were classes are stored
sys.path.append('../subAnalyser')

import importlib

import base
import plots

importlib.reload(base)
importlib.reload(plots)


workdir="../calb-MeOH/"
os.chdir(workdir)
results=workdir+"notebook_results"
if not os.path.exists(results):
    os.makedirs(results)
    
    
#concentrations=['300mM', '600mM']
concentrations=['50mM', '150mM', '300mM', '600mM', '1M', '2.5M', '5.5M']
#concentrations_scalar=[0.3, 0.6]
parameter_scalar={"50mM":0.05, "150mM":0.150, "300mM":0.3, "600mM":0.6, "1M":1, "2.5M":2.5, "5.5M":5.5}
stride=2

project=base.System(workdir=workdir, parent_folder='calb-MeOH', results_folder=results,
                      parameter=parameter_scalar.keys(), parameter_scalar=parameter_scalar.values(), 
                      replicas=10, trajectory='calb.xtc', topology='calb.pdb', protein='calb', ligand='MeOH', timestep=5)

#system.getProperties('protein')


systems=project.create() 

print(systems['protein'])

importlib.reload(base)

selection='resname MeOH'    


#for parameter, v in systems.items():
#    print(parameter)
#    for k,i in v.items():
#        print(f'{k}: {i}')


base.Trajectory(systems).density(selection=selection, unit='Molar')





#base.Trajectory(systems).superposeTrajs(selection='name CA')
