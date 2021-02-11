

"""
Main
====


The core python classes of the package.
Classes are implemented in jupyter notebooks of in offline runs (optional).


Created on Fri Jul  3 11:30:16 2020
@author: hca
"""

import numpy as np
import pandas as pd
import tools
import tools_plots
import Trajectory
import MSM
import Discretize



try:
    import pyemma
except:
    pass


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path



class Project:
    """Base class define a project.
    
    Parameters
    ----------
    workdir : path
        The working folder. The default is Path(os.getcwd()).
    title : str
        The title of the project. The default is "Untitled".
    hierarchy : TYPE, optional
        The ordered set of parameters to generate systems. 
        The default is ("protein", "ligand", "parameter").
        Note: For default value, folders will be protein-ligand-parameter/replicaN, with N being the replicate index.
    parameter : list
        The list of parameters. The default is "None".
    replicas : int
        Number of replicas contained within each parent_folder-parameter folder. The default is 1.
    protein : list
        Name of the protein(s). The default is "protein".
    ligand : list
        Name of the ligand(s). The default is "ligand".
    timestep : int
        The physical time of frame (ps). The default is 1.
    results : path
        The path where results will be stored.
    
    
    Returns
    -------
    self: object
        The project instance. 
    """

    
    def __init__(self, workdir, results, title="Untitled",
                 hierarchy=("protein", "ligand", "parameter"), parameter=None, 
                 replicas=1, protein='protein', ligand='ligand', timestep=1):
        
        self.workdir=workdir
        self.title=title
        self.hierarchy=hierarchy
        self.parameter=parameter
        self.replicas=int(replicas)
        self.protein=protein
        self.ligand=ligand
        self.timestep=timestep
        self.results=results
     

        
    
    def getProperties(self, *args) -> str:
       """
       Checks if queried properties is defined in self.
       
       Parameters
       ----------
       property: list
           Variable(s) to que queried.
           
       Returns
       -------
       str
           The value of property, if it is defined.
       """
       
       for prop in args:
           if prop in self.__dict__.keys():
               return self.__dict__[prop]
           else:
               print(f'Property "{prop}" not defined in system.')  
               
    
    def setSystems(self) -> dict:
        """
        Defines project systems based on hierarchy.


        Returns
        -------
        systems : dict
            A dictionary of system names and instances as defined in the project (see class `System`).

        """
        
        import itertools
        
        #tools.Functions.fileHandler([self.workdir, self.results], confirmation=defaults[confirmation])
        
        elements=[self.getProperties(element) for element in self.hierarchy] #retrieve values for each element in hierarchy.
        replicas=[str(i) for i in range(1,self.replicas+1)]
        
        elements.append(replicas)
    
        systems=list(itertools.product(*elements)) #generate all possible combinations of elements: systems.
        systems_obj=[] #Retrieve the systems from class System.
        for system in systems:
            systems_obj.append(System(system, workdir=self.workdir))
        
        systems={}
        for system in systems_obj:
            systems[system.name]=system
            #print(f'{system.name} \n\t{system}\n\t{system.name_folder}\n\t{system.path}')
            
        for k, v in systems.items():
            v.path=tools.Functions.fileHandler([v.path]) #Update the path if fileHandler generates a new folder
            
        self.systems=systems

        return systems
        
        
class System:
    """The base class to define a system. Takes as input a system tuple.
    The system tuple can be accepted as a instance of the parent class or given manually.
    See parent class method `setSystems` for details of tuple generation.
    
    Examples
    --------
    A tuple of the form ('protein', 'ligand', 'parameter') or other with permutations of the elements (hierarchical, top first).
    
    
    Parameters
    ----------

    system: tuple
        The tuple representing the system.
    workdir: path
        The default working directory.
    replica_name: str
        The prefix for replicate subsfolder. The default is 'sim'.
    linker: str
        The file name delimiter. The default is '-'.

    
    Returns
    -------
    
    systems: obj
        An instance of System.
    """
    
    
    def __init__(self, system, workdir='workdir', replica_name='sim', linker='-'):
        self.system=system
        self.workdir=workdir
        self.replica_name=replica_name
        self.linker=linker
        self.name=self.linker.join(self.system)
        self.name_folder=f'{self.linker.join(self.system[:-1])}/{self.replica_name}{self.system[-1]}'
        self.path=f'{self.workdir}/{self.name_folder}'
        self.results_folder=f'{self.path}/results/'
        self.trajectory=f'{self.path}/calb.xtc'
        self.topology=f'{self.path}/calb.pdb'
    
    

class Features:
    """Base class to create a *features* object. Different featurization schemes can be coded.
    
    Parameters
    ----------
    project: object
        Object instance of the class "Project".
        
    Returns
    -------
    
    feature: object
        Feature instance
    
    """
    
    def __init__(self, systems, results):
        self.systems=systems
        self.results=results
        
    def nac(self):
        """
        Read the stored NAC file. Returns a dictionary with NAC arrays for each concentration
        Uses the pyEMMA load module to load the NAC .npy array
        TODO: Migrate NAC calculation to this function"""
        
        import glob
        
        nac={}
        
        
        for name, system in self.systems.items():
            file=str(glob.glob(f'{system.results_folder}NAC2*.npy')[0])
            
            system.features={}
            nac[system.name]=file
            #system.features={'nac':str(file)} #Legacy
            system.features['nac']=file
                
        return ('nac', nac)
    

    def dist(self, dists, stride=1, skip_frames=0):
        '''Calculates distances between pairs of atom groups (dist) for each set of selections "dists" using MDAnalysis D.distance_array method.
        Stores the distances in results_dir. Returns the distances as dataframes.
	    Uses the NAC_frames files obtained in extract_frames(). 
	    NOTE: Can only work for multiple trajectories if the number of atoms is the same (--noWater)'''
         
        import MDAnalysis as mda
        import MDAnalysis.analysis.distances as D
        
        
        dist={}
        
        for name, system in self.systems.items():
            
            print(name)
            
            
            #distances_df=pd.DataFrame() #Dataframe that stores all the distances for all cuttoffs
            
            for iterable, xtc_pdb in system.items():
                
                trj, top=xtc_pdb
                
                u=mda.Universe(top, trj)
                
                print(f'\tIterable: {iterable}')
                
                distances={} # This is the dictionary of d distances.
                
                for idx, dist in enumerate(dists, 1): #iterate through the list of dists. For each dist, define atomgroup1 (sel1) and atomgroup2 (sel2)
                        
                    sel1, sel2 =u.select_atoms(dist[0]), u.select_atoms(dist[1])
				
                    distance=np.around([D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory], decimals=3)
				
                    plt.hist(distance.flat, bins=np.arange(0,30,1))
                    plt.show()
                    
                    
                    distances[idx]=distance
             
                
    @classmethod
    def plot(cls, df_to_plot, level='l3'):
            
            print(df)

            levels=df.columns.levels[:-1] #Exclude last, the values of states
            
            print(levels)
    
            for iterable in levels[2]:
                
                df_it=df.loc[:,df.columns.get_level_values(level) == iterable]
                
                
                print(df_it.columns.unique(level=2).tolist())
                df_it.plot(#kind='line',
                               subplots=True,
                               sharey=True,
                               title=iterable,
                               figsize=(6,8),
                               legend=False,
                               sort_columns=True)
            
                plt.savefig(f'{cls.results}/discretized_{name}_{iterable}.png', dpi=300)
                plt.show()
    
        
        
    
class Visualize():
    """
    Class to visualize trajectories
    """
    
    def __init__(self, results):
        self.results=results

    def viewer(self, base_name='superposed', stride=1):
        
        import nglview
        
        c=input('Value of parameter: ')
        it=input('Value of iterable: ')
    
        base_name=f'{self.results}/{base_name}'
        print(f'{base_name}_{c}-it{it}-s{stride}.xtc')

        view=nglview.show_file(f'{base_name}_{c}-it{it}-s{stride}.pdb')
        view.add_component(f'{base_name}_{c}-it{it}-s{stride}-Molar.dx')
        view.add_component(f'{base_name}_{c}-it{it}-s{stride}.xtc')
        view.clear_representations()
        view.add_representation('cartoon')
        view.add_representation('licorice', selection='not hydrogen')

    
        return view
    


class Test:
    
    def __init__(self, input_file='input.xml'):
        self.input = input_file
    
    def test(self):
        from lxml import etree as et
        #import xml.etree.ElementTree as et
        parser = et.XMLParser(recover=True)

        workdir=Path(os.getcwd())
        name=f'{workdir}\input.xml'
        #TODO: fix this for unix
        print(name)
        tree = et.parse(name, parser=parser)        
        root=tree.getroot()
        
        print(root.tag)
        
        for child in root:
            print(child.tag, child.attrib)
            
        #print([elem.tag for elem in root.iter()])
        
        for system in root.iter('system'):
            print(system.attrib)
            
        for description in root.iter('description'):
            print(description.text)
    
        for system in root.findall("./kind/interaction/system/[protein='CalB']"):
            print(system.attrib)
    
        return root
        
