

"""


The core python classes of the package.
Classes are implemented in jupyter notebooks or in offline runs (optional).


Created on Fri Jul  3 11:30:16 2020
@author: hca
"""
   
#from Decorators import MLflow, MLflow_draft

#SFv2
try:

    import tools
    import tools_plots
    
    from MSM import MSM
    import Discretize
    import Featurize
    from tools import Tools


      
#    import pyemma
except:
    pass

import Trajectory
import os
import simtk.unit as unit


# =============================================================================
# @MLflow_draft
# def test():
#     
#     
#     import numpy as np
#     
#     x=np.random.rand(1,10)
# 
#     y=np.random.rand(2,3)
# 
#     return x, y
# =============================================================================


class Project:
    """
    
    Base class to define a project.
    
    """

    
    def __init__(self, 
                 workdir,  
                 title="Untitled",
                 hierarchy=("protein", "ligand", "parameter"), 
                 parameter=None, 
                 replicas=1, 
                 protein=None, 
                 ligand=None, 
                 timestep=1*unit.picoseconds, 
                 topology='system.pdb',
                 initial_replica=1,
                 results='results'):
        
        """
    
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
        results : path (optional)
            The path where results will be stored.
        
        
        Returns
        -------
        self: object
            The project instance. 
    
        """
        
        self.workdir=workdir
        self.title=title
        self.hierarchy=hierarchy
        self.parameter=parameter
        self.parameter_dict=tools.Functions.setScalar(self.parameter)
        
        self.replicas=int(replicas)
        self.initial_replica=initial_replica
        
        self.protein=protein
        self.ligand=ligand
        self.timestep=timestep
        
        self.results=f'{self.workdir}/{results}'
        self.def_input_struct=f'{self.workdir}/inputs/structures'
        self.def_input_ff=f'{self.workdir}/inputs/forcefields'
        self.input_topology=f'{self.def_input_struct}/{topology}'
        
        tools.Functions.fileHandler([self.results, self.def_input_struct, self.def_input_ff])
        
    
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
               
    
    def setSystems(self, replica_name='replicate') -> dict:
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
        replicas=[str(i) for i in range(self.initial_replica,self.replicas+1)]
        elements.append(replicas)
        systems=list(itertools.product(*elements)) #generate all possible combinations of elements: systems.
        systems_obj=[] 
        

        
        #Retrieve the systems from class System.
        for system in systems:
            
            #TODO: Flexible parameter and ligand on/off.
            protein_=system[self.hierarchy.index('protein')] # system[0],
            if self.ligand != None:
                ligand_=system[self.hierarchy.index('ligand')] #system[1],
            else:
                ligand_= self.ligand
            parameter_=system[self.hierarchy.index('parameter')] #system[2],

            systems_obj.append(System(system, 
                                      self.workdir, 
                                      self.input_topology,
                                      self.results,
                                      protein=protein_, 
                                      ligand=ligand_,
                                      parameter=parameter_,
                                      parameter_dict=self.parameter_dict,
                                      replicate=system[-1],
                                      timestep=self.timestep,
                                      replica_name=replica_name))
        
        systems={}
        for system in systems_obj:
            systems[system.name]=system
            #print(f'{system.name} \n\t{system}\n\t{system.project_results}')
            tools.Functions.fileHandler([system.name_folder]) #Update the path if fileHandler generates a new folder
            
        self.systems=systems

        return systems
        
        
class System(Project):
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
    

    linker='-'
    
    
    def __init__(self, 
                 system_ID,
                 workdir,
                 input_topology,
                 results,
                 timestep=1*unit.picoseconds,
                 protein=None,
                 ligand=None,
                 parameter='parameter',
                 parameter_dict={'parameter':0},
                 replicate=1,
                 topology='system.pdb',
                 replica_name='replicate'):

        
        #TODO: class inheritance to fix it.
        self.system_ID = system_ID
        self.protein=protein
        self.ligand=ligand
        self.parameter=parameter       
        self.scalar=parameter_dict[parameter]
        self.timestep=timestep
        self.input_topology=input_topology
        self.replicate=replicate
        self.workdir=workdir
        self.results=results
        self.replica_name=replica_name
        self.name=System.linker.join(system_ID)
        self.name_folder=os.path.abspath(f'{self.workdir}/{System.linker.join(self.system_ID[:-1])}/{self.replica_name}{self.system_ID[-1]}')
        self.project_results = os.path.abspath(f'{self.results}/{System.linker.join(self.system_ID[:-1])}')
        self.results_folder=os.path.abspath(f'{self.name_folder}/results')
        
        self.trajectory=Trajectory.Trajectory.findTrajectory(self.name_folder)
        self.topology=Trajectory.Trajectory.findTopology(self.name_folder, topology, input_topology=input_topology)
        
        self.structures = {}
        self.data={}
        self.features={}
        self.traj = None
        
        tools.Functions.fileHandler([self.results_folder])

        print(f'System {self.name} with parameter {self.scalar} defined')
        
        if self.topology == None:
             
            print('top', self.name)
        if self.trajectory == None:
            print('traj', self.name)
                       
    
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

