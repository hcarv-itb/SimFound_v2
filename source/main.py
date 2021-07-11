

"""


The core python classes of the package.
Classes are implemented in jupyter notebooks or in offline runs (optional).


Created on Fri Jul  3 11:30:16 2020
@author: hca
"""
   
from Decorators import MLflow, MLflow_draft

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


@MLflow_draft
def test():
    
    
    import numpy as np
    
    x=np.random.rand(1,10)

    y=np.random.rand(2,3)

    return x, y


class Project:
    """
    
    Base class to define a project.
    
    """

    
    def __init__(self, workdir, 
                 results, 
                 title="Untitled",
                 hierarchy=("protein", "ligand", "parameter"), 
                 parameter=None, 
                 replicas=1, 
                 protein='protein', 
                 ligand='ligand', 
                 timestep=1*unit.picoseconds, 
                 topology='system.pdb',
                 initial_replica=1):
        
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
        results : path
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
        self.replicas=int(replicas)
        self.initial_replica=initial_replica
        self.protein=protein
        self.ligand=ligand
        self.timestep=timestep
        self.results=results
        #TODO: make confirmation tasks
        
        self.def_input_struct=f'{self.workdir}/inputs/structures'
        self.def_input_ff=f'{self.workdir}/inputs/forcefields'
        self.input_topology=f'{self.def_input_struct}/{topology}'
        
        self.parameter_dict={}
        
        for idx, p in enumerate(self.parameter):

            try:
                scalar=float(str(p).split('K')[0])*unit.kelvin

                #self.parameter_scalar.append(scalar)
                #parameter_label.append(str(scalar).replace(" ",""))
                
                print(f'Converted parameter "temperature" (in K) into scalar: {scalar}')
            except ValueError:
                
                try:
                    scalar=float(str(p).split('M')[0])*unit.molar
                except:
                    scalar=float(str(p).split('mM')[0])/1000*unit.molar
            except:
                scalar=idx
                self.parameter_scalar.append(scalar)
                print(f'Converted unidentified parameter into scalar: {scalar}')
                
                

            self.parameter_dict[p]=scalar
    
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
        #TODO: allow init replica > 1
        elements.append(replicas)
        
        systems=list(itertools.product(*elements)) #generate all possible combinations of elements: systems.
        systems_obj=[] #Retrieve the systems from class System.
        
        for system in systems:
            
            #TODO: This is hardcoding protein, ligand and parameter. Flexible is REQUIRED!!!
            systems_obj.append(System(system, 
                                      self.workdir, 
                                      self.input_topology,
                                      protein=system[0],
                                      ligand=system[1],
                                      parameter=system[2],
                                      parameter_dict=self.parameter_dict,
                                      replicate=system[3],
                                      timestep=self.timestep,
                                      replica_name=replica_name))
            
# =============================================================================
#             systems_obj.append(System(system, 
#                                       self.timestep, 
#                                       topology=self.topology_input, 
#                                       trajectory=self.trajectory_output,
#                                       workdir=self.workdir))
# =============================================================================
        
        systems={}
        for system in systems_obj:
            systems[system.name]=system
            #print(f'{system.name} \n\t{system}\n\t{system.name_folder}\n\t{system.path}')
            
        for k, v in systems.items():
            v.name_folder=tools.Functions.fileHandler([v.name_folder]) #Update the path if fileHandler generates a new folder
            
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
                 system,
                 workdir,
                 input_topology,
                 timestep=1*unit.picoseconds,
                 protein='protein',
                 ligand='ligand',
                 parameter='parameter',
                 parameter_dict={'parameter':0},
                 replicate=1,
                 topology='system.pdb',
                 replica_name='replicate'):

        
        #TODO: class inheritance to fix it.
        self.protein=protein
        self.ligand=ligand
        self.parameter=parameter        
        self.scalar=parameter_dict[parameter]
        self.replicate=replicate
        self.system=system
        self.workdir=workdir
        self.input_topology=input_topology
        self.timestep=timestep
        self.replica_name=replica_name
        self.name=System.linker.join(self.system)
        self.name_folder=os.path.abspath(f'{System.linker.join(self.system[:-1])}/{self.replica_name}{self.system[-1]}')        
        self.results_folder=os.path.abspath(f'{self.name_folder}/results')
        self.trajectory=Trajectory.Trajectory.findTrajectory(self.name_folder)
        self.topology=Trajectory.Trajectory.findTopology(self.name_folder, topology)
        self.data={}
        self.features={}
        
        if not os.path.exists(self.results_folder):
            os.makedirs(os.path.abspath(self.results_folder))

        print(f'System defined: {self.system}, ID: {self.scalar}')
    
                
    @classmethod
    def plot(cls, input_df, level='l3'):
            
            print(input_df)

            levels=input_df.columns.levels[:-1] #Exclude last, the values of states
            
            print(levels)
    
            for iterable in levels[2]:
                
                df_it=input_df.loc[:,input_df.columns.get_level_values(level) == iterable]
                
                
                print(df_it.columns.unique(level=2).tolist())
                df_it.plot(#kind='line',
                               subplots=True,
                               sharey=True,
                               title=iterable,
                               figsize=(6,8),
                               legend=False,
                               sort_columns=True)
            
                plt.savefig(f'{cls.results}/discretized_{cls.name}_{iterable}.png', dpi=300)
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

