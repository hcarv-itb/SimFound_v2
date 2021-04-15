

"""


The core python classes of the package.
Classes are implemented in jupyter notebooks or in offline runs (optional).


Created on Fri Jul  3 11:30:16 2020
@author: hca
"""


#import tools
#import tools_plots
#from Trajectory import Trajectory
#from MSM import MSM
#from Discretize import Discretize
#from Featurize import Featurize



#try:
#    import pyemma
#except:
#    pass


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
            systems_obj.append(System(system, self.timestep, workdir=self.workdir))
        
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
    
    
    def __init__(self, system, timestep, workdir=os.getcwd(), replica_name='sim', linker='-'):
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
        self.timestep=timestep
    
    
# =============================================================================
# 
# class Features:
#     """Base class to create a *features* object. Different featurization schemes can be coded.
#     
#     Parameters
#     ----------
#     project: object
#         Object instance of the class "Project".
#         
#     Returns
#     -------
#     
#     feature: object
#         Feature instance
#     
#     """
#     
#     def __init__(self, systems, results):
#         self.systems=systems
#         self.results=results
#         self.features={}
#         #self.timestep=int(f'{self.systems.timestep}')
#  
#         
#     def dist(self, dists, stride=1, skip_frames=0):
#         """
#         Calculates distances between pairs of atom groups (dist) for each set of selections "dists" 
#         using MDAnalysis D.distance_array method.
#         Stores the distances in results_dir. Returns the distances as dataframes.
# 	    Uses the NAC_frames files obtained in extract_frames(). 
# 	    NOTE: Can only work for multiple trajectories if the number of atoms is the same (--noWater)
# 
#         Parameters
#         ----------
#         dists : TYPE
#             DESCRIPTION.
#         stride : TYPE, optional
#             DESCRIPTION. The default is 1.
#         skip_frames : TYPE, optional
#             DESCRIPTION. The default is 0.
# 
#         Returns
#         -------
#         None.
# 
#         """
#         
#         
#         
#         ''''''
#          
#         import MDAnalysis as mda
#         import MDAnalysis.analysis.distances as D
#         
#         
#         dist={}
#         
#         for name, system in self.systems.items():
#             
#             print(name)
#             
#             
#             #distances_df=pd.DataFrame() #Dataframe that stores all the distances for all cuttoffs
#             
#             for iterable, xtc_pdb in system.items():
#                 
#                 trj, top=xtc_pdb
#                 
#                 u=mda.Universe(top, trj)
#                 
#                 print(f'\tIterable: {iterable}')
#                 
#                 distances={} # This is the dictionary of d distances.
#                 
#                 for idx, dist in enumerate(dists, 1): #iterate through the list of dists. For each dist, define atomgroup1 (sel1) and atomgroup2 (sel2)
#                         
#                     sel1, sel2 =u.select_atoms(dist[0]), u.select_atoms(dist[1])
# 				
#                     distance=np.around([D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory], decimals=3)
# 				
#                     plt.hist(distance.flat, bins=np.arange(0,30,1))
#                     plt.show()
#                     
#                     
#                     distances[idx]=distance
#              
#         
#     def nac(self,  
#                 dists,
#                 start=0,
#                 stop=-1,
#                 stride=1,
#                 use_precalc=False,
#                 processes=1):
#         """
#         Calculates the d_NAC values from an arbitrary number of distance "sels" pairs. 
#         Increase "processes" to speed up calculations.
#         Option "use_precalc" retrieves pre-calculated d_NAC data.
#         
# 
#         Parameters
#         ----------
# 
# 
#         sels : list of tuples
#             A list of distance tuples of the kind [(ref1-sel1), ..., (refN-selN)].
#         start : int, optional
#             The starting frame used to calculate. The default is 0.
#         stop : int, optional
#             The last frame used to calculate. The default is -1.
#         stride : int, optional
#             Read every nth frame. The default is 1.
#         use_precalc : bool, optional
#             Whether to use or not a pre-calculated file. The default is False.
#         processes : int, optional
#             The number of cores to execute task. The default is 1.
# 
#         Returns
#         -------
#         nac_df : dataframe
#             A dataframe containing all the d_NAC values across the list of iterables*replicas*pairs.
# 
#         """
#         
#         import psutil
#         import time
#         from functools import partial
#         from multiprocessing import Pool
#         
#         self.features='dNAC'
#                         
#         traj_specs=(dists, start, stop, stride, use_precalc)      
#                 
#             
#         # Decide how many proccesses will be created
#             
#         if processes <=0:
#             num_cpus = psutil.cpu_count(logical=False)
#         else:
#             num_cpus = processes
#             
#         print(f'Working on {num_cpus} logical cores.')
#     
#         # Create the pool
#         process_pool = Pool(processes=num_cpus)
#         start = time.time()
#            
#         # Start processes in the pool
#         systems=[]
#         
#         for name, system in self.systems.items(): #system cannot be sent as pickle for multiproc, has to be list
#             
#             trajectory=system.trajectory
#             topology=system.topology
#             results_folder=system.results_folder
#             timestep=system.timestep
#             
#             systems.append((trajectory, topology, results_folder, timestep, name))
#         
#         #setup multiprocessing
#         calc=partial(Features.nac_single, traj_specs=traj_specs)
#         df_data = process_pool.map(calc, systems)
#         
#         process_pool.close()
#         process_pool.join()
#             
#         print(df_data)
#             
#         # concat dataframes from multiprocessing into one dataframe
#         nac_df=pd.DataFrame()
#         for df_d in df_data:
#             nac_df = pd.concat([nac_df,  df_d[0]], axis=1)
#                                
#         print(nac_df)
#         return nac_df
#  
#          #self.shells=shells
#          #shells_name='_'.join(str(self.shells))
#      
# 
#     @staticmethod        
#     def nac_single(system_t, traj_specs):
#         """
#         Retrieves the d_NAC array and corresponding dataframe.
#         Recieves instructions from multiprocessing calls.
#         Makes calls to "nac_calculation" or "nac_precalculated".
#         Operation is adjusted for multiprocessing calls, hence the requirement for tuple manipulation.
# 
#         Parameters
#         ----------
#         system_tuple : tuple
#             A tuple containing system information (trajectory, topology, results_folder, timestep, name).
#         traj_specs : tuple, optional
#             A tuple containing trajectory information (dists, start, stop, stride, use_precalc). The default is None.
# 
#         Returns
#         -------
#         nac_df_system : dataframe
#             The d_NAC dataframe of the system
# 
#         """   
#         
#         (trajectory, topology, results_folder, timestep, name)=system_t
#         (dists, start, stop, stride, use_precalc)=traj_specs
#             #print(sels, start, stop, stride, use_precalc)
#         
# 
#         if use_precalc:
#             
#             data, nac_df_system=Features.nac_precalculated(results_folder, name=None)
#                 
#         else:
#                 
#             data, nac_df_system=Features.nac_calculation(topology, trajectory, name, timestep, dists, 
#                                                          start, stop, stride, results_folder=results_folder)
#         
#         #TODO update system feature with data. Has to provide 2 outs to Pool.
#         
#         return nac_df_system
#    
#     
#     @staticmethod
#     def nac_calculation(topology,
#                         trajectory,
#                         name,
#                         timestep,
#                         dists,
#                         start=0,
#                         stop=10,
#                         stride=1,
#                         results_folder=os.getcwd()):
#         """
#         The workhorse function for nac_calculation. 
# 
#         Parameters
#         ----------
#         topology : TYPE
#             DESCRIPTION.
#         trajectory : TYPE
#             DESCRIPTION.
#         name : TYPE
#             DESCRIPTION.
#         timestep : TYPE
#             DESCRIPTION.
#         dists : TYPE
#             DESCRIPTION.
#         start : TYPE, optional
#             DESCRIPTION. The default is 0.
#         stop : TYPE, optional
#             DESCRIPTION. The default is 10.
#         stride : TYPE, optional
#             DESCRIPTION. The default is 1.
#         results_folder : TYPE, optional
#             DESCRIPTION. The default is os.getcwd().
# 
#         Returns
#         -------
#         nac_file : TYPE
#             DESCRIPTION.
#         nac_df_system : TYPE
#             DESCRIPTION.
#         
#         TODO: make call to ask for start stop, etc.
#         
#         """
#         
#         import MDAnalysis as mda
#         import MDAnalysis.analysis.distances as D
# 
#         print(f'Calculating {name} \n')
# 
#         indexes=[[n] for n in name.split('-')] 
#         names=[f'l{i}' for i in range(1, len(indexes)+2)] # +2 to account for number of molecules 
# 
#        
#         nac_file=f'{results_folder}/dNAC_{len(dists)}-i{start}-o{stop}-s{stride}-{timestep}ps.npy'
#         
#         if not os.path.exists(nac_file):
#                     
#             dists=[] #list of distance arrays to be populated
#             
#             #iterate through the list of sels. 
#             #For each sel, define atomgroup1 (sel1) and atomgroup2 (sel2)
#             for idx, dist in enumerate(dists, 1): 
#                 dist_file=f'{results_folder}/distance{idx}-i{start}-o{stop}-s{stride}-{timestep}.npy'
#             
#                 if not os.path.exists(dist_file):
#                 
#                     print(f'\tDistance {idx} not found for {name}. Reading trajectory...')  
#                     u=mda.Universe(topology, trajectory)
#                     sel1, sel2 =u.select_atoms(dist[0]).positions, u.select_atoms(dist[1]).positions
#                     
#                     print(f'\t\tsel1: {dist[0]} ({len(sel1)})\n\t\tsel2: {dist[1]} ({len(sel2)})\n\t\tCalculating...')        
#                     
#                     dists_=np.around(
#                                 [D.distance_array(sel1, sel2, box=u.dimensions) for ts in u.trajectory[start:stop:stride]],
#                                 decimals=3) 
#                                 
#                     np.save(dist_file, dists_)
#                         
#                 else:
#                     dists_=np.load(dist_file)
#                     print(f'\tDistance {idx} found for {name}. Shape: {np.shape(dists_)}')
#                     
#                 dists.append(dists_)
#                         
#             #NAC calculation
#             nac_array=np.around(np.sqrt(np.power(np.asarray(dists), 2).sum(axis=0)/len(dists)), 3) #SQRT(SUM(di^2)/#i)  
#             np.save(nac_file, nac_array)
#             
#         else:
#             print(f'dNAC file for {name} found.') 
#             nac_array=np.load(nac_file)
#         
#         
#         #TODO: Check behaviour for ref > 1
# 
#         
#         frames, sel, ref=np.shape(nac_array)
#         nac_array=nac_array.reshape(frames, ref*sel)
#         
#         indexes.append([e for e in np.arange(1,sel+1)])
#         column_index=pd.MultiIndex.from_product(indexes, names=names)        
# 
#         nac_df_system=pd.DataFrame(nac_array, index=np.arange(start, stop, stride), columns=column_index)
#         #nac_df_system=pd.DataFrame(np.round(nac_array.ravel(), decimals=1))
#         
#         #print(nac_df_system)    
#             
#         return nac_file, nac_df_system
#     
#     
#     
#     @staticmethod
#     def nac_precalculated(results_folder, name=None):
#         
#         import glob
#    
#         #TODO: Make it less specific by defining start, stop, etc. using name specs.
#         data=str(glob.glob(f'{results_folder}NAC2*.npy')[0]) 
#                 
#         print(f'Using pre-calculated file: {data}')
#                 
#         raw_data=np.load(data)
#                
#         #Reconstruct a DataFrame of stored feature into a Dataframe
#         #TODO: Check behaviour for ref > 1.
#                     
#         frames, ref, sel=np.shape(raw_data)
#         raw_reshape=raw_data.reshape(frames, ref*sel)
#                             
#         if ref == 1:
#             nac_df_system=pd.DataFrame(raw_reshape) 
#             nac_df_system.columns=nac_df_system.columns + 1
#         else:
#             nac_df_system=pd.DataFrame()
#             split=np.split(raw_reshape, ref, axis=1)
#             for ref in split:
#                 df_ref=pd.DataFrame(ref)
#                 df_ref.columns=df_ref.columns + 1
#                 nac_df_system=pd.concat([nac_df_system, df_ref], axis=0)
#     
#         return data, nac_df_system
# =============================================================================

                
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

