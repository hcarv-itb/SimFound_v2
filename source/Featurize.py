# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:34:14 2021

@author: hcarv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import MDAnalysis as mda
import MDAnalysis.analysis.distances as D

#import tools

class Featurize:
    """
    Base class to create a *features* object. Different featurization schemes can be coded.
    
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
        self.features={}
        self.levels=self.systems.getlevels()
        #self.timestep=int(f'{self.systems.timestep}')
 
        
    def dist(self, distances, start=0, stop=-1, stride=1)->dict:
        
        """
        Calculates distances between pairs of atom groups (dist) for each set of selections "dists" 
        using MDAnalysis D.distance_array method.
        Stores the distances in results_dir. Returns the distances as dataframes.
	    Uses the NAC_frames files obtained in extract_frames(). 
	    NOTE: Can only work for multiple trajectories if the number of atoms is the same (--noWater)
        
        

        Parameters
        ----------
        distances : list
            DESCRIPTION.
        start : int, optional
            DESCRIPTION. The default is 0.
        stop : int, optional
            DESCRIPTION. The default is -1.
        stride : int, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        dict
            A dictionary of distance.

        """
        
        dist={}
        
        for name, system in self.systems.items():
            
            print(name)
            
            timestep=system.timestep
            results_folder=system.results_folder
            
            dist[name]=Featurize.distance_calculation(system.topology, system.trajectory, 
                                                      distances, start, stop, results_folder, 
                                                      name, stride, timestep)

        return dist
             
        
    def nac(self,  
                distances,
                feature_name='feature',
                start=0,
                stop=-1,
                stride=1,
                n_cores=-1):
        """
        Calculates the d_NAC values from an arbitrary number of distance "sels" pairs. 
        Increase "n_cores" to speed up calculations.
        Retrieves the d_NAC array and corresponding dataframe.
        Recieves instructions from multiprocessing calls.
        Makes calls to "nac_calculation" method.
        Operation is adjusted for multiprocessing calls, hence the requirement for tuple manipulation.
        
        
        Parameters
        ----------


        distances : list of tuples
            A list of distance tuples of the kind [(ref1-sel1), ..., (refN-selN)].
        feature_name : string
            The name given to the featurized object. The default is 'feature'.
        start : int, optional
            The starting frame used to calculate. The default is 0.
        stop : int, optional
            The last frame used to calculate. The default is -1.
        stride : int, optional
            Read every nth frame. The default is 1.
        n_cores : int, optional
            The number of cores to execute task. Set to -1 for all cores. The default is 2.

        Returns
        -------
        dataframe
            A dataframe containing all the d_NAC values across the list of iterables*replicas*pairs.

        """
           
        # Define system specifications and measurement instructions
        systems_specs=[]
        
        for name, system in self.systems.items(): #system cannot be sent as pickle for multiproc, has to be list
            
            trajectory=system.trajectory
            topology=system.topology
            results_folder=system.results_folder
            timestep=system.timestep
            
            systems_specs.append((trajectory, topology, results_folder, name))
            
        measurements=(distances, start, stop, timestep, stride)
        #print(system_specs)
        #print(measurements)
        
        
        data_t=tools.Tasks.parallel_task(Featurize.nac_calculation, 
                                             systems_specs, 
                                             measurements, 
                                             n_cores=n_cores)
        
        # concat dataframes from multiprocessing into one dataframe
        #Update state of system and features with the feature df.
        
        nac_df=pd.DataFrame()
        
        for data_df, system in zip(data_t, self.systems.values()):
          
            system.data[feature_name]=data_df[0]
            system.features[feature_name]=data_df[1]
            
            nac_df=pd.concat([nac_df,  data_df[1]], axis=1)
                                       
        #print(nac_df)
        
        self.features[feature_name]=nac_df
        
        return nac_df
 
    
    
    @staticmethod
    def nac_calculation(system_specs, specs=(), results_folder=os.getcwd()):
        """
        The workhorse function for nac_calculation. 
        Retrieves the d_NAC array and corresponding dataframe.
        Recieves instructions from multiprocessing calls.
        Makes calls to "nac_calculation" or "nac_precalculated".
        Operation is adjusted for multiprocessing calls, hence the requirement for tuple manipulation.
        

        Parameters
        ----------
        system_specs : tuple
            A tuple containing system information (trajectory, topology, results_folder, name).
        measurements : tuple, optional
            A tuple containing measurement information (distances, start, stop, timestep, stride).

        Returns
        -------
        nac_df_system : dataframe
            The d_NAC dataframe of the system

        Returns
        -------
        nac_file : TYPE
            DESCRIPTION.
        nac_df_system : TYPE
            DESCRIPTION.
        
        
        """

        (trajectory, topology, results_folder, name)=system_specs
        (distances, start, stop, timestep, stride)=specs  

        #print(f'Working on {name}')

        indexes=[[n] for n in name.split('-')] 
        names=[f'l{i}' for i in range(1, len(indexes)+2)] # +2 to account for number of molecules
        #TODO: Make function call to get the real names of the levels. Current l1, l2, l3, etc.

       
        nac_file=f'{results_folder}/dNAC_{len(distances)}-i{start}-o{stop}-s{stride}-{timestep}ps.npy'
        
        if not os.path.exists(nac_file):
             
            dists=Featurize.distance_calculation(system_specs, specs)

            #print(f'This is dists  {np.shape(dists[0])}: \n {dists[0]}')
            #print(f'This is power: \n{np.power(dists[0], 2)}')

            print(f'Calculating d_NAC of {name}')
            #NAC calculation
            
            #SQRT(SUM(di^2)/#i)
            nac_array=np.around(np.sqrt(np.power(dists, 2).sum(axis=0)/len(dists)), 3) #sum(axis=0)
            
            #SQRT(MEAN(di^2))
            #nac_array=np.around(np.sqrt(np.mean(np.power(dists))), 3)
            np.save(nac_file, nac_array)

            
        else:
            #print(f'dNAC file for {name} found.') 
            nac_array=np.load(nac_file)

        
        #TODO: Check behaviour for ref > 1
        #TODO: switch ref and sel for full.
        frames, sel, ref=np.shape(nac_array)
        
        print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}\n\t{np.min(nac_array)}, {np.max(nac_array)}')

        try:
            nac_array=nac_array.reshape(frames, ref*sel)
        except:
            
            print(f'\tError found in {name}: the dNAC shape is {np.shape(nac_array)}')
            
            nac_array=nac_array.reshape(frames-1, ref*sel)
        

        pairs=np.arange(1,sel+1)
        #print(pairs)
        indexes.append(pairs)
        #print(indexes)
        column_index=pd.MultiIndex.from_product(indexes, names=names)
        #print(column_index)
        nac_df_system=pd.DataFrame(nac_array, columns=column_index) #index=np.arange(start, stop, stride)
        
        #print(nac_df_system)    
        
# =============================================================================
#         plt.hist(nac_array.flat, bins=np.arange(0,151))
#         plt.title(f'{name}')
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.show()
#         plt.clf()
# =============================================================================
            
        return (nac_file, nac_df_system)
    
    

    @staticmethod
    def distance_calculation(system_specs, measurements):
        """
        The workhorse function to calculate distances. Uses MDAnalysis.
        Retrieves infromation from two tuples passed by callers contained in "Parameters".
        

        Parameters
        ---------
        
        topology : TYPE
            DESCRIPTION.
        trajectory : TYPE
            DESCRIPTION.
        distances : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        stop : TYPE
            DESCRIPTION.
        results_folder : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        stride : TYPE
            DESCRIPTION.
        timestep : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
 
        import time   
 
        (trajectory, topology, results_folder, name)=system_specs
        (distances, start, stop, timestep, stride)=measurements    
 
    
        dists=[] #list of distance arrays to be populated
            
        #iterate through the list of sels. 
        #For each sel, define atomgroup1 (sel1) and atomgroup2 (sel2)
#        fig,axes=plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        for idx, dist in enumerate(distances, 1): 
            dist_file=f'{results_folder}/distance{idx}-i{start}-o{stop}-s{stride}-{timestep}ps.npy'
            clock_s=time.time()
            if not os.path.exists(dist_file):
                
                
                                
                print(f'Distance {idx} not found for {name}. Reading trajectory...')  
                u=mda.Universe(topology, trajectory)
                
                #print(u.dimensions)
                ref, sel =u.select_atoms(dist[0]).positions, u.select_atoms(dist[1]).positions
                    
                print(f'\t\tref: {u.select_atoms(dist[0])[0]} x {len(ref)}\n\t\tsel: {u.select_atoms(dist[1])[0]} x {len(sel)}')
                print(f'\tCalculating distance {idx} of {name}')          
                dists_=np.around(
                                [D.distance_array(ref, 
                                                  sel, 
                                                  box=u.dimensions, 
                                                  result=np.empty((len(ref), len(sel)))) for ts in u.trajectory[start:stop:stride]],
                                decimals=3) 
                                
                np.save(dist_file, dists_)
                        
            else:
                #print(f'\tDistance {idx} found for {name}')
                dists_=np.load(dist_file)
            clock_e=time.time()
            
            #print(f'distance {idx} of {name}: {np.shape(dists_)}')
            dists.append(dists_)
        
        for idx, dists_ in enumerate(dists, 1):
            print(f'\tDistance {idx}: {np.shape(dists_)}, {np.min(dists_), np.max(dists_)} ({np.round(clock_e-clock_s, decimals=2)} s)')    
      
        
#       plt.show()

        #print(np.min(np.asarray(dists)), np.max(np.asarray(dists)))
        return np.asarray(dists)
        

            
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
                