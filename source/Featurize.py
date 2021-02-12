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

class Featurize:
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
        self.features={}
        #self.timestep=int(f'{self.systems.timestep}')
 
        
    def dist(self, dists, stride=1, skip_frames=0):
        """
        Calculates distances between pairs of atom groups (dist) for each set of selections "dists" 
        using MDAnalysis D.distance_array method.
        Stores the distances in results_dir. Returns the distances as dataframes.
	    Uses the NAC_frames files obtained in extract_frames(). 
	    NOTE: Can only work for multiple trajectories if the number of atoms is the same (--noWater)

        Parameters
        ----------
        dists : TYPE
            DESCRIPTION.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.
        skip_frames : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
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
        
        return dist
             
        
    def nac(self,  
                dists,
                start=0,
                stop=-1,
                stride=1,
                use_precalc=False,
                processes=2):
        """
        Calculates the d_NAC values from an arbitrary number of distance "sels" pairs. 
        Increase "processes" to speed up calculations.
        Option "use_precalc" retrieves pre-calculated d_NAC data.
        

        Parameters
        ----------


        sels : list of tuples
            A list of distance tuples of the kind [(ref1-sel1), ..., (refN-selN)].
        start : int, optional
            The starting frame used to calculate. The default is 0.
        stop : int, optional
            The last frame used to calculate. The default is -1.
        stride : int, optional
            Read every nth frame. The default is 1.
        use_precalc : bool, optional
            Whether to use or not a pre-calculated file. The default is False.
        processes : int, optional
            The number of cores to execute task. Set to -1 for all cores. The default is 2.

        Returns
        -------
        nac_df : dataframe
            A dataframe containing all the d_NAC values across the list of iterables*replicas*pairs.

        """
        
        import psutil
        from functools import partial
        from multiprocessing import Pool
        
        self.features='dNAC'
                        
        traj_specs=(dists, start, stop, stride, use_precalc)      
                
            
        # Decide how many proccesses will be created
            
        if processes <=0:
            num_cpus = psutil.cpu_count(logical=False)
        else:
            num_cpus = processes
            
        print(f'Working on {num_cpus} logical cores.')
    
        # Create the pool
        process_pool = Pool(processes=num_cpus)
           
        # Start processes in the pool
        systems=[]
        
        for name, system in self.systems.items(): #system cannot be sent as pickle for multiproc, has to be list
            
            trajectory=system.trajectory
            topology=system.topology
            results_folder=system.results_folder
            timestep=system.timestep
            
            systems.append((trajectory, topology, results_folder, timestep, name))
        
        #setup multiprocessing
        calc=partial(Featurize.job, traj_specs=traj_specs)

        df_data=list(process_pool.map(calc, systems))
#        print(df_data)
        
        process_pool.close()
        process_pool.join()
         

            
        # concat dataframes from multiprocessing into one dataframe
        nac_df=pd.DataFrame()
        for df_d in df_data:
            nac_df = pd.concat([nac_df,  df_d], axis=1)
                               
        print(nac_df)
        return nac_df
 
         #self.shells=shells
         #shells_name='_'.join(str(self.shells))
     

    @staticmethod        
    def job(system_t, traj_specs):
        """
        Retrieves the d_NAC array and corresponding dataframe.
        Recieves instructions from multiprocessing calls.
        Makes calls to "nac_calculation" or "nac_precalculated".
        Operation is adjusted for multiprocessing calls, hence the requirement for tuple manipulation.

        Parameters
        ----------
        system_tuple : tuple
            A tuple containing system information (trajectory, topology, results_folder, timestep, name).
        traj_specs : tuple, optional
            A tuple containing trajectory information (dists, start, stop, stride, use_precalc). The default is None.

        Returns
        -------
        nac_df_system : dataframe
            The d_NAC dataframe of the system

        """   
        
        (trajectory, topology, results_folder, timestep, name)=system_t
        (dists, start, stop, stride, use_precalc)=traj_specs
            #print(sels, start, stop, stride, use_precalc)
        
        if use_precalc:
            
            data, nac_df_system=Featurize.nac_precalculated(results_folder, name=None)
                
        else:
                
            data, nac_df_system=Featurize.nac_calculation(topology, trajectory, name, timestep, dists, 
                                                         start, stop, stride, results_folder=results_folder)
        
        #TODO update system feature with data. Has to provide 2 outs to Pool.
        print(nac_df_system)
        
        return nac_df_system
   
    
    @staticmethod
    def nac_calculation(topology,
                        trajectory,
                        name,
                        timestep,
                        distances,
                        start=0,
                        stop=10,
                        stride=1,
                        results_folder=os.getcwd()):
        """
        The workhorse function for nac_calculation. 

        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        trajectory : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        timestep : TYPE
            DESCRIPTION.
        distances : TYPE
            DESCRIPTION.
        start : TYPE, optional
            DESCRIPTION. The default is 0.
        stop : TYPE, optional
            DESCRIPTION. The default is 10.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.
        results_folder : TYPE, optional
            DESCRIPTION. The default is os.getcwd().

        Returns
        -------
        nac_file : TYPE
            DESCRIPTION.
        nac_df_system : TYPE
            DESCRIPTION.
        
        TODO: make call to ask for start stop, etc.
        
        """

        print(f'Calculating {name}')

        indexes=[[n] for n in name.split('-')] 
        names=[f'l{i}' for i in range(1, len(indexes)+2)] # +2 to account for number of molecules 

       
        nac_file=f'{results_folder}/dNAC_{len(distances)}-i{start}-o{stop}-s{stride}-{timestep}ps.npy'
        
        if not os.path.exists(nac_file):
             
            dists=Featurize.distance_calculation(topology, trajectory, distances, start, stop, results_folder, name, stride, timestep)

            #NAC calculation
            nac_array=np.around(np.sqrt(np.power(dists, 2).sum(axis=0)/len(dists)), 3) #SQRT(SUM(di^2)/#i)  
            np.save(nac_file, nac_array)
            
        else:
            #print(f'dNAC file for {name} found.') 
            nac_array=np.load(nac_file)

        
        #TODO: Check behaviour for ref > 1
        
        frames, sel, ref=np.shape(nac_array)
        nac_array=nac_array.reshape(frames, ref*sel)
        
        indexes.append([e for e in np.arange(1,sel+1)])
        column_index=pd.MultiIndex.from_product(indexes, names=names)        

        nac_df_system=pd.DataFrame(nac_array, index=np.arange(start, stop, stride), columns=column_index)
        
        print(nac_df_system)    
            
        return nac_file, nac_df_system
    
    
    
    @staticmethod
    def nac_precalculated(results_folder, name=None):
        """
        TODO: TO BE deprecated.

        Parameters
        ----------
        results_folder : TYPE
            DESCRIPTION.
        name : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        data : TYPE
            DESCRIPTION.
        nac_df_system : TYPE
            DESCRIPTION.

        """
        
        import glob
   
        #TODO: Make it less specific by defining start, stop, etc. using name specs.
        data=str(glob.glob(f'{results_folder}NAC2*.npy')[0]) 
                
        print(f'Using pre-calculated file: {data}')
                
        raw_data=np.load(data)
               
        #Reconstruct a DataFrame of stored feature into a Dataframe
        #TODO: Check behaviour for ref > 1.
                    
        frames, ref, sel=np.shape(raw_data)
        raw_reshape=raw_data.reshape(frames, ref*sel)
                            
        if ref == 1:
            nac_df_system=pd.DataFrame(raw_reshape) 
            nac_df_system.columns=nac_df_system.columns + 1
        else:
            nac_df_system=pd.DataFrame()
            split=np.split(raw_reshape, ref, axis=1)
            for ref in split:
                df_ref=pd.DataFrame(ref)
                df_ref.columns=df_ref.columns + 1
                nac_df_system=pd.concat([nac_df_system, df_ref], axis=0)
    
        return data, nac_df_system

    
    @staticmethod
    def distance_calculation(topology, 
                             trajectory, 
                             distances, 
                             start, 
                             stop, 
                             results_folder, 
                             name, 
                             stride, 
                             timestep):
        """
        The workhorse function to calculate distances. Uses MDAnalysis.

        Parameters
        ----------
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
        
        dists=[] #list of distance arrays to be populated
            
        #iterate through the list of sels. 
        #For each sel, define atomgroup1 (sel1) and atomgroup2 (sel2)
        
        for idx, dist in enumerate(distances, 1): 
            dist_file=f'{results_folder}/distance{idx}-i{start}-o{stop}-s{stride}-{timestep}ps.npy'
            
            if not os.path.exists(dist_file):
                                
                print(f'Distance {idx} not found for {name}. Reading trajectory...')  
                u=mda.Universe(topology, trajectory)
                sel1, sel2 =u.select_atoms(dist[0]).positions, u.select_atoms(dist[1]).positions
                    
                #print(f'\t\tsel1: {dist[0]} ({len(sel1)})\n\t\tsel2: {dist[1]} ({len(sel2)})
                print(f'Calculating {name}')        
                    
                dists_=np.around(
                                [D.distance_array(sel1, sel2, box=u.dimensions) for ts in u.trajectory[start:stop:stride]],
                                decimals=3) 
                                
                np.save(dist_file, dists_)
                        
            else:
                dists_=np.load(dist_file)
                #print(f'\tDistance {idx} found for {name}. Shape: {np.shape(dists_)}')
                    
            dists.append(dists_)
            
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