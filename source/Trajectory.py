# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:50:14 2021

@author: hcarv
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda
import tools

class Trajectory:
    """Class to perform operatoin on trajectories using MDTRAJ."""
    
    
    def __init__(self, systems, results):
        self.systems=systems
        self.results=results
  
    
  
    def start_simulations(self):
        
        print(self.systems)
  
        
    
  
    
    @staticmethod
    def trj_filter(df, name, value):
        """
        

        Parameters
        ----------
        df : DataFrame
            A dataframe which contains values of project trajectories (e.g. some feature).
        name : str
            The name of the system of each trajectory, used to filter the DataFrame.
        value : The value to be searched in the dataframe. 
            DESCRIPTION.

        Returns
        -------
        numpy.array
            An array of index values of the trajectory where value was found.
        """
            
        #n=name.split('-') #alternative of n, ready for hierarchical operator     
        
        for index, level in enumerate(name.split('-'), 1):
            if index == 1:
                filtered=df
            else:
                filtered=filtered
                
            filtered=filtered.loc[:,filtered.columns.get_level_values(f'l{index}').isin([level])]
                       

        sel_frames=filtered.loc[(filtered.values == value)]
        frames=sel_frames.index.values #np.array(list(sel_frames.index.values))
        
        #print(frames)
        return frames
    
    
    
    def extractFrames_by_iterable(self, df, iterable, feature, t_start=20000, extract='not water') -> dict:
        """
        
        Extract frames trajectories based on iterable and dataframe.
        Writes the corresponding frames as .xtc and the corresponding .pdb. 
        Legacy: option to save also the waters controlled with --noWater.

        Parameters
        ----------
        df : DataFrame
            A dataframe which contains featurized project trajectory values.
        iterable : list
            A set of values to be used as criteria to filter the featurized trajectories.
        feature : str
            The name of the feature.
        t_start : int, optional
            The starting frame index. 
        extract : str, optional
            A string of atom selections (MDTRAJ). The default is 'not water'.

        
        NOTES
        -----
        
        IMPORTANT: automate t_start definition (equilibration phase discarded)


        Returns
        -------
        dict
            A dictionary of 'names': 'extracted Frame Files'.
            
        

        """
        
        import mdtraj as md
        
        #print(iterable)
        #print(feature)
            
        def calculate(name, system):
            
            system.frames=[]
            frames_iterables=[]
            
            for value in iterable:
                
                #print(f'Applying filter for iterable {value}')
                frames=Trajectory.trj_filter(df, name, value)   #Call to method for df filtering
                frames_iterables.append(frames)
            
            
            frames_iterables_dict={}    
            for frames, value in zip(frames_iterables, iterable):

                ref_topology=f'{system.results_folder}frames_{name}_it{value}_{feature}.pdb'
                
                if len(frames) != 0:
                    frames=frames+t_start #IMPORTANT NOTE.      
                    file=f'{system.results_folder}frames_{name}_it{value}_{feature}.xtc'
                    if not os.path.exists(file):
                        ref_topology_obj=md.load_frame(system.trajectory, index=0, top=system.topology)
                        atom_indices=ref_topology_obj.topology.select(extract)
                        ref_topology_obj.atom_slice(atom_indices, inplace=True)
                        ref_topology_obj.save_pdb(ref_topology)
                    
                        print(ref_topology_obj)
                    
                        system.frames.append(file)   #Update system attribute
                
                        print(f'\tLoading full trajectory of {name}...')
                        traj=md.load(system.trajectory, top=system.topology)
                        atom_indices=traj.topology.select(extract)
                        
                        traj.atom_slice(atom_indices, inplace=True)
                    
                        print('\tExtracting...')
                        extracted_frames=traj[frames]
                        print(extracted_frames)
                    
                        print('\tSaving')
                        extracted_frames.save_xtc(file)
                        extracted_frames[0].save(ref_topology)

                    else:
                        pass
                        
                    
                    frames_iterables_dict[value]=(file, ref_topology)
                    
            return frames_iterables_dict

        
        extracted_frames_dict={}

        for name, system in self.systems.items():
            extracted_frames_dict[name]=calculate(name, system)
        
            
        self.trajectories=extracted_frames_dict
        
        #print(extracted_frames_dict)
        return extracted_frames_dict
    

    def filterTraj(self, t_start=0, extract='not water'):
        """
        
        Extract trajectory subsets.
        Writes the corresponding trajectories as .xtc and the corresponding .pdb. 
        Legacy: option to save also the waters controlled with --noWater.

        Parameters
        ----------
        t_start : int, optional
            The starting frame index. 
        extract : str, optional
            A string of atom selections (MDTRAJ). The default is 'not water'.

        
        NOTES
        -----
        
        IMPORTANT: automate t_start definition (equilibration phase discarded)


        """
        
        import mdtraj as md
        
        for name, system in self.systems.items():
        
            file=f'{system.path}/filtered_{name}.xtc'            
       
            if not os.path.exists(file):
                
                print(f'\tLoading full trajectory of {name}...')
                traj=md.load(system.trajectory, top=system.topology)
                atom_indices=traj.topology.select(extract)
                        
                traj.atom_slice(atom_indices, inplace=True)
                    
                print('\tSaving')
                traj.save_xtc(file)
                traj[0].save(f'{system.path}/filtered_{name}.pdb')
                
            else:
                pass
                        
    
    def DensityMap_frames(self, frames={}, 
                          level=2, 
                          density_selection='not protein', 
                          convert=True, 
                          unit='Molar',
                          stride=1,
                          dists=None):
    
        
        if not bool(frames):
            print('Trajectories not defined. Using those defined in trajectory instance.')
            try:
                frames=self.frames
            except AttributeError:
                frames=input("Trajectories not defined in trajectory instance. \n\tDo it: ")
    
        iterables=[]
        level_values=[]
        
        #Retrieve level values
        for name, v in frames.items():
            name_=name.split('-')
            level_values.append(name_[level])
            
            for iterable, file in v.items():
                iterables.append(iterable)

        uniques=np.unique(np.asarray(iterables))
        level_values_unique=np.unique(np.asarray(level_values))
     
        
        #Collect all dictionary of iterables into a dictionary of levels
        collect={}
        stats=pd.DataFrame()
        for level_unique in level_values_unique: #The level
            print(f'Level: {level_unique}')
            
            unique_dict={}
            for unique in uniques: #The iterable
                
                file_concat=f'{self.results}/superposed_{level_unique}-it{unique}-s{stride}'
                trj=f'{file_concat}.xtc'
                top=f'{file_concat}.pdb'
                files_to_concatenate=[]
                for name, v in frames.items():
                    if name.split('-')[level] == level_unique:
                        
                        for iterable, xtc_pdb in v.items():
                    
                            if iterable == unique:
                        
                                files_to_concatenate.append(xtc_pdb) #The files that are both concentration and iterable
                print(f'\tIterable: {unique}')
                
                if len(files_to_concatenate) != 0:
                    
                    superposed=self.concatenate_superpose(files_to_concatenate, trajectory=trj, topology=top, stride=stride)
                    
                    density=self.densityCalc(file_concat, trajectory=trj, topology=top)
                    
                    conf=self.distances(file_concat, dists, trajectory=trj, topology=top, stride=1, skip_frames=0)
                    
                    cluster=self.clusterMDAnalysis(file_concat, trajectory=trj, topology=top)
                    stats_unique=tools.Functions.density_stats(level_unique, stride, self.results, unique=unique)
                    
                    unique_dict[unique]=(superposed, density, cluster) #THIS IS IT
                    

                    
                else:
                    print('\tNo frames found.')

                collect[level_unique]=unique_dict #The dictionary of levels with a dictionary of iterables
                stats=pd.concat([stats, stats_unique], axis=1)
        #print(collect)
        
        return collect, stats


    def DensityMap_fullTraj(self,  
                             level=2, 
                             density_selection='not protein', 
                             convert=True,
                             unit='Molar',
                             filtered=False,
                             stride=1) -> dict:
        """
        

        Parameters
        ----------
        level : int, optional
            The hierarchy level to collect. The default is 2 (parameter).
        density_selection : str, optional
            The selection to calculate the density map (MDANALYSIS). The default is 'not protein' (water striped trj's).
        convert : bool, optional
            Whether to convert the density value into another unit. The default is True.
        unit : str, optional
            The concentration unit (MDANALYSIS). The default is 'Molar'.
        filtered : bool, optional
            Whether to use a filtered trajectory or not. The default is False.
        stride : int, optional
            The trajectory stride value (MDTRAJ). The default is 1.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        
    
        level_values=[]
    
        #Access individual systems. 
        #The id of project upon feature level should match the current id of project 
        for name, system in self.systems.items(): 
            name_=name.split('-')
            level_values.append(name_[level])       
        
  
        level_values_unique=np.unique(np.asarray(level_values))
        
        collect={}

        for level_unique in level_values_unique: #The concentration
            print(f'Level: {level_unique}')
                
            file_concat=f'{self.results}/superposed_{level_unique}-s{stride}'
            trj=f'{file_concat}.xtc'
            top=f'{file_concat}.pdb'
            
            files_to_concatenate=[]
            
            #Iterate though all systems and proceed if level match
            for name, system in self.systems.items(): 
                #print(name)
                
                if name.split('-')[level] == level_unique:
                    
                    if filtered == True: #Useful to work faster with e.g. no-water.
                        files_to_concatenate.append((f'{system.path}/filtered_{name}.xtc', f'{system.path}/filtered_{name}.pdb'))
                    else:
                        files_to_concatenate.append((system.trajectory, system.topology))            
            
            #print(f'Files to concatenate: {files_to_concatenate}')
                
            if len(files_to_concatenate) != 0:   
                
                superposed=self.concatenate_superpose(files_to_concatenate, trajectory=trj, topology=top, stride=stride)
                
                density=self.densityCalc(file_concat, trajectory=trj, topology=top)
                
                cluster=self.clusterMDAnalysis(file_concat, trajectory=trj, topology=top)
                #stats_unique=tools.Functions.density_stats(level_unique, stride, self.results)
       
                collect[level_unique]=(superposed, density, cluster) 
                print(collect[level_unique])   
        else:
            print('\tNo frames found.')

        plt.show()
        print(collect)
        
        return collect
    
    
    
    @staticmethod
    def clusterMDAnalysis(base_name, trajectory, topology, select="all"):
        """
        Function to obtain cluster using the encore.cluster tool from MDAnalysis.

        Parameters
        ----------
        base_name : str
            The base name of the system to be processed (incl. path).
        trajectory : str
            The trajectory file.
        topology : str
            The topology file.
        selection : str, optional
            The selection for calculation (MDAnalysis syntax). The default is 'protein'.

        Returns
        -------
        cluster_file : str
            The name of the cluster files.

        """
        
        import MDAnalysis.analysis.encore as encore


        #if base_name == '/media/dataHog/hca/proLig_CalB-Methanol/project_results/superposed_50mM-it13-s1':

            
        cluster_file=f'{base_name}-clusters.pdb'
            
        if not os.path.exists(cluster_file):
            print('\tLoading files...')
            u=mda.Universe(topology, trajectory)
        
        
            print(f'\tNumber of frames: {len(u.trajectory)}')
            if len(u.trajectory) > 1:
                print('\tCluster file not found. Calculating...')
                n_c=20
                finished = False
                while finished == False:
                    try:
                        clusters = encore.cluster(u, method=encore.KMeans(n_clusters=n_c))
                        #clusters = encore.cluster(u, method=encore.DBSCAN(eps=3, min_samples=200, n_jobs=60))
                        print('\t', clusters)
                        finished = True
                    
                    except ValueError:
                        print('not possible')
                        
                centroids=[]
                elements=[]
                
                for cluster in clusters:    
                    print(f'\t\tCluster: {cluster}')
                    centroids.append(cluster.centroid)
                    elements.append(len(cluster.elements))
                
                for c, e in zip(centroids, elements):
                    print(f'\t\tCentroid: {c}, Element(s): {e}')

                
                print('\t\tSaving file')
                selection = u.select_atoms(select)
                with mda.Writer(cluster_file, selection) as W:
                    for centroid in centroids:
                        u.trajectory[centroid]
                        W.write(selection)
                
                #print(W)
            
            
        else:
            print('\tCluster file found')
                
        return cluster_file

    @staticmethod
    def concatenate_superpose(files_to_concatenate, trajectory, topology, ref_frame=0, stride=1, atom_set='backbone') -> object:
        """
        

        Parameters
        ----------
        files_to_concatenate : tuple
            A list of tuples (trajectory, topology).
        trajectory : path
            The path (name) of the output trajectory file.
        topology : path
            The path (name) of the output topology file.
        ref_frame : int, optional
            Reference frame to superpose (MDTRAJ). The default is 0.
        stride : int, optional
            The trajectory stride value (MDTRAJ). The default is 1.
        atom_set : str, optional
            The selection used as reference atom group for superposing (MDTRAJ). The default is 'backbone'.

        Returns
        -------
        superposed : object
            A superposed MDTRAJ trajectory object.

        """
        
        import mdtraj as md
        
        try:    
   
            if not os.path.exists(trajectory): 
                
                concatenated_trajectories=md.join([md.load(file[0], top=file[1], stride=stride) for file in files_to_concatenate]) 
                print(f'\tN. frames: {concatenated_trajectories.n_frames}')   
                
                print('\tSuperposing')
                concatenated_trajectories.image_molecules(inplace=True)
                atom_indices=concatenated_trajectories.topology.select(atom_set)
                superposed=concatenated_trajectories.superpose(concatenated_trajectories, frame=ref_frame, atom_indices=atom_indices)
                
                superposed.save_xtc(trajectory)
                superposed[0].save(topology)
                
            return (topology, trajectory)
        
        except OSError:
            print('tFile not found.')
            pass
    
    @staticmethod
    def densityCalc(base_name, trajectory, topology, 
                    selection='name OA', 
                    convert=True, 
                    unit='Molar', 
                    start=0, 
                    stop=-1):
        """
        Method to calculate the spatial densities of atom selection. 
        Uses the 'DensityAnalysis' tool from MDAnalysis. 
        

        Parameters
        ----------
        base_name : str
            The base name of the system to be processed (incl. path).
        trajectory : str
            The trajectory file.
        topology : str
            The topology file.
        selection : str, optional
            The selection for calculation (MDAnalysis syntax). The default is 'name OA'.
        convert : bool, optional
            Wether to convert or not density values into another unit. The default is True.
        unit : str, optional
            The density unit. The default is 'Molar'.
        start : int, optional
            The initial frame for calculation. The default is 0.
        stop : int, optional
            The last frame for calculation. The default is -1 (last).

        Returns
        -------
        density_file : str
            The name of the grid file ".dx".

        """
                    
        
        from MDAnalysis.analysis.density import DensityAnalysis
        
        
        density_file=f'{base_name}-{unit}.dx'
        
        try:
        
            if not os.path.exists(density_file):
            
                print("\tCalculating density")
                u= mda.Universe(topology, trajectory)
                    
                selected_set=u.select_atoms(selection)
                
                #print(len(selected_set))
            
                D = DensityAnalysis(selected_set)
                D.run(start=start, stop=stop, step=1)
                
                if convert == True:
                
                    D.density.convert_density(unit)
                    D.density.export(density_file, type="double")
                else:
                    D.density.export(density_file, type="double")
            
                prob_density= D.density.grid / D.density.grid.sum()
                np.save(f'{base_name}-prob_density.npy', prob_density)
                
                # Histogram. Ensure that the density is A^{-3}
                D.density.convert_density("A^{-3}")
                dV = np.prod(D.density.delta)
                atom_count_histogram = D.density.grid * dV
                np.save(f'{base_name}-histogram.npy', atom_count_histogram)
        
            else:
                print('\tDensity file found')
            
        except FileNotFoundError:
            print('File not found')
            pass
        
        return density_file  
    
    
    
    @staticmethod
    def distances(self,  dists, trajectory, topology, stride=1, skip_frames=0):
        '''Calculates distances between pairs of atom groups (dist) for each set of selections "dists" using MDAnalysis D.distance_array method.
        Stores the distances in results_dir. Returns the distances as dataframes.
	    Uses the NAC_frames files obtained in extract_frames(). 
	    NOTE: Can only work for multiple trajectories if the number of atoms is the same (--noWater)'''
         
        
        import MDAnalysis.analysis.distances as D
        
        u=mda.Universe(topology, trajectory)
                
        #print(f'\tBase name: {base_name}')
                
        distances={} # This is the dictionary of d distances.
                
        for idx, dist in enumerate(dists, 1): 
                    
            sel1, sel2 =u.select_atoms(dist[0]), u.select_atoms(dist[1])
				
            distance=np.around([D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory], decimals=3)
				
            q=np.quantile(distance.flat, q=np.arange(0,1.05, 0.05))
            print(q)
            plt.plot(np.arange(0,1.05, 0.05), q, label=topology)
            #plt.show()
                    
            distances[idx]=distance