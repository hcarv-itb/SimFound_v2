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
import mdtraj as md
import glob
import re
import pickle
from time import process_time_ns

#SFv2
try:
 import tools
except:
    pass

class Trajectory:
    """Class to perform operatoin on trajectories using MDTRAJ."""
    

    trj_types=['h5', 'dcd', 'xtc']
    top_types=['pdb', 'gro']
    
    def __init__(self, systems, results):
        """
        

        Parameters
        ----------
        systems : TYPE
            DESCRIPTION.
        results : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.systems=systems
        self.results=results
        
        
    @staticmethod
    def fileFilter(name, 
                   file, 
                   equilibration, 
                   production,
                   def_file=None,
                   warnings=False, 
                   filterW=False):
        """
        

        Parameters
        ----------
        name : TYPE
            DESCRIPTION
        trajectories : TYPE
            DESCRIPTION.
        equilibration : TYPE
            DESCRIPTION.
        production : TYPE
            DESCRIPTION.
        warnings : TYPE, optional
            DESCRIPTION. The default is False.
        filterW : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        trajectories : TYPE
            DESCRIPTION.

        """

        to_del = []
        if type(file) == str:
            file = [file]
        for idx, t in enumerate(file):
                    
            if not equilibration:
                if re.search('equilibration', t):
                    to_del.append(idx)
            if not production:
                if re.search('production', t):
                    to_del.append(idx)
                    
        if len(to_del) > 0:
            for d in reversed(to_del):
                file.pop(d)
    
        if filterW:
            filtered = []
            for t in file:
                if re.search('_noW', t):
                    filtered.append(t)
            if len(filtered) > 0:
                file = filtered
    
        if def_file == None:
            if len(file) > 1 and warnings:
                print(f'Warning: Found more than one file for {name}:')
                for idx, t in enumerate(file):
                    print(f'\t{idx} : {t}')
                    
                select=input('Select one or more by index: ').split()
                file = [file[int(i)] for i in select]
            else:
                file = file
        
        else:
            select = []
            for def_f in def_file:
                for f in file:
                    if re.search(def_f, f):
                        select.append(f)
            if len(select) > 0:
                file = select
                #print('Selected ', file)
                 
        to_load = []

        for f in file:
            file_format=str(f).split('.')[-1]
            if (file_format in Trajectory.trj_types or Trajectory.top_types):
                to_load.append(f)

                    
        #print('to load', to_load)
        return to_load
        

    @staticmethod
    def findTrajectory(workdir):
        """
        

        Parameters
        ----------
        workdir : TYPE
            DESCRIPTION.

        Returns
        -------
        trajectories : TYPE
            DESCRIPTION.

        """
        
        trajectories = []
        for trj_type in Trajectory.trj_types:
            files=glob.glob(f'{workdir}/*.{trj_type}')
    
            for f in files:    
                trajectories.append(f)
        
        #print(trajectories)
        
        return trajectories
                                
    @staticmethod
    def findTopology(workdir, ref_name, input_topology=None):
        """
        

        Parameters
        ----------
        workdir : TYPE
            DESCRIPTION.
        ref_name : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        topologies = []

        files=glob.glob(f'{workdir}/{ref_name}')
        
        if len(files) > 0:
            return files[0]
        
        else:
            print(f'Reference structure {workdir}/{ref_name} not found')
            for top_type in Trajectory.top_types:
                files=glob.glob(f'{workdir}/*.{top_type}')
                for f in files:
                    topologies.append(f)
                    
            if len(topologies) == 0:
                topologies = input_topology
                
            
            
            
            print('Defined topology: ', topologies)
            return topologies

    
    @staticmethod
    def loadTrajectory(system_specs, specs):
        """
        Workhorse function to load a trajectory.

        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        trajectory : TYPE
            DESCRIPTION.
        task : TYPE
            DESCRIPTION.
        subset : TYPE, optional
            DESCRIPTION. The default is 'all'.
        priority : TYPE, optional
            DESCRIPTION. The default is ('h5', 'dcd', 'xtc').

        Returns
        -------
        traj : TYPE
            DESCRIPTION.

        """
        clock_start=process_time_ns()   
        traj = None
        
        (trajectories, topologies, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs 
        #TODO: not water is hardcoded. pass it up.
        store_traj = False
        pkl_path = os.path.abspath(f'{results_folder}/{name}_{start}_{stop}_{stride}.pkl')
        
        if os.path.exists(pkl_path) and store_traj:
            
            print('\tUnpickling for', name)
            pkl_file=open(pkl_path, 'rb')
            traj = pickle.load(pkl_file)
            
        else:
            if len(topologies) > 1:
                print('Warning! more than one topology for ', name)

            trj_formats=[t.split('.')[1] for t in trajectories]
            for trj_format in set(trj_formats):
                for topology in topologies:
                    try:    
                        if task == 'MDTraj':
                            if trj_format == 'h5':
                                traj=md.load(trajectories, top=topology)
                                print('loading h5 for', name)
                            else:  
                                ref_top_indices=md.load(topology).topology.select(subset)
                                traj=md.load(trajectories, top=topology, atom_indices=ref_top_indices)
                                
                            print(f'Load time for {name} ({len(traj)} frames): {np.round((process_time_ns() - clock_start)*1e-9, decimals=3)} s.')
                        elif task == 'MDAnalysis':
                            if trj_format == 'xtc':
                                traj=mda.Universe(topology, trajectories, continuous=True)
                            elif trj_format != 'h5':
                                traj=mda.Universe(topology, trajectories)
                            
                            print(f'Load time for {name} ({traj.n_frames} frames): {np.round((process_time_ns() - clock_start)*1e-9, decimals=3)} s.')
                        
                        #print(name, task, topology, trajectories, end='\r')
                        break
                    except:
                        pass
                        #print('\tWarning! Trajectory not loaded for',  name)

                    
            if store_traj and traj != None:
            
                traj_pkl=open(pkl_path, 'wb')
                pickle.dump(traj, traj_pkl)
                traj_pkl.close()

        if traj == None:
            print('Warning! Could not load ', name)    
        
        
        return traj
            
    @staticmethod
    def trj_filter(df, name, value):
        """
        Extract frames based on discretized trajectory dataframes.        

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
        
        print(name)
        
        n=name.split('-') #alternative of n, ready for hierarchical operator     
        
        for index, level in enumerate(name.split('-'), 1):
            print(index, level)
            if index == 1:
                filtered=df
            else:
                filtered=filtered
                
            filtered=filtered.loc[:,filtered.columns.get_level_values(f'l{index}').isin([level])]

        sel_frames=filtered.loc[(filtered.values == value)]
        frames=sel_frames.index.values #np.array(list(sel_frames.index.values))

        return frames
    
    
    
    @staticmethod
    def pre_process_MSM(trajectory_set, 
                    topology, 
                    superpose_to='Protein and backbone',
                    slice_traj='not water',
                    ref_frame=0,
                    overwrite=False):
        """
        
        Serves MSM module. Trajectory files are loaded and superposed to the provided Topology object, the one used for feature extraction.

        Parameters
        ----------
        trajectories : TYPE
            DESCRIPTION.
        topology : TYPE
            DESCRIPTION.
        superpose_to : TYPE, optional
            DESCRIPTION. The default is 'Protein and backbone'.
        ref_frame : TYPE, optional
            DESCRIPTION. The default is 0.
        slice_traj : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        trajectories_mod : TYPE
            DESCRIPTION.

        """
        
        trajectories_mod=[]
        for trajectories in trajectory_set:
            for trajectory in trajectories:
                file_name, ext=trajectory.split('.')
                
                mod_file_name=f'{file_name}_superposed.{ext}'
                
                if not os.path.exists(mod_file_name) or overwrite:
                    print(f'\tPre-processing {trajectory}', end='\r')
                    traj=md.load(trajectory, top=topology)
                    traj.image_molecules(inplace=True)
                    atom_indices=traj.topology.select(slice_traj)
                    traj.atom_slice(atom_indices, inplace=True)
                    traj.superpose(topology, frame=ref_frame, atom_indices=superpose_to)
                    
                    if ext == 'xtc':
                        traj.save_xtc(f'{file_name}_superposed.xtc')
                    elif ext == 'dcd':
                        traj.save_dcd(f'{file_name}_superposed.dcd')
            
                trajectories_mod.append(mod_file_name)   
    
        return trajectories_mod

    def extractFrames_by_iterable(self, 
                                  df, 
                                  iterable, 
                                  feature, 
                                  t_start=20000, 
                                  extract='not water',
                                  equilibration=False, 
                                  production=True, 
                                  def_top=None, 
                                  def_traj=None):
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
        
            
        def calculate(name, system):
            
            system.frames=[]
            frames_iterables=[]
            
            for value in iterable:
                
                
                frames=Trajectory.trj_filter(df, name, value)   #Call to method for df filtering
                frames_iterables.append(frames)            
            
            topology=self.fileFilter(name, 
                                     system.topology, 
                                     equilibration, 
                                     production, 
                                     def_file=def_top)[0]

            trajectory=self.fileFilter(name, 
                                       system.trajectory, 
                                       equilibration, 
                                       production,
                                       def_file=def_traj)[0]
            

            frames_iterables_dict={}    
            for frames, value in zip(frames_iterables, iterable):

                ref_topology=f'{system.results_folder}frames_{name}_it{value}_{feature}.pdb'
                
                if len(frames) != 0:
                    
                    print(f'Applying filter for iterable {value}')
                    #TODO: Fix this for automated fetch of propert t_start. input df is reseting to 0, should keep track of what featurization did.
                    frames=frames+t_start #IMPORTANT NOTE. 
                    
                    file=f'{system.results_folder}frames_{name}_it{value}_{feature}.xtc'
                    if not os.path.exists(file):
                        ref_topology_obj=md.load_frame(trajectory, index=0, top=topology)
                        atom_indices=ref_topology_obj.topology.select(extract)
                        ref_topology_obj.atom_slice(atom_indices, inplace=True)
                        ref_topology_obj.save_pdb(ref_topology)
                    
                        #print(ref_topology_obj)
                        system.frames.append(file)   #Update system attribute
                
                        print(f'\tLoading full trajectory of {name}...')
                        traj=md.load(trajectory, top=topology)
                        atom_indices=traj.topology.select(extract)
                        
                        traj.atom_slice(atom_indices, inplace=True)
                    
                        print('\tExtracting...')
                        extracted_frames=traj[frames]
                        print('\t', extracted_frames)
                    
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
    

    def monitorTraj(self):
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
        import glob
        
        for name, system in self.systems.items():
        
            
            #TODO: get definitions from protocols as well
            
            
            #Get definitions from file search. 
            
            print(name)
            
            import re
                
            trj_files={}
            report_files={}
            
            files={}
            
            file_pattern={'report': '.csv', 
                        'trajectory' : '.dcd',
                        'trajectory_h5' : '.h5',
                        'reference_positions' : '_0.pdb',
                        'final_positions' : '.pdb',
                        'checkpoint' : '.chk',
                        'topology' : 'system.pdb',
                        'minimmized' : 'minimization.pdb'}
            
            
            def checkFile(file):
                
                for label, v in file_pattern.items():
                   
                    name = re.split(r'/', file)[-1]                  
                    
                    if v in name:
                        
                        if 'NPT' in name:
                            
                            return f'{label}-NPT'
                        
                        elif 'NVT' in name:
                            
                            return f'{label}-NVT'
                        
                        else:
                            
                            print('this is bat country')
                    
                
            
            equilibrations = {}
            for file in glob.glob(os.path.abspath(f'{system.path}/equilibration*')):
                
                label = checkFile(file)
                #print(f'{label} : {file}') 
                equilibrations[label]=file
                
            productions = {}
            for file in glob.glob(os.path.abspath(f'{system.path}/production*')):
                
                label = checkFile(file)
                #print(f'{label} : {file}') 
                productions[label]=file
            
            print(f'\tEquilibrations: {equilibrations}')
            
            #print(f'\tProductions: {productions}')
            
            #print(equilibrations['report'])

            


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
                        
    
    def DensityMap_frames(self, 
                          frames, 
                          level=2, 
                          density_selection='not protein', 
                          convert=True, 
                          unit='Molar',
                          stride=1,
                          dists=None):
        """
        

        Parameters
        ----------
        frames : TYPE, optional
            DESCRIPTION. The default is {}.
        level : TYPE, optional
            DESCRIPTION. The default is 2.
        density_selection : TYPE, optional
            DESCRIPTION. The default is 'not protein'.
        convert : TYPE, optional
            DESCRIPTION. The default is True.
        unit : TYPE, optional
            DESCRIPTION. The default is 'Molar'.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.
        dists : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        collect : TYPE
            DESCRIPTION.
        stats : TYPE
            DESCRIPTION.

        """
    
        
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
                trj=os.path.abspath(f'{file_concat}.xtc')
                top=os.path.abspath(f'{file_concat}.pdb')
                
                files_to_concatenate=[]
                for name, v in frames.items():
                    if name.split('-')[level] == level_unique:
                        
                        for iterable, xtc_pdb in v.items():
                    
                            if iterable == unique:
                        
                                files_to_concatenate.append(xtc_pdb) #The files that are both concentration and iterable
                print(f'\tIterable: {unique}')
                
                if len(files_to_concatenate) != 0:
                    
                    #important stuff going on here
                    superposed=self.concatenate_superpose(files_to_concatenate, trajectory=trj, topology=top, stride=stride)
                    density=self.densityCalc(file_concat, trajectory=trj, topology=top)
                    cluster=self.clusterMDAnalysis(file_concat, trajectory=trj, topology=top)
                    #conf=self.distances(file_concat, dists, trajectory=trj, topology=top, stride=1, skip_frames=0)
                    stats_unique=tools.Functions.density_stats(level_unique, stride, self.results, unique=unique)
                    
                    unique_dict[unique]=(density, cluster) #THIS IS IT
                    
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
    def clusterMDAnalysis(base_name, trajectory, topology, n_clusters_=20, select="name CA", n_cores=6):
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

        #TOODO: make cluster file name more specific    
        cluster_file=f'{base_name}-{n_clusters_}clusters.pdb'  
        
        if not os.path.exists(cluster_file):
            print('\tCalculating cluster: \n\tLoading files...')
            u=mda.Universe(topology, trajectory)
        
            print(f'\tNumber of frames to cluster: {len(u.trajectory)}')
            if len(u.trajectory) > 1:
                print('\tCluster file not found. Clustering...')

                try:
                    clusters = encore.cluster(u, select=select, ncores=n_cores, method=encore.KMeans(n_clusters=n_clusters_))
                    #clusters = encore.cluster(u, method=encore.DBSCAN(eps=3, min_samples=200, n_jobs=60))
                    
                except ValueError:
                    print('\tWarning! Clustering failed.')
                        
                centroids=[]
                elements=[]
                
                for cluster in clusters:    
                    print(f'\t\tCluster: {cluster}')
                    centroids.append(cluster.centroid)
                    elements.append(len(cluster.elements))
                
# =============================================================================
#                 for c, e in zip(centroids, elements):
#                     print(f'\t\tCentroid: {c}, Element(s): {e}')
# =============================================================================
       
                print('\tSaving cluster file...')
                selection = u.select_atoms(select)
                with mda.Writer(cluster_file, selection) as W:
                    for centroid in centroids:
                        u.trajectory[centroid]
                        W.write(selection)
                
                #print(W)

        else:
            print('\Cluster file found')
                
        return cluster_file

    @staticmethod
    def concatenate_superpose(files_to_concatenate, 
                              trajectory, 
                              topology, 
                              ref_frame=0, 
                              stride=1, 
                              atom_set='backbone',
                              start=0,
                              stop=-1):
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
        
        if not os.path.exists(trajectory): 
                
            #concatenated_trajectories=md.join([md.load(file[0], top=file[1][0], stride=stride) for file in files_to_concatenate])
            concatenated_trajectories=md.join([md.load(file[0], top=file[1])[start:stop:stride] for file in files_to_concatenate])
            print(f'\tNumber of frames to superpose: {concatenated_trajectories.n_frames}')   
            
            print('\tSuperposing...')
            concatenated_trajectories.image_molecules(inplace=True)
            atom_indices=concatenated_trajectories.topology.select(atom_set)
            superposed=concatenated_trajectories.superpose(concatenated_trajectories, frame=ref_frame, atom_indices=atom_indices)
            
            superposed.save_xtc(trajectory)
            superposed[0].save(topology)
      
            

        
        
        
    
    
    @staticmethod
    def densityCalc(base_name, 
                    trajectory, 
                    topology, 
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
            
                print("\tCalculating density...")
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
    
