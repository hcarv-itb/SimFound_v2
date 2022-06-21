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
    state_samples = 100
    
    
    def __init__(self, 
                 project,
                 equilibration=False, 
                 production=True, 
                 def_top=None, 
                 def_traj=None,
                 results=None):
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
        
        self.equilibration = equilibration
        self.production = production
        self.def_top = def_top
        self.def_traj = def_traj
        self.systems=project.systems
        self.project = project
        if results != None:
            self.results=results
        else:
            self.results = project.results
        
        


        

    
    
        
    
    
    
    @staticmethod
    def pre_process_trajectory(trajectory_set, 
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

    def frame_selector(self, df, states, mode='all'):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        states : TYPE
            DESCRIPTION.

        Returns
        -------
        extracted_frames : TYPE
            DESCRIPTION.

        """
        extracted_frames= []
        def extract():
            if not sel_frames.empty: 
            
                frames = sel_frames.index.values
                if self.n_samples != 0:
                    frames = np.random.choice(frames, size=min(int(self.n_samples / samples[state][parameter]), len(frames)))
                #frames = np.random.choice(frames, size=self.n_samples, replace=False)
                topology=self.fileFilter(name, system.topology, self.equilibration, self.production, def_file=self.def_top)[0]
                trajectory=self.fileFilter(name, system.trajectory, self.equilibration, self.production, def_file=self.def_traj)[0]
                if mode == 'all':
                    extracted_frames.append((state, system.parameter, system.replicate, topology, trajectory, frames))  
                elif mode == 'by_reference':
                    extracted_frames.append((state, system.parameter, system.replicate, topology, trajectory, {ref : frames})) 
                
        samples = {}
        parameters_extract = []
        for state in states:
            samples[state] = {}
            parameters = df.columns.get_level_values(2).unique()
            for parameter in parameters:
                parameter_df = df.loc[:, df.columns.get_level_values(2) == parameter]
                samples[state][parameter] = {}
                replicates = parameter_df.columns.get_level_values(3).unique()
                n_replicates = 0
                for replicate in replicates:
                    replicate_df = parameter_df.loc[:, parameter_df.columns.get_level_values(3) == replicate]
                    if not replicate_df.loc[(replicate_df.values == state)].empty:
                        n_replicates +=1
        
                samples[state][parameter] = n_replicates
                if n_replicates > 0:
                    parameters_extract.append(parameter)
        
        
        for name, system in self.systems.items():
            
            if system.parameter in set(parameters_extract):
            
                parameter_cols = df.columns.get_level_values(2) == system.parameter
                replicate_cols = df.columns.get_level_values(3) == str(system.replicate)
                extract_columns = np.logical_and(parameter_cols, replicate_cols)
                system_df = df.loc[:, extract_columns]
                
                for state in states:
                    system_df.reset_index(drop=True, inplace=True)
                    if mode == 'all':
                        sel_frames = system_df.loc[(system_df.values == state)]
                        extract()
                        if not sel_frames.empty: 
                            frames = sel_frames.index.values
                            if self.n_samples != 0:
                                frames = np.random.choice(frames, size=min(int(self.n_samples / samples[state][parameter]), len(frames)))
                            #frames = np.random.choice(frames, size=self.n_samples, replace=False)
                            topology=self.fileFilter(name, system.topology, self.equilibration, self.production, def_file=self.def_top)[0]
                            trajectory=self.fileFilter(name, system.trajectory, self.equilibration, self.production, def_file=self.def_traj)[0]
                            extracted_frames.append((state, system.parameter, system.replicate, topology, trajectory, frames))
                    elif mode == 'by_reference':
                        references = system_df.columns.get_level_values(4).unique()
                        for ref in references:
                            ref_df = system_df.loc[:, system_df.columns.get_level_values(4) == ref]
                            sel_frames = ref_df.loc[(ref_df.values == state)]
                            extract()
                            
              
                                    
        return np.asarray(extracted_frames) #needs to be like this for file selector

    def file_selector(self, frames_to_extract, states, state_labels, parameters):
        """
        

        Parameters
        ----------
        frames_to_extract : TYPE
            DESCRIPTION.
        states : TYPE
            DESCRIPTION.
        state_labels : TYPE
            DESCRIPTION.
        parameters : TYPE
            DESCRIPTION.

        Returns
        -------
        out_files : TYPE
            DESCRIPTION.

        """
        
        out_files = []
        for iterable in states:
            iterable_frames = frames_to_extract[frames_to_extract[:,0] == iterable]
            for parameter in parameters:
                parameter_frames = iterable_frames[iterable_frames[:,1] == parameter]
                if len(parameter_frames):
                    print(f'Processing {iterable}: {parameter}', end='\r')
                    _trajs = parameter_frames[:,4]
                    _topology = parameter_frames[:,3]
                    _frames = parameter_frames[:,5]
                    systems_to_concatenate = list(zip(_trajs, _topology, _frames))
                    if state_labels is None:
                        trajectory_name = f'{self.project.results}/structures/stateFrames_{self.feature_name}_{iterable}_{parameter}'
                    else:
                        trajectory_name = f'{self.project.results}/structures/stateFrames_{self.feature_name}_{state_labels[iterable]}_{parameter}'
                    if self.n_samples != 0:
                        trajectory_name = f'{trajectory_name}_samples'
                    out_files.append((systems_to_concatenate, trajectory_name))

        return out_files



    def extractFrames_by_state(self, 
                               df, 
                               states, 
                               feature_name=None, 
                               state_labels= None,
                               subset=None,
                               n_samples = 0):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        states : TYPE
            DESCRIPTION.
        feature_name : TYPE, optional
            DESCRIPTION. The default is None.
        state_labels : TYPE, optional
            DESCRIPTION. The default is None.
        subset : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        self.feature_name = feature_name
        self.n_samples = n_samples

        
        if not isinstance(states, list):
            states = [states]
        
        if subset != None:
            parameters = subset
        else:
            parameters = self.project.parameter 
            
 
        frames_to_extract= self.frame_selector(df, states)
        files_to_process = self.file_selector(frames_to_extract, states, state_labels, parameters)
    
        for system_files in files_to_process:
            (systems_to_concatenate, trajectory_name) = system_files
            self.concatenate_superpose(systems_to_concatenate, 
                                       trajectory_name,  
                                       w_frames=True)                  
            



    def clusters_by_state(self, 
                          df, 
                          states,                              
                          feature_name=None, 
                          state_labels= None,
                          subset=None,
                          n_clusters=20,
                          selection='name CA',
                          n_cores=6,
                          n_samples = 0):
        
        
        self.feature_name = feature_name
        self.n_samples = n_samples
        
        
        if not isinstance(states, list):
            states = [states]
        
        if subset != None:
            parameters = subset
        else:
            parameters = self.project.parameter 
            
 
        frames_to_extract= self.frame_selector(df, states)
        files_to_process = self.file_selector(frames_to_extract, states, state_labels, parameters)
    
        for system_files in files_to_process:
            (systems_to_concatenate, trajectory_name) = system_files
            self.concatenate_superpose(systems_to_concatenate, 
                                       trajectory_name,  
                                       w_frames=True) 
            self.cluster_calculation(trajectory_name, 
                                     n_clusters = n_clusters, 
                                     selection = selection, 
                                     n_cores = n_cores) 
    
                    
    
    def densityMaps_by_state(self, 
                             df, 
                             states, 
                             selection='not protein',
                             feature_name=None, 
                             state_labels= None,
                             subset=None,
                             convert=True, 
                             unit='Molar',
                             n_samples = 0,
                             extract_mode='all'):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        states : TYPE
            DESCRIPTION.
        selection : TYPE, optional
            DESCRIPTION. The default is 'not protein'.
        feature_name : TYPE, optional
            DESCRIPTION. The default is None.
        state_labels : TYPE, optional
            DESCRIPTION. The default is None.
        subset : TYPE, optional
            DESCRIPTION. The default is None.
        convert : TYPE, optional
            DESCRIPTION. The default is True.
        unit : TYPE, optional
            DESCRIPTION. The default is 'Molar'.

        Returns
        -------
        None.

        """
    
        self.feature_name = feature_name
        self.n_samples = n_samples
        
        if not isinstance(states, list):
            states = [states]
        
        if subset != None:
            parameters = subset
        else:
            parameters = self.project.parameter 
            
 
        frames_to_extract= self.frame_selector(df, states, mode=extract_mode)
        files_to_process = self.file_selector(frames_to_extract, states, state_labels, parameters)
    
        for system_files in files_to_process:
            (systems_to_concatenate, trajectory_name) = system_files
            if extract_mode == 'all':
                self.concatenate_superpose(systems_to_concatenate, 
                                       trajectory_name,  
                                       w_frames=True,
                                       mode=extract_mode)
                self.density_calculation(trajectory_name,                             
                                         selection, 
                                         convert=convert, 
                                         unit=unit) 
                
            elif extract_mode == 'by_reference':
               ref_traj_names = self.concatenate_superpose(systems_to_concatenate, 
                           trajectory_name,  
                           w_frames=True,
                           mode=extract_mode)
               
               for ref_traj_name in ref_traj_names:                
                   self.density_calculation(ref_traj_name,                             
                                            selection, 
                                            convert=convert, 
                                            unit=unit) 
       
    
    @staticmethod
    def cluster_calculation(trajectory_name, 
                            n_clusters=20, 
                            selection='name CA', 
                            n_cores=6,):
        """
        

        Parameters
        ----------
        trajectory_name : TYPE
            DESCRIPTION.
        n_clusters_ : TYPE, optional
            DESCRIPTION. The default is 20.
        select : TYPE, optional
            DESCRIPTION. The default is "name CA".
        n_cores : TYPE, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        None.

        """
        
        import MDAnalysis.analysis.encore as encore
           
        cluster_file=f'{trajectory_name}-{selection.replace(" ", "")}{n_clusters}clusters.pdb'  
        trajectory = f'{trajectory_name}.dcd'
        topology = f'{trajectory_name}.pdb'
        
        if not os.path.exists(cluster_file):
            print('\tCalculating cluster: ', cluster_file)
            u=mda.Universe(topology, trajectory)
        
            print(f'\tNumber of frames to cluster: {len(u.trajectory)}')
            if len(u.trajectory) > 1:
                print('\tCluster file not found. Clustering...')
                centroids=[]
                elements=[]
                try:
                    clusters = encore.cluster(u, select=selection, ncores=n_cores, method=encore.KMeans(n_clusters=n_clusters))
                    #clusters = encore.cluster(u, method=encore.DBSCAN(eps=3, min_samples=200, n_jobs=60))    
                except ValueError:
                    print('\tWarning! Clustering failed.')
                for cluster in clusters:    
                    print(f'\t\tCluster: {cluster}')
                    centroids.append(cluster.centroid)
                    elements.append(len(cluster.elements))       
# =============================================================================
#                 for c, e in zip(centroids, elements):
#                     print(f'\t\tCentroid: {c}, Element(s): {e}')
# =============================================================================
                print('\tSaving cluster file...')
                _selection = u.select_atoms(selection)
                with mda.Writer(cluster_file, u) as W: #_selection
                    for centroid in centroids:
                        u.trajectory[centroid]
                        W.write(u)
                #print(W)
        else:
            print('\Cluster file found')
                
        return cluster_file

    def concatenate_superpose(self, 
                              systems_to_concatenate, 
                              trajectory_name,  
                              ref_frame=0, 
                              stride=1, 
                              atom_set='backbone',
                              start=0,
                              stop=-1,
                              w_frames=False,
                              mode='all'):
        """
        

        Parameters
        ----------
        systems_to_concatenate : TYPE
            DESCRIPTION.
        trajectory_name : TYPE
            DESCRIPTION.
        ref_frame : TYPE, optional
            DESCRIPTION. The default is 0.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.
        atom_set : TYPE, optional
            DESCRIPTION. The default is 'backbone'.
        start : TYPE, optional
            DESCRIPTION. The default is 0.
        stop : TYPE, optional
            DESCRIPTION. The default is -1.
        w_frames : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        import mdtraj as md
        
        def superpose_save(save_file=None):
            
            if save_file != None:
                trajectory_name = save_file
            
                
            n_frames = concatenated_trajectories.n_frames
            out_frames = np.random.choice(range(n_frames), size=min(self.n_samples, n_frames))
            out_traj = concatenated_trajectories[out_frames]
            print(f'Superposing {out_traj.n_frames} frames of {trajectory_name}', end='\r')   
            out_traj.image_molecules(inplace=True)
            atom_indices=out_traj.topology.select(atom_set)
            superposed=out_traj.superpose(out_traj, frame=ref_frame, atom_indices=atom_indices)
            
            superposed.save_dcd(f'{trajectory_name}.dcd')
            superposed[0].save(f'{trajectory_name}.pdb')
        
        #Warning! check_topology set to False since top files are not the same. works OK for replicates with same top/trj    
        if not w_frames:
            if os.path.exists(f'{trajectory_name}.dcd'):
                print(f'Concatenating {len(systems_to_concatenate)} trajectory file(s) of {trajectory_name}') #, end='\r')
                concatenated_trajectories=md.join([md.load(file[0], top=file[1])[start:stop:stride] for file in systems_to_concatenate], check_topology=False)  
                superpose_save()
        else:        
            if mode == 'all':
                if not os.path.exists(f'{trajectory_name}.dcd'):
                    try:
                        trajs = [md.load(file[0], top=file[1])[file[2]] for file in systems_to_concatenate]
                        concatenated_trajectories=md.join(trajs, check_topology=False)
                    except MemoryError as v:
                        print(f'Warning! {v.__class__} : {v}\n Attempting load frame by frame')                   
                        concatenated_trajectories = md.join([md.join(
                                [md.load_frame(file[0], frame, top=file[1]) for frame in file[2]],
                                check_topology = False) for file in systems_to_concatenate], check_topology = False)
                    superpose_save()
                    
            elif mode == 'by_reference':
                chain_rep_traj = []
                for file in systems_to_concatenate:
                    #traj = md.load(file[0], top=file[1])
                    for ref, frames in file[2].items():
                        if len(frames):
                            ref_traj = md.join([md.load_frame(file[0], frame, top=file[1]) for frame in frames], check_topology = False)
                            #TODO: this, together with previous comment out make full traj load and  then slice. Consider memoryError/speed up 
                            #ref_traj = traj[frames] 
                            chain_rep_traj.append((ref, ref_traj))
                chain_rep_traj = np.asarray(chain_rep_traj)
                chains = set(chain_rep_traj[:,0])
                
                ref_trajectory_names = []
                for chain_ in chains:
                    ref_trajectory_name = f'{trajectory_name}_{chain_}'
                    ref_trajectory_names.append(ref_trajectory_name)
                    if not os.path.exists(f'{ref_trajectory_name}.dcd'):                  
                        chain_trajs = [chain_traj[1] for chain_traj in chain_rep_traj if chain_traj[0] == chain_]
                        concatenated_trajectories = md.join(chain_trajs, check_topology = False)
                        superpose_save(save_file=ref_trajectory_name)
                        
                return ref_trajectory_names
            
                        
                          
      
    
    @staticmethod
    def density_calculation(trajectory_name, 
                            selection, 
                            convert=True, 
                            unit='Molar', 
                            start=0, 
                            stop=-1):
        """
        

        Parameters
        ----------
        trajectory_name : TYPE
            DESCRIPTION.
        selection : TYPE, optional
            DESCRIPTION. The default is 'name OA'.
        convert : TYPE, optional
            DESCRIPTION. The default is True.
        unit : TYPE, optional
            DESCRIPTION. The default is 'Molar'.
        start : TYPE, optional
            DESCRIPTION. The default is 0.
        stop : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        density_file : TYPE
            DESCRIPTION.

        """
                    
        
        from MDAnalysis.analysis.density import DensityAnalysis
        
        density_file=f'{trajectory_name}-{unit}.dx'
        trajectory = f'{trajectory_name}.dcd'
        topology = f'{trajectory_name}.pdb'
        

        if not os.path.exists(density_file):
        
            print("\tCalculating density of ", density_file)
            u= mda.Universe(topology, trajectory)
                
            selected_set=u.select_atoms(selection)
            
            #print(len(selected_set))
        
            D = DensityAnalysis(selected_set)
            D.run(start=start, stop=stop, step=1)
            
            if convert == True:
                D.density.convert_density(unit)

            D.density.export(density_file, type="double")
        
            prob_density= D.density.grid / D.density.grid.sum()
            np.save(f'{trajectory_name}-prob_density.npy', prob_density)
            
            # Histogram. Ensure that the density is A^{-3}
            D.density.convert_density("A^{-3}")
            dV = np.prod(D.density.delta)
            atom_count_histogram = D.density.grid * dV
            np.save(f'{trajectory_name}-histogram.npy', atom_count_histogram)
            
            return density_file 
    
        else:
            print('\tDensity file found', density_file)
            pass
        
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
            print('filtered ', file)
    
        if def_file == None:
            if len(file) > 1 and warnings:
                print(f'Warning: Found more than one file for {name}:')
                for idx, t in enumerate(file):
                    print(f'\t{idx} : {t}')
                    
                select=input('Select one or more by index: ').split()
                file = [file[int(i)] for i in select]
            print('No default', file)
        
        else:
            select = []
            for def_f in def_file:
                for f in file:
                    if re.search(def_f, f):
                        select.append(f)
            if len(select) > 0:
                file = select
                #print('Selected ', file)
            else:
                raise FileNotFoundError(f'Def file(s) not found:', def_file)
                 
        to_load = []
        for f in file:
            file_format=str(f).split('.')[-1]
            if (file_format in Trajectory.trj_types or Trajectory.top_types):
                to_load.append(f)
                    
        #print('to load', to_load)
        return to_load
        

    @staticmethod
    def findTrajectory(workdir):
        
        trajectories = []
        for trj_type in Trajectory.trj_types:
            files=glob.glob(f'{workdir}/*.{trj_type}')
            for f in files:    
                trajectories.append(f)
        
        #print(trajectories)
        return trajectories
                                
    @staticmethod
    def findTopology(workdir, ref_name, input_topology=None):


        topologies = []
        files=glob.glob(f'{workdir}/{ref_name}')
        if len(files) > 0:
            return files[0]
        
        else:
            #print(f'Reference structure {workdir}/{ref_name} not found')
            for top_type in Trajectory.top_types:
                files=glob.glob(f'{workdir}/*.{top_type}')
                for f in files:
                    topologies.append(f)       
            if len(topologies) == 0:
                topologies = input_topology

            #print('Defined topology: ', topologies)
            return topologies

    
    @staticmethod
    def loadTrajectory(system_specs, specs):

        clock_start=process_time_ns()   
        traj = None
        
        (trajectories, topologies, results_folder, name, _)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs 

        #print('Trajectory: ', trajectories)
        #print('Topology: ', topologies)
        if len(topologies) > 1:
            #print('Warning! more than one topology for ', name, topologies)
            topology = topologies[0] #[int(input('Which one (index)?'))]
        else:
            topology = topologies[0]
# =============================================================================
#             idx_top = int(input('which one (index)?'))
#             topology = topologies[idx_top]
# =============================================================================
                    
        trj_formats=[t.split('.')[1] for t in trajectories]
        for trj_format in set(trj_formats):
            #for topology in topologies:
            try:    
                if task == 'MDTraj':
                    if trj_format == 'h5':
                        traj=md.load(trajectories, top=topology)
                        print('loading h5 for', name)
                    else:  
                        ref_top_indices=md.load(topology).topology.select(subset)
                        traj=md.load(trajectories, top=topology, atom_indices=ref_top_indices)
                    
                    frames = traj.n_frames
                elif task == 'MDAnalysis':
                    if trj_format == 'xtc':
                        traj=mda.Universe(topology, trajectories, continuous=True)
                    elif trj_format != 'h5':
                        traj=mda.Universe(topology, trajectories)
                    frames = len(traj.trajectory)
                print(f'Load time for {name} ({frames} frames): {np.round((process_time_ns() - clock_start)*1e-9, decimals=3)} s.')
                
            except Exception as v:
                print('\tWarning! Trajectory not loaded for',  name)
                print(v.__class__, v)
                                 

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
        
        
        #sel_frames=filtered.loc[(filtered.values == value)]
        #sel_frames = df.loc[(df.values == value)]
        df.reset_index(drop=True, inplace=True)
        sel_frames = df.loc[(df.values == value)]
        if not sel_frames.empty:
            print(sel_frames)

        frames=sel_frames.index.values #np.array(list(sel_frames.index.values))

        return frames
    
    