# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:34:14 2021

@author: hcarv
"""

#SFv2
import MSM
try:
    import Trajectory
    import Discretize
    import tools_plots
    import tools
except:
    pass

import matplotlib.gridspec as gridspec
import functools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
import pickle
import time

import glob
#import holoviews as hv
#hv.extension('bokeh')
#pd.options.plotting.backend = 'holoviews'
#import hvplot.pandas as hvplot

import MDAnalysis.analysis.distances as D


import mdtraj as md
from simtk.unit import picoseconds






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
    
    def __init__(self,
                 project,
                 timestep=None, 
                 results=None,
                 systems=None,
                 warnings=False,
                 filter_water=True,
                 overwrite=False,
                 def_top=None,
                 def_traj=None):
        """
        

        Parameters
        ----------
        systems : TYPE
            DESCRIPTION.
        timestep : TYPE
            DESCRIPTION.
        results : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.project = project
        if systems is not None:
           self.systems =systems 
        else:
            self.systems=project.systems
        self.scalars={}
        self.features={}
        #self.levels=self.systems.getlevels()
        if timestep is None:
            self.timestep=self.project.timestep
        else:
            self.timestep = timestep
        self.unit='ps'
        self.warnings=warnings 
        self.filter_water=filter_water
        self.subset='not water'
        self.overwrite = overwrite
        self.def_top= def_top
        self.def_traj = def_traj
        self.results = tools.Functions.pathHandler(self.project, results)
        self.save_df = False
        
# =============================================================================
#         result_paths = []
#         for name, system in self.systems.items(): 
#             result_paths.append(system.project_results)
#         if results is not None:
#             if len(list(set(result_paths))) == 1:
#                 result_path = list(set(result_paths))[0]
#                 self.results = os.path.abspath(f'{result_path}') 
#             else:
#                 self.results=os.path.abspath(f'{self.project.results}')
#                 print('Warning! More than one project result paths defined. reverting to default.')
#         else:
#             self.results = results
# =============================================================================
        self.stored=os.path.abspath(f'{self.results}/storage')
        print('Results will be stored under: ', self.results)
        if def_traj != None or def_top != None:
            print(f'Using pre-defined trajectory {self.def_traj} and/or topology {self.def_top}')
        tools.Functions.fileHandler([self.results, self.stored])
 

    def calculate(self, 
                inputs,
                method=None,
                start=0,
                stop=-1,
                stride=1,
                n_cores=-1,
                feature_name=None,
                equilibration=False,
                production=True,
                load=True,
                subset=None,
                shells = None,
                labels = None):
        """
        Wrapper function to calculate features based on "method" and "feature". 
        
        The featurization hyperparameters "start", "stop", "stride" can be specified.

        
        Increase "n_cores" to speed up calculations.
        Retrieves the output array and corresponding dataframe.
        Recieves instructions from multiprocessing calls.
        Makes calls to "<method>_calculation" and "tools.Tasks.parallel_task".
        
        
        Notes
        -----
        
        
        Increasing *n_cores* is RAM intensive as each task loads its own simulation.
        Operation is adjusted for multiprocessing calls, hence the requirement for tuple manipulation.
        The default hyparameters are set to full simulation length.  
        This makes tasks resource expensive, but allow further operators to access subsets the full simulation.
        If known optimal simulation length is know beforehand, change featurization hyperparameters accordingly.
        
        
        Parameters
        ----------


        inputs : list of tuples
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
            A dataframe containing all featurized values across the list of iterables*replicas*pairs.

        """
        self.shells = shells
        self.labels = labels
        
        if subset == None:
            subset = self.project.parameter
        else:
            for s in subset:
                if not s in self.project.parameter:
                    print('Subset not defined under Project parameters')
                    raise ValueError()
            
        supra_df = pd.DataFrame()
        timestep_=''.join(str(self.timestep).split(' '))
        df_name=f'{method}_{feature_name}_{start}_{stop}_{timestep_}_{stride}.csv'

        
        #System looper
        systems_specs=[]
        
        for name, system in self.systems.items():
            if system.parameter in subset:
                file= glob.glob(f'{system.results_folder}/{df_name}')
                if len(file) and not self.overwrite:                    
                    print(f'Loading {system.name}', end='\r')
                    out_df = pd.read_csv(file[0], index_col=0, header=Discretize.Discretize.headers['combinatorial']) #[0,1,2,3,4,5,6])
                    #os.rename(file, f'{system.results_folder}/{df_name}')
                    supra_df = pd.concat([supra_df, out_df], axis=1)

                else:
                    print(f'Setting {system.name} for calculation', end='\r')
                    #self.scalars[name] = system.scalar
                    results_folder=system.results_folder
                    topology=Trajectory.Trajectory.fileFilter(name, 
                                                                system.topology, 
                                                                equilibration, 
                                                                production, 
                                                                def_file=self.def_top,
                                                                warnings=self.warnings, 
                                                                filterW=self.filter_water)
    
                    trajectory=Trajectory.Trajectory.fileFilter(name, 
                                                                system.trajectory, 
                                                                equilibration, 
                                                                production,
                                                                def_file=self.def_traj,
                                                                warnings=self.warnings, 
                                                                filterW=self.filter_water)
                    systems_specs.append((trajectory, topology, results_folder, name, df_name))
        #Spawn multiprocessing tasks
        if len(systems_specs):
            print(f'Calculating:  {feature_name} : {len(systems_specs)}')
            method_function, _, unit_x, unit_y, task=self.getMethod(method) 
            out_df=pd.DataFrame()  
            measurements=(inputs, start, stop, self.timestep, stride, unit_x, unit_y, task, self.filter_water, self.subset)
            if n_cores != 1:
                data_t=tools.Tasks.parallel_task(method_function, systems_specs, measurements, n_cores=n_cores)
            else:
                for system_specs in systems_specs:
                    method_function(system_specs, measurements)
        
        print(supra_df)
        if self.save_df:
            supra_df.to_csv(f'{self.results}/{df_name}')
        if feature_name != None:
            self.features[f'{method}_{feature_name}']=supra_df
        else:
            self.features['{method}_undefined']=supra_df

        print('Featurization updated: ', [feature for feature in self.features.keys()])

        return supra_df



    def getMethod(self, method):
        
        #TODO: revert to simtk.unit?
        # {method name : function, kind, xy_units(tuple), package}
        methods_hyp= {'RMSD' : (Featurize.rmsd_calculation, 'global', ('ps', 'nm'), 'MDTraj'),
                     'RMSF' : (Featurize.rmsf_calculation, 'by_element', ('Index', 'nm'), 'MDTraj'),
                     'RDF' : (Featurize.rdf_calculation, 'by_element', (r'$G_r$', r'$\AA$'), 'MDAnalysis'), 
                     'distance' : (Featurize.distance_calculation, 'global', ('ps', r'$\AA$'), 'MDAnalysis'),
                     'dNAC' : (Featurize.nac_calculation, 'global', ('ps', r'$\AA$'), 'MDAnalysis'),
                     'test' : (Featurize.test, None, (None, None), 'MDAnalysis'),
                     'dNAC_combinatorial_onTheFly' : (Featurize.dNAC_combinatorial_onTheFly, 'global', (self.shells, self.labels), 'MDAnalysis')} #('ps', r'$\AA$')
        
        try:
            method_function = methods_hyp[method][0]
            kind = methods_hyp[method][1]
            unit_x, unit_y = methods_hyp[method][2][0], methods_hyp[method][2][1]
            task = methods_hyp[method][3]
            print(f'Selected method: {method} using the function "{method_function.__name__}"')

            return method_function, kind, unit_x, unit_y, task 
           
        except:
            
            print(f'Method {method} not supported.')
            print('Try: ', [method for method in methods_hyp.keys()])


    def clustering(self,
                   inputs='name CA',
                   start=0,
                   stop=-1,
                   level=2,  
                   stride=1,
                   clusters=10,
                   rmsd=False,
                   equilibration=False,
                   production=True,
                   def_top=None,
                   def_traj=None):
        
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
                
            file_concat=os.path.abspath(f'{self.results}/structures/superposed_{level_unique}-s{stride}')
            tools.Functions.fileHandler([file_concat], confirmation=False, _new=False)
            trj=f'{file_concat}.xtc'
            top=f'{file_concat}.pdb'
            
            files_to_concatenate=[]
            
            #Iterate though all systems and proceed if level match
            for name, system in self.systems.items(): 
                #print(name)
                
                topology=Trajectory.Trajectory.fileFilter(name, 
                                                        system.topology, 
                                                        equilibration, 
                                                        production, 
                                                        def_file=def_top,
                                                        warnings=self.warnings, 
                                                        filterW=self.heavy_and_fast)

                trajectory=Trajectory.Trajectory.fileFilter(name, 
                                                        system.trajectory, 
                                                        equilibration, 
                                                        production,
                                                        def_file=def_traj,
                                                        warnings=self.warnings, 
                                                        filterW=self.heavy_and_fast)
                
                
                if name.split('-')[level] == level_unique:
                    files_to_concatenate.append((trajectory, topology))            
            
            #print(f'Files to concatenate: {files_to_concatenate}')
            
            if len(files_to_concatenate) != 0:  
                
                if type(top) == list:
                    print('Warning! top is list', top)
                    top = top[0]
                superposed=Trajectory.Trajectory.concatenate_superpose(files_to_concatenate,
                                                                       start=start,
                                                                       stop=stop,
                                                                       atom_set=inputs, 
                                                                       trajectory=trj, 
                                                                       topology=top, 
                                                                       stride=stride)

                cluster=Trajectory.Trajectory.clusterMDAnalysis(file_concat, 
                                                                select=inputs, 
                                                                n_clusters_=clusters, 
                                                                trajectory=trj, 
                                                                topology=top)
       
                collect[level_unique]=(superposed, cluster) 
                #print(collect[level_unique])   
            else:
                print('\tNo frames found.')


        return collect



    @staticmethod
    def rdf_calculation(system_specs, specs):
        """
        

        Parameters
        ----------
        system_specs : TYPE
            DESCRIPTION.
        specs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        import MDAnalysis.analysis.rdf as RDF
        
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        indexes=[[n] for n in name.split('-')]
        names=[f'l{i}' for i in range(1, len(indexes)+1)]
        print(indexes, names)
        column_index=pd.MultiIndex.from_product(indexes, names=names)
                
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
    
        if traj != None:
            
            ref, sel = traj.select_atoms(selection[0]), traj.select_atoms(selection[1])
            #print(ref, sel)
        
            rdf_=RDF.InterRDF(ref, sel)
            rdf_.run(start, stop, stride)
            
            rows=pd.Index(rdf_.bins, name=units_x)
            df_system=pd.DataFrame(rdf_.rdf, columns=column_index, index=rows)
            #df_system=df_system.mask(df_system > 90)
            
            #print(df_system)
            
            return df_system
    
        else:
            return pd.DataFrame()
    
    @staticmethod 
    def rmsd_calculation(system_specs, specs):
        """
        

        Parameters
        ----------
        system_specs : TYPE
            DESCRIPTION.
        specs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        (trajectory, topology, results_folder, name, df_name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs  
  
        
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        
        if traj != None and topology != None:
            
            if type(selection) is tuple:
                other_traj=md.load(selection[1])
                atom_indices=traj.topology.select(selection[0])
                other_atom_indices=other_traj.topology.select(selection[0])
                common_atom_indices=list(set(atom_indices).intersection(other_atom_indices))
                try:
                    traj.image_molecules(inplace=True)
                except ValueError:
                    pass
                    #print('could not image traj')
                traj.atom_slice(common_atom_indices, inplace=True)
                
                other_traj.image_molecules(inplace=True)
                other_traj.atom_slice(common_atom_indices, inplace=True)
                rmsd=md.rmsd(traj[start:stop:stride],
                             other_traj,
                             frame=0,
                             precentered=False,
                             parallel=False) #TODO: make flexible if single run
            
            else:
                try:
                    traj.image_molecules(inplace=True)
                except:
                    pass
                    #print('could not image traj')
                atom_indices=traj.topology.select(selection)
                traj.atom_slice(atom_indices, inplace=True)
                
                #traj[start:stop:stride].center_coordinates()
                #traj.image_molecules(inplace=True, anchor_molecules=[atom_indices])
                rmsd=md.rmsd(traj[start:stop:stride], 
                         traj, 
                         frame=0,
                         precentered=False,
                         parallel=False)  




            rows=pd.Index(np.arange(0, len(rmsd), stride)*timestep, name=units_x)
            df_system=pd.DataFrame(rmsd, index=rows)
            
            return df_system
    
        else:
            
            return pd.DataFrame()


    @staticmethod
    def rmsf_calculation(system_specs, specs):
        """
        

        Parameters
        ----------
        system_specs : TYPE
            DESCRIPTION.
        specs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        names, indexes, column_index=Featurize.df_template(system_specs, unit=[units_y])
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
    
        if traj != None:
            
            
            atom_indices=traj.topology.select(selection)
            traj.atom_slice(atom_indices, inplace=True)
            traj.center_coordinates()
            rmsf=md.rmsf(traj[start:stop:stride], 
                         traj[start:stop:stride], 
                         0,
                         precentered=True)

            rows=rows=pd.Index(np.arange(0, len(atom_indices)), name='Index')
            column_index=pd.MultiIndex.from_product(indexes, names=names)
            
            df_system=pd.DataFrame(rmsf, columns=column_index, index=rows) #index=np.arange(start, stop, stride)
            #df_system.index.rename ='Index'
            
            #Remove non-sense
            #df_system=df_system.mask(df_system > 90)
            
            return df_system
    
        else:
            return pd.DataFrame()


    @staticmethod
    def sasa_calculation(system_specs, specs):
        """
        

        Parameters
        ----------
        system_specs : TYPE
            DESCRIPTION.
        specs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        names, indexes, column_index=Featurize.df_template(system_specs, unit=[units_y])
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
    
        if traj != None:
            
            atom_indices=traj.topology.select(selection)
            sasa=md.shrake_rupley(traj[start:stop:stride], 
                                  traj[start:stop:stride], 
                                  mode='residue')
            print(sasa)
            rows=rows=pd.Index(np.arange(0, len(sasa)), name='Index')
            column_index=pd.MultiIndex.from_product(indexes, names=names)
            
            df_system=pd.DataFrame(sasa, columns=column_index, index=rows) #index=np.arange(start, stop, stride)
            #df_system.index.rename ='Index'
            
            #Remove non-sense
            #df_system=df_system.mask(df_system > 90)
            
            return df_system
    
        else:
            return pd.DataFrame()

    @staticmethod
    def test(system_specs, specs=()):
        
        
        (trajectory, topology, results_folder, name, df_name)=system_specs
        (selections,  start, stop, timestep, stride, shells, labels, task, store_traj, subset)=specs
        units_y = r'$\AA$'
        #names, indexes, column_index=Featurize.df_template(system_specs, unit=[units_y])
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        if traj != None and topology != None:
            for selection in selections:
                ref, sel = traj.select_atoms(selection[0]), traj.select_atoms(selection[1]) 
                for r in ref:
                    print(r)
                print(f'\tReference: {ref[0]} x {len(ref)}')
                print(f'\tSelection: {sel[0]} x {len(sel)}')


    def distance_calculation(self, system_specs, specs): #system_specs, measurements):
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
 
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        names, indexes, column_index=Featurize.df_template(system_specs, unit=[units_y])
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        
        
        if traj != None and topology != None:
            
            ref, sel = traj.select_atoms(selection[0]), traj.select_atoms(selection[1]) 
            
            print(f'\tReference: {ref[0]} x {len(ref)}')
            print(f'\tSelection: {sel[0]} x {len(sel)}')
            
            names, indexes, column_index=Featurize.df_template(system_specs, unit=[r'$\AA$'], multi=True, multi_len=len(ref)*len(sel))
            
            dist=[D.distance_array(ref.positions, sel.positions, box=traj.dimensions) for ts in traj.trajectory[start:stop:stride]] 
            
            #TODO: Check behaviour for ref > 1
            frames, sel, ref=np.shape(dist)
            dist_reshape=np.asarray(dist).reshape(frames, ref*sel)
            
            print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}')
            print(f'\tLimits: {np.round(np.min(dist_reshape), decimals=2)} and {np.round(np.max(dist_reshape), decimals=2)}')
        
            rows=pd.Index(np.arange(start, frames)*(stride*timestep), name=self.unit)
            df_system=pd.DataFrame(dist_reshape, columns=column_index, index=rows)
            #df_system=df_system.mask(df_system > 90)
            
            #print(df_system)
            
            return df_system
    
        else:
            return pd.DataFrame()
      
    
    
    @staticmethod
    @tools.log
    def dNAC_combinatorial_onTheFly(system_specs, specs=(), shells=None, labels=None):
        
        (trajectory, topology, results_folder, name, df_name)=system_specs
        (selection,  start, stop, timestep, stride, shells, labels, task, store_traj, subset)=specs   
        units_y = r'$\AA$'

        def run(frame):
            #print('frame ', frame, end='\r') 
            t0 = time.clock()
            distances_ = []
            for distance in distances:        
                sel_, ref_ = distance[0], distance[1]  
                dist=D.distance_array(ref_.positions, sel_.positions, box=traj.dimensions)
                distances_.append(dist)

            #distances=np.asarray(distances) 

            dO = distances_[0]
            dH1 = distances_[1]
            dH2 = distances_[2]

            nac_d = [dO, np.minimum(dH1, dH2)]
            nac_array = np.around(np.sqrt(np.power(nac_d, 2).sum(axis=0)/len(nac_d)), 3) 
            
            
            
            hist_subunits = np.empty([len(ref), len(feature_range)-1])
            
            
            df = pd.DataFrame(nac_array.T)
            
            states=tools.Functions.state_mapper(shells, df, labels=labels)
            states_total_frames[frame,:] = states
            
            for (idx_sub, subunit) in df.iterrows():
                hist, _ = np.histogram(subunit, bins=feature_range) 
                hist_subunits[idx_sub,:] = hist

            hist_total_frames[frame,:] = hist_subunits 
            t1 = time.clock()
            #print(frame, t1-t0, end='\r')
            

        print(name)
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        
        fig=plt.figure(figsize=(10,4))
        gs = gridspec.GridSpec(nrows=1, ncols=3)
        plot_hist = fig.add_subplot(gs[0, 0]) 
        plot_states = fig.add_subplot(gs[0, 1:]) 
        
        limits=(0,150)
        thickness = 0.25
        feature_range=np.arange(limits[0], limits[1], thickness)
        feature_center_bin=feature_range+(thickness/2) 
        
        distances = []
        for selection_ in selection:        
            sel, ref = traj.select_atoms(selection_[0]), traj.select_atoms(selection_[1])  
            if not len(ref) or not len(sel):
                #print(f'\tReference: {ref[0]} x {len(ref)}')
                #print(f'\tSelection: {sel[0]} x {len(sel)}')
                print(f'Warning on {name}! No atoms selected:\n\tRef: {ref}\n\tSel: {sel}')
                sel, ref = traj.select_atoms(input('Which sel ?')), traj.select_atoms(input('Which ref ?'))
            distances.append((ref, sel))


        n_frames = len(traj.trajectory[start:stop:stride])

        shells_name=''.join(labels)
        base_name = f'{df_name.split(".")[0]}.npy'
        states_name = f'{results_folder}/states_water_{shells_name}_b{start}_e{stop}_s{stride}.npy'
        hist_name = f'{results_folder}/histogram_water_b{start}_e{stop}_s{stride}.npy'
        if not os.path.exists(hist_name) or not os.path.exists(states_name):
            print('Calculating: ', name)
            hist_total_frames = np.empty([n_frames, len(ref), len(feature_range)-1])
            states_total_frames = np.empty([n_frames, len(ref)])
            [run(frame) for frame, _ in enumerate(traj.trajectory[start:stop:stride])]
            np.save(states_name, states_total_frames)
            hist_final= hist_total_frames.sum(axis=0)
            np.save(hist_name, hist_final)
        else:
            print('Loading: ', hist_name)
            hist_final = np.load(hist_name)
            states_total_frames = np.load(states_name)
        
        
        
        #print(hist_final, np.shape(hist_final))
        #print(states_total_frames, np.shape(states_total_frames))
        for sub in hist_final:
            plot_hist.plot(feature_center_bin[:-1], sub)
        plot_hist.set_xscale('log')
        plot_hist.set_yscale('log')
        
        for sub in states_total_frames.T:
            plot_states.plot(np.arange(0, len(sub)), sub, '.')
        plt.show()
        
        
        names, indexes, column_index, rows=Featurize.df_template(name,
                                                                 (start, stride, timestep, 'ps'),
                                                                 (n_frames, len(sel), len(ref)),
                                                                 sel_range=True)
        
        df_hist=pd.DataFrame(hist_final.T.astype(int), columns=column_index, index=feature_center_bin[:-1])
        df_hist.to_csv(f'{results_folder}/{df_name}')
        
        df_states = pd.DataFrame(states_total_frames.astype(int), columns=column_index, index=rows)
        df_states.to_csv(f'{results_folder}/discTraj_{df_name}')
        print(df_states)
        print(df_hist, df_states)
                
    
    @staticmethod
    @tools.log
    def nac_calculation(system_specs, specs=()):
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

        (trajectory, topology, results_folder, name, df_name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   

        print(name)
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        distances = []
        for selection_ in selection:        
            ref, sel = traj.select_atoms(selection_[0]), traj.select_atoms(selection_[1])  
            if not len(ref) or not len(sel):
                #print(f'\tReference: {ref[0]} x {len(ref)}')
                #print(f'\tSelection: {sel[0]} x {len(sel)}')
                print(f'Warning on {name}! No atoms selected:\n\tRef: {ref}\n\tSel: {sel}')
                ref_ = input('Which ref ?')
                sel_ = input('Which sel ?')
                ref, sel = traj.select_atoms(ref_), traj.select_atoms(sel_)
            dist=[D.distance_array(ref.positions, sel.positions, box=traj.dimensions) for ts in traj.trajectory[start:stop:stride]]             
            distances.append(dist)

            #print(f'\t{name} {ref[0]}: {np.round(np.min(dist), decimals=2)} and {np.round(np.max(dist), decimals=2)}')
        
        if len(distances) == len(selection):
            distances=np.asarray(distances)  
            #SQRT(SUM(di^2)/#i)
            nac_array=np.around(np.sqrt(np.power(distances, 2).sum(axis=0)/len(distances)), 3) #sum(axis=0)  
    
            frames, sel, ref=np.shape(nac_array)
            dist_reshape=np.asarray(nac_array).reshape(frames, ref*sel)
            print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}')
            print(f'\tLimits: {np.round(np.min(dist_reshape), decimals=2)} and {np.round(np.max(dist_reshape), decimals=2)}')
    
            names, indexes, column_index, rows=Featurize.df_template(name,
                                                               (start, stride, timestep, 'ps'),
                                                               (frames, sel, ref), 
                                                               unit=[units_y])
            
            
            
            df_system=pd.DataFrame(dist_reshape, columns=column_index, index=rows)
            df_system.to_csv(f'{results_folder}/{df_name}')
    
        else:
            print('No df for ', name)
            df_system = pd.DataFrame()
        
        return df_system
        

    @staticmethod
    def df_template(name, traj_tuple, dims_tuple, unit=None, sel_range=False):
        """
        

        Parameters
        ----------
        system_specs : TYPE
            DESCRIPTION.
        unit : TYPE, optional
            DESCRIPTION. The default is ['nm'].
        multi : TYPE, optional
            DESCRIPTION. The default is False.
        multi_len : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        names : TYPE
            DESCRIPTION.
        indexes : TYPE
            DESCRIPTION.
        column_index : TYPE
            DESCRIPTION.

        """

        (start, stride, timestep, unit_) = traj_tuple
        (frames, sel, ref) = dims_tuple
        print('WARNING! not giving right for ternary mixtures. one level is added')
        #TODO: Fix this for differente "separator" value. This is now hardcorded to '-'.
        indexes=[[n] for n in name.split('-')]

        refs=[f'ref_{i}' for i in range(1, ref+1)]
        indexes.append(refs)
        
        if not sel_range:
            sels=[f'sel_{i}' for i in range(1, sel+1)]
            indexes.append(sels)
        else:
            indexes.append([f'sel_1-{sel}'])
        
        if unit != None:
            indexes.append(unit)
        
        names=[f'l{i}' for i in range(1, len(indexes)+1)]
        column_index=pd.MultiIndex.from_product(indexes, names=names)
        
        rows=pd.Index(np.arange(start, frames)*(stride*timestep), name=unit_)
        
        
        return names, indexes, column_index, rows



    
    def plot(self, 
             input_df=None, 
             method=None,
             feature_name=None,
             level=2, 
             subplots_=True,
             stats=True):
        """
        General function to print *Featurized* pandas dataFrame, either stored under a Featurize instance or external input.
        Takes as input the level referring to the Project *hierarchy* ontology. 
        Alternatively, it will extract data based on multi-index definitions of provided dataFrame.
        Generates statistical plots of the *level*.
        Generates subplots of sublevel.

        Parameters
        ----------
        input_df : TYPE, optional
            DESCRIPTION. The default is None.
        method : TYPE, optional
            DESCRIPTION. The default is 'RMSD'.
        feature_name : TYPE, optional
            DESCRIPTION. The default is None.
        level : TYPE, optional
            DESCRIPTION. The default is 2.
        subplots_ : TYPE, optional
            DESCRIPTION. The default is True.
        stats : TYPE, optional
            DESCRIPTION. The default is True.
                
        
        Example
        -------
        When Project.hierarchy=['protein', 'ligand', 'parameter']
        Using level=2 will plot *ligand* stats and sublevel *parameter* plots.
        

        Returns
        -------
        None.
        

        """
        
        
        try:
            input_df=self.features[f'{feature_name}']
            print(f'Feature {feature_name} found.')

        except KeyError:
            
            input_df=input_df
            print(f'Feature {feature_name} not found. Using input DataFrame.')
        
        try:
            method=input_df.name
            function, kind, _, _,_=self.getMethod(method)
        except:
            function, kind, _, _,_=self.getMethod(method)

        levels=input_df.columns.levels #might need to subselect here
        units=levels[-1].to_list()[0]
        
        #Set stats
        level_iterables=input_df.columns.get_level_values(f'l{level}').unique()
        sublevel_iterables=input_df.columns.get_level_values(f'l{level+1}').unique()  
        
        
        #plot level
        rows, columns=tools_plots.plot_layout(level_iterables)
        fig_,axes_=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        
        
        try:
            flat=axes_.flat
        except AttributeError:
            flat=axes_
            
            flat=[flat]
        
        for sup_it, ax_ in zip(level_iterables, flat): 
                
            sup_df=input_df.loc[:,input_df.columns.get_level_values(f'l{level}') == sup_it]  
                
            #Plot sublevel
            rows_, columns_=tools_plots.plot_layout(sublevel_iterables)
            fig_it,axes_it=plt.subplots(rows_, columns_, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
            
            try:
                axes_it.flat
            except:
                
                axes_it=np.asarray(axes_it)
                
            for iterable, ax_it in zip(sublevel_iterables, axes_it.flat):
                
                #level +1 to access elements below in hierarchy
                df_it=sup_df.loc[:,sup_df.columns.get_level_values(f'l{level+1}') == iterable]
                title_=f'{method}: {feature_name}'
                #if kind == 'global':
                #     df_it=df_it.min(axis=1)
                #     title_ = f'{feature_name}(min)'
                df_it.plot(kind='line', 
                       subplots=subplots_, 
                       sharey=True,  
                       figsize=(9,6), 
                       legend=False, 
                       sort_columns=True,
                       linewidth='1',
                       ax=ax_it)

                 #{df_it.index.values[0]} to {df_it.index.values[-1]} {df_it.index.name}'
                #print(units)
                ax_it.set_xlabel(df_it.index.name)
                ax_it.set_ylabel(f'{method} ({units})')
            
                #print(df_it)
            
                #Print sublevel into level
                if kind == 'by_time':
                 
                    mean=df_it.mean()
                    mean.plot()
                 
                elif kind == 'by_element':
                    
                    mean=df_it.mean(axis=1)
                    std=df_it.std(axis=1).values
                    ax_.plot(mean.index.values, mean)
                    ax_.fill_between(mean.index.values, mean + std, mean - std, alpha=0.3)
                         
                    mean, std_lower, std_upper=mean, mean.values-std, mean.values+std
                         
                    #print(std_lower, std_upper)
                    if not (np.isnan(std_lower).any() and np.isnan(std_upper).any()):
                        ax_.fill_between(mean, std_lower, std_upper, alpha=0.8)
                             
                    ax_.set_xlabel(input_df.index.name)
                    ax_.set_ylabel(f'{method} ({units})')
                    
                elif kind == 'global':
                        
                    sns.distplot(df_it.to_numpy().flatten(),
                                     ax=ax_,
                                     hist = False, 
                                     kde = True,
                                     kde_kws = {'shade': True, 'linewidth': 2})
                    #ax_.set_xscale('log')
                    ax_.set_yscale('log')
                        
                    #ax_.hist(sup_df.to_numpy
                ax_.set_xlabel(f'{method} ({levels[-1].to_list()[0]})')
                ax_.set_title(f'{method}: {sup_it}')
              
        

            fig_it.suptitle(title_)
            fig_it.show()
            fig_it.savefig(os.path.abspath(f'{self.results}/{method}_{feature_name}_sub_l{level+1}-{sup_it}.png'), bbox_inches="tight", dpi=600)
         
        fig_.legend(sublevel_iterables)
        #fig_.show()
        fig_.savefig(os.path.abspath(f'{self.results}/{method}_{feature_name}_l{level}_stats.png'), bbox_inches="tight", dpi=600)  


        return plt.show()


    