# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:34:14 2021

@author: hcarv
"""

#SFv2
import MSM
try:
    import Trajectory
    
    import tools_plots
    import tools
except:
    pass


import functools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
import pickle

import MDAnalysis as mda
import MDAnalysis.analysis.distances as D


import mdtraj as md
from simtk.unit import picoseconds


def calculator(func):
     
     @functools.wraps(func)
     def wrapper(system_specs, specs):
         
        print('Calculator')
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        names, indexes, column_index=Featurize.Featurize.df_template(system_specs, unit=[units_y])
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        

        if traj != None and topology != None:
            
            result=func(selection, traj, start, stop, stride)
            rows=pd.Index(np.arange(0, len(result), stride)*timestep, name=units_x)
            df_system=pd.DataFrame(result, columns=column_index, index=rows)
    
        else:
            df_system = pd.DataFrame()
        
        return df_system
 
     return wrapper



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
                 systems, 
                 timestep, 
                 results,
                 warnings=False,
                 heavy_and_fast=True):
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
        
        self.systems=systems
        self.scalars={}
        self.results=results
        self.features={}
        #self.levels=self.systems.getlevels()
        self.timestep=timestep
        self.unit='ps'
        self.warnings=warnings 
        self.heavy_and_fast=heavy_and_fast
        self.subset='not water'
        
        print('Results will be stored under: ', self.results)
 
        

        
 

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
                def_top=None,
                def_traj=None):
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
        #Get function, kind (optional), xy units, task
        method_function, _, unit_x, unit_y, task=self.getMethod(method) 
 
        #Pack fixed measurement hyperparameters
        measurements=(inputs, start, stop, self.timestep, stride, unit_x, unit_y, task, self.heavy_and_fast, self.subset)
        
        # Define system definitions and measurement hyperparameters
        #Note: system cannot be sent as pickle for multiproc, has to be list
        systems_specs=[]

        for name, system in self.systems.items():
            
             #input_topology
            self.scalars[name] = system.scalar
            results_folder=system.results_folder
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
               
            systems_specs.append((trajectory, topology, results_folder, name))
        
        

        
        #Spawn multiprocessing tasks    
        data_t=tools.Tasks.parallel_task(method_function, systems_specs, measurements, n_cores=n_cores)
        
        
        
        # concat dataframes from multiprocessing into one dataframe
        #Update state of system and features with the feature df.
        out_df=pd.DataFrame()
            
        for data_df, system in zip(data_t, self.systems.values()):
            system.data[f'{method}-{feature_name}']=data_df
            system.features[f'{method}-{feature_name}']=inputs
            out_df=pd.concat([out_df,  data_df], axis=1)
        out_df.index.rename=data_df.index.name
        out_df.to_csv(os.path.abspath(f'{self.results}/{method}_{feature_name}.csv'))
    
        if feature_name != None:
            self.features[f'{feature_name}']=out_df
        else:
            self.features['undefined']=out_df
                
        out_df.name=method

        print('Featurization updated: ', [feature for feature in self.features.keys()])

        return out_df


    def getMethod(self, method):
        
        #TODO: revert to simtk.unit?
        # {method name : function, kind, xy_units(tuple), package}
        methods_hyp= {'RMSD' : (Featurize.rmsd_calculation, 'global', ('ps', 'nm'), 'MDTraj'),
                     'RMSF' : (Featurize.rmsf_calculation, 'by_element', ('Index', 'nm'), 'MDTraj'),
                     'RDF' : (Featurize.rdf_calculation, 'by_element', (r'$G_r$', r'$\AA$'), 'MDTraj'), 
                     'distance' : (Featurize.distance_calculation, 'global', ('ps', r'$\AA$'), 'MDAnalysis'),
                     'dNAC' : (Featurize.nac_calculation, 'global', ('ps', r'$\AA$'), 'MDAnalysis')} 
        
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
        (selection,  start, stop, timestep, stride, units_x, units_y)=specs   
        
        names, indexes, column_index=Featurize.df_template(system_specs, unit=[units_y])
        
        task='MDAnalysis'
                
        traj=Trajectory.Trajectory.loadTrajectory(topology, trajectory, task)
    
        if traj != None:
            
            ref, sel = traj.select_atoms(selection[0]), traj.select_atoms(selection[1])
        
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
        
        
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        names, indexes, column_index=Featurize.df_template(system_specs, unit=[units_y])
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
                traj.atom_slice(atom_indices, inplace=True)
                
                other_traj.image_molecules(inplace=True)
                other_traj.atom_slice(other_atom_indices, inplace=True)
                
                
                
                rmsd=md.rmsd(traj[start:stop:stride],
                             other_traj,
                             frame=0,
                             precentered=False,
                             parallel=False)
            
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
            df_system=pd.DataFrame(rmsd, columns=column_index, index=rows)
            
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
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        
        if traj != None and topology != None:
            
            distances = []
            for selection_ in selection:        
                ref, sel = traj.select_atoms(selection_[0]), traj.select_atoms(selection_[1])            
                #print(f'\tReference: {ref[0]} x {len(ref)}')
                #print(f'\tSelection: {sel[0]} x {len(sel)}')          
                dist=[D.distance_array(ref.positions, sel.positions, box=traj.dimensions) for ts in traj.trajectory[start:stop:stride]]             
                distances.append(dist)
                #print(f'\t{name} {ref[0]}: {np.round(np.min(dist), decimals=2)} and {np.round(np.max(dist), decimals=2)}')
                
            distances=np.asarray(distances)  
            #SQRT(SUM(di^2)/#i)
            nac_array=np.around(np.sqrt(np.power(distances, 2).sum(axis=0)/len(distances)), 3) #sum(axis=0)  
            #TODO: Check behaviour for ref > 1
            frames, sel, ref=np.shape(nac_array)
            dist_reshape=np.asarray(nac_array).reshape(frames, ref*sel)
            print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}')
            print(f'\tLimits: {np.round(np.min(dist_reshape), decimals=2)} and {np.round(np.max(dist_reshape), decimals=2)}')

            names, indexes, column_index, rows=Featurize.df_template(name,
                                                               (start, stride, timestep, 'ps'),
                                                               (frames, sel, ref), 
                                                               unit=[units_y])
            
            #rows=pd.Index(np.arange(start, frames)*(stride*timestep), name=self.unit)
            df_system=pd.DataFrame(dist_reshape, columns=column_index, index=rows)            
            #print(df_system)
            
            return df_system
    
        else:
            return pd.DataFrame()



    @staticmethod
    def df_template(name, traj_tuple, dims_tuple, unit=['nm']):
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
        
        #TODO: Fix this for differente "separator" value. This is now hardcorded to '-'.
        indexes=[[n] for n in name.split('-')]

        refs=[f'ref_{i}' for i in range(1, ref+1)]
        indexes.append(refs)
        
        sels=[f'sel_{i}' for i in range(1, sel+1)]
        indexes.append(sels)
        
        indexes.append(unit)
        
        names=[f'l{i}' for i in range(1, len(indexes)+1)]
        column_index=pd.MultiIndex.from_product(indexes, names=names)
        
        rows=pd.Index(np.arange(start, frames)*(stride*timestep), name=unit_)
        
        
        return names, indexes, column_index, rows

    @staticmethod
    def plot_hv(input_df):
        
        
        import hvplot.pandas 
        input_df.hvplot()
        
        return input_df.hvplot() 


    
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
        rows, columns, fix_layout=tools_plots.plot_layout(level_iterables)
        fig_,axes_=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        
        
        try:
            flat=axes_.flat
        except AttributeError:
            flat=axes_
            
            flat=[flat]
        
        for sup_it, ax_ in zip(level_iterables, flat): 
                
            sup_df=input_df.loc[:,input_df.columns.get_level_values(f'l{level}') == sup_it]  
                
            #Plot sublevel
            rows_, columns_, fix_layout=tools_plots.plot_layout(sublevel_iterables)
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


    