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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re

import MDAnalysis as mda
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
    
    def __init__(self, systems, timestep, results):
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
        self.results=results
        self.features={}
        #self.levels=self.systems.getlevels()
        self.timestep=timestep
        self.unit='ps'
        
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
                production=True):
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
           
        # Define system definitions and measurement hyperparameters
        #Note: system cannot be sent as pickle for multiproc, has to be list
        systems_specs=[]  
        for name, system in self.systems.items():
            
            topology=system.topology #input_topology
            results_folder=system.results_folder
            timestep=system.timestep
            trajectory=Trajectory.Trajectory.eq_prod_filter(system.trajectory, equilibration, production)

            systems_specs.append((trajectory, topology, results_folder, name))            
        
        #Get function, kind (optional), xy units 
        method_function, _, unit_x, unit_y=self.getMethod(method) 
        #TODO: to deprecate, handle by current method_dict with updated getMethod outputs
        
        #method_dict= self.getMethod_dict(method)
        
        #Pack fixed measurement hyperparameters
        measurements=(inputs, start, stop, timestep, stride, unit_x, unit_y)
        
   
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


    @staticmethod
    def getMethod(method):
        
        methods_functions={'dNAC' : Featurize.nac_calculation,
                      'distance' : Featurize.distance_calculation,
                      'RMSD' : Featurize.rmsd_calculation,
                      'RMSF' : Featurize.rmsf_calculation,
                      'RDF' : Featurize.rdf_calculation,
                      'SASA' : Featurize.sasa_calculation}
        
        
        kind={'RMSD': 'global',
              'RMSF': 'by_element',
              'RDF' : 'by_element',
              'distance' : 'global',
              'SASA' : 'by_element'}
        
        #TODO: revert to simtk.unit?
        units = {'dNAC' : ('ps', 'nm'),
                      'distance' : ('ps', 'nm'),
                      'RMSD' : ('ps', 'nm'),
                      'RMSF' : ('Index', 'nm'),
                      'RDF' : (r'$G_r$', r'$\AA$'),
                      'SASA' : ('Index', r'$\(nm)^2$')}
        
        # {method name : function, kind, xy_units(tuple)}
        methods_hyp= {'RMSD' : (Featurize.rmsd_calculation, 'global', ('ps', 'nm')),
                     'RMSF' : (Featurize.rmsf_calculation, 'by_element', ('Index', 'nm')),
                     'RDF' : (Featurize.rdf_calculation, 'by_element', (r'$G_r$', r'$\AA$')), 
                     'distance' : (Featurize.distance_calculation, 'global', ('ps', 'nm'))} 
        
        
        try:
            method_function = methods_hyp[method][0]
            kind = methods_hyp[method][1]
            print(f'Selected method: {method} using the function "{method_function.__name__}"')
            
            unit_x, unit_y = methods_hyp[method][2][0], methods_hyp[method][2][1]
           
            return method_function, kind, unit_x, unit_y 
           
        except:
            
            print(f'Method {method} not supported.')
            print('Try: ', [method for method in methods_hyp.keys()])





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
        (selection,  start, stop, timestep, stride, units_x, units_y)=specs   
        
        #print(f'\tCalculating RMSD for {name}')         
        #TODO: include reference frame as inputs.

        names, indexes, column_index=Featurize.df_template(system_specs, unit=[units_y])   
 
        task='MDTraj'
        
        traj=Trajectory.Trajectory.loadTrajectory(topology, trajectory, task)
        
        if traj != None and topology != None:
            
            if type(selection) is tuple:
                other_traj=md.load(selection[1])
                atom_indices=traj.topology.select(selection[0])
                other_atom_indices=other_traj.topology.select(selection[0])
                common_atom_indices=list(set(atom_indices).intersection(other_atom_indices))
                
                traj.atom_slice(atom_indices, inplace=True)
                other_traj.atom_slice(other_atom_indices, inplace=True)
                rmsd=md.rmsd(traj[start:stop:stride],
                             other_traj,
                             frame=0,
                             precentered=False)
            
            else:
                atom_indices=traj.topology.select(selection)
                
                traj.atom_slice(atom_indices, inplace=True)
                traj[start:stop:stride].center_coordinates()
                
                rmsd=md.rmsd(traj[start:stop:stride], 
                         traj, 
                         frame=0,
                         precentered=True)  

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
        
        names, indexes, column_index=Featurize.df_template(system_specs, unit=['nm'])
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y)=specs
        
        
        task='MDTraj'
                
        traj=Trajectory.Trajectory.loadTrajectory(topology, trajectory, task)
    
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
        
        names, indexes, column_index=Featurize.df_template(system_specs, unit=[r'$\(nm)^2$'])
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y)=specs
        
        task='MDTraj'
                
        traj=Trajectory.Trajectory.loadTrajectory(topology, trajectory, task)
    
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
    def distance_calculation(system_specs, specs): #system_specs, measurements):
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
        (selection,  start, stop, timestep, stride, unit_x, unit_y)=specs   
        
        task='MDAnalysis'
                
        traj=Trajectory.Trajectory.loadTrajectory(topology, trajectory, task)
        
        if traj != None:
            
            ref, sel = traj.select_atoms(selection[0]), traj.select_atoms(selection[1]) 
            
            #print(f'\tReference: {ref[0]} x {len(ref)}')
            #print(f'\tSelection: {sel[0]} x {len(sel)}')
            
            names, indexes, column_index=Featurize.df_template(system_specs, unit=[r'$\AA$'], multi=True, multi_len=len(ref)*len(sel))
            
            dist=[D.distance_array(ref.positions, sel.positions, box=traj.dimensions) for ts in traj.trajectory[start:stop:stride]] 
            
            #TODO: Check behaviour for ref > 1
            frames, sel, ref=np.shape(dist)
            dist_reshape=np.asarray(dist).reshape(frames, ref*sel)
            
            print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}')
            print(f'\tLimits: {np.round(np.min(dist_reshape), decimals=2)} and {np.round(np.max(dist_reshape), decimals=2)}')
        
            rows=pd.Index(np.arange(0, frames, stride)*timestep, name='ps')
            df_system=pd.DataFrame(dist_reshape, columns=column_index, index=rows)
            #df_system=df_system.mask(df_system > 90)
            
            #print(df_system)
            
            return df_system
    
        else:
            return pd.DataFrame()
      

    @staticmethod
    def df_template(system_specs, unit=['nm'], multi=False, multi_len=1):
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
        
        (trajectory, topology, results_folder, name)=system_specs
        
        #TODO: Fix this for differente "separator" value. This is now hardcorded to '-'.
        indexes=[[n] for n in name.split('-')]

        if multi:        
            pairs=[i for i in range(1, multi_len+1)]
            indexes.append(pairs)
        
        indexes.append(unit)
        names=[f'l{i}' for i in range(1, len(indexes)+1)]
        column_index=pd.MultiIndex.from_product(indexes, names=names)
        
        return names, indexes, column_index

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
            function, kind, _, _=self.getMethod(method)
        except:
            function, kind, _, _=self.getMethod(method)

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
                df_it.plot(kind='line', 
                       subplots=subplots_, 
                       sharey=True,  
                       figsize=(9,6), 
                       legend=False, 
                       sort_columns=True,
                       ax=ax_it)

                title_=f'{method}: {feature_name}' #{df_it.index.values[0]} to {df_it.index.values[-1]} {df_it.index.name}'
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
                                     kde_kws = {'shade': True, 'linewidth': 3})
                        
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




#####LEGACY

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

          
        nac_file=f'{results_folder}/dNAC_{len(distances)}-i{start}-o{stop}-s{stride}-{str(timestep).replace(" ","")}.npy'
        
        if not os.path.exists(nac_file):
             
            dists=Featurize.distance_calculation(system_specs, specs, child=True)

            #print(f'This is dists  {np.shape(dists[0])}: \n {dists[0]}')
            #print(f'This is power: \n{np.power(dists[0], 2)}')


            print(f'Calculating d_NAC of {name}')
            
            #SQRT(SUM(di^2)/#i)
            nac_array=np.around(np.sqrt(np.power(dists, 2).sum(axis=0)/len(dists)), 3) #sum(axis=0)
            
            #SQRT(MEAN(di^2))
            #nac_array=np.around(np.sqrt(np.mean(np.power(dists))), 3)
            
            print(nac_array)
            
            np.save(nac_file, nac_array)
                
                

        else:
            #print(f'dNAC file for {name} found.') 
            nac_array=np.load(nac_file)

        try:
            #TODO: Check behaviour for ref > 1
            #TODO: switch ref and sel for full.
            frames, sel, ref=np.shape(nac_array)
            
            print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}\n\t{np.min(nac_array)}, {np.max(nac_array)}')
            
            pairs=np.arange(1,sel+1)
            #print(pairs)
            indexes.append(pairs)
            #print(indexes)
            column_index=pd.MultiIndex.from_product(indexes, names=names)
            #print(column_index)
            nac_df_system=pd.DataFrame(nac_array, columns=column_index) #index=np.arange(start, stop, stride)
        
        except:
            
            print(f'Empty array for {name}.')
            
            nac_df_system=pd.DataFrame()
            
            
        return (nac_file, nac_df_system)



