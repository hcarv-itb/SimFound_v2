# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:34:14 2021

@author: hcarv
"""

import Trajectory
import tools_plots


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

import MDAnalysis as mda
import MDAnalysis.analysis.distances as D


import mdtraj as md


from simtk.unit import picoseconds

import tools

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
        self.systems=systems
        self.results=results
        self.features={}
        #self.levels=self.systems.getlevels()
        self.timestep=timestep
        self.unit='ps'
        
        print('Results will be stored under: ', self.results)
 
        

        
 

    def calculate(self, 
                inputs,
                method='method',
                start=0,
                stop=-1,
                stride=1,
                n_cores=-1,
                feature_name=None,
                multi=False):
        """
        Calculates something based on some 'inputs'. 
        Increase "n_cores" to speed up calculations.
        Retrieves the output array and corresponding dataframe.
        Recieves instructions from multiprocessing calls.
        Makes calls to "*_calculation" methods.
        Operation is adjusted for multiprocessing calls, hence the requirement for tuple manipulation.
        
        
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
            A dataframe containing all the d_NAC values across the list of iterables*replicas*pairs.

        """
           
        # Define system specifications and measurement instructions
        systems_specs=[]
        
        for name, system in self.systems.items(): #system cannot be sent as pickle for multiproc, has to be list
            
        
            if not multi:
                trajectory=system.trajectory
            else:
                pass
            
            topology=system.topology #input_topology
            results_folder=system.results_folder
            timestep=system.timestep
            
            systems_specs.append((trajectory, topology, results_folder, name))
            
        measurements=(inputs, start, stop, timestep, stride)
        #print(system_specs)
        #print(measurements)
        
            
        methods_dict={'dNAC' : Featurize.nac_calculation,
                      'distance' : Featurize.distance_calculation,
                      'RMSD' : Featurize.rmsd_calculation,
                      'RMSF' : Featurize.rmsf_calculation}  
        
        
        method_function, _=self.getMethod(method)
                    
        data_t=tools.Tasks.parallel_task(method_function, systems_specs, measurements, n_cores=n_cores)
        
        # concat dataframes from multiprocessing into one dataframe
        #Update state of system and features with the feature df.
        
        out_df=pd.DataFrame()
        
        
        for data_df, system in zip(data_t, self.systems.values()):
            #print(data_df.index.name)
            #print(data_df)
            #print(system)
            system.data[method]=data_df
            system.features[method]=inputs
            
            out_df=pd.concat([out_df,  data_df], axis=1)
        out_df.index.rename=data_df.index.name
        
        out_df.to_csv(os.path.abspath(f'{self.results}/{method}_{str(inputs)}.csv'))

        if feature_name != None:

            self.features[f'{method}-{feature_name}']=out_df
        else:
            self.features[f'{method}']=out_df
            
        
        
        
        print('Featurization updated: ', [feature for feature in self.features.keys()])
        
        return out_df



            
        
 
 
    
    
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
    
    
    




    @staticmethod
    def rdf_calculation(system_specs, specs):
        
        import MDAnalysis.analysis.rdf as RDF
        
        names, indexes, column_index=Featurize.df_template(system_specs, unit=[r'$G_r$'])
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride)=specs   
        
        
        traj, status=Trajectory.Trajectory.loadTraj(topology, trajectory, name)
    
        if status == ('full') or len(traj) > 1:
            
            ref, sel = traj.select_atoms(selection[0]), traj.select_atoms(selection[1])
        
            rdf_=RDF.InterRDF(ref, sel)
            rdf_.run(start, stop, stride)
            
            rows=pd.Index(rdf_.bins, name=r'\AA')
            df_system=pd.DataFrame(rdf_.rdf, columns=column_index, index=rows)
            #df_system=df_system.mask(df_system > 90)
            
            #print(df_system)
            
            return df_system
    
        else:
            return pd.DataFrame()
    
    @staticmethod
    def rmsd_calculation(system_specs, specs):
        
        names, indexes, column_index=Featurize.df_template(system_specs, unit=['nm'])
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride)=specs   
        
        #print(f'\tCalculating RMSD for {name}') 
 
        traj, status=Trajectory.Trajectory.loadMDTraj(topology, trajectory, name)
    
        if status == ('full' or 'incomplete') and len(traj) > 1:
            
            atom_indices=traj.topology.select(selection)
            
            
            #print(atom_indices)
            traj.atom_slice(atom_indices, inplace=True)
            traj.center_coordinates()
            
            rmsd=md.rmsd(traj[start:stop:stride], 
                         traj[start:stop:stride], 
                         0, 
                         precentered=True)  
            rows=pd.Index(np.arange(0, len(rmsd), stride)*timestep, name='ps')
            df_system=pd.DataFrame(rmsd, columns=column_index, index=rows)
            #df_system=df_system.mask(df_system > 90)
            
            #print(df_system)
            
            return df_system
    
        else:
            return pd.DataFrame()


    @staticmethod
    def rmsf_calculation(system_specs, specs):
        
        names, indexes, column_index=Featurize.df_template(system_specs, unit=['nm'])
        
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride)=specs   
        
        
        traj, status=Trajectory.Trajectory.loadMDTraj(topology, trajectory, name)
    
        if status == ('full' or 'incomplete') and len(traj) > 1:
            
            
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
    def df_template(system_specs, unit=['nm'], multi=False, multi_len=1):
        
        (trajectory, topology, results_folder, name)=system_specs
        
        indexes=[[n] for n in name.split('-')]

        if multi:        
            pairs=[i for i in range(1, multi_len+1)]
            indexes.append(pairs)
        
        indexes.append(unit)
        names=[f'l{i}' for i in range(1, len(indexes)+1)]
        column_index=pd.MultiIndex.from_product(indexes, names=names)
        
        return names, indexes, column_index


    @staticmethod
    def distance_calculation_original(system_specs, specs, child=False): #system_specs, measurements):
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
        (distances, start, stop, timestep, stride)=specs    
        
        
        indexes=[[n] for n in name.split('-')] 
        names=[f'l{i}' for i in range(1, len(indexes)+2)] # +2 to account for number of molecules
        #TODO: Make function call to get the real names of the levels. Current l1, l2, l3, etc.
        
 
        dists=[] #list of distance arrays to be populated
            
        #iterate through the list of sels. 
        #For each sel, define atomgroup1 (sel1) and atomgroup2 (sel2)
#        fig,axes=plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        for idx, dist in enumerate(distances, 1): 
            
            dist_file=f'{results_folder}/distance{idx}-i{start}-o{stop}-s{stride}-{str(timestep).replace(" ","")}.npy'
            clock_s=time.time()
            if not os.path.exists(dist_file):
                
                print(f'Distance {idx} not found for {name}. Reading trajectory...')               
 
    
                try:
                    u=mda.Universe(topology, trajectory)

                except OSError:
                    
                    print(f'\tDCD parser could not handle file of {name}.')
                    u=mda.Universe(topology)
                
                except FileNotFoundError:
                    
                    print(f'\tFile(s) not found for {name}.')
                    u=None
                
                
                if u != None:
                    
                    #print(u.dimensions)
                    ref, sel =u.select_atoms(dist[0]).positions, u.select_atoms(dist[1]).positions
                
                    #print(ref)
                    #print(sel)
                    
                    print(f'\t\tref: {u.select_atoms(dist[0])[0]} x {len(ref)}\n\t\tsel: {u.select_atoms(dist[1])[0]} x {len(sel)}')
                    print(f'\tCalculating distance {idx} of {name}') 
                    
                    try:
# =============================================================================
#                         dists_=np.around([D.distance_array(ref, 
#                                                            sel, 
#                                                            box=u.dimensions, 
#                                                   result=np.empty((len(ref), len(sel)))) for ts in u.trajectory[start:stop:stride]],
#                                 decimals=3) 
#                         
# =============================================================================
                        dists_=np.around([D.distance_array(ref, 
                                                           sel, 
                                                           box=u.dimensions) for ts in u.trajectory[start:stop:stride]], decimals=3) 
                        
                        print('in traj: ', dists_)
                
                    except:
                        
                        dists_=np.around(D.distance_array(ref, sel, box=u.dimensions, result=np.empty((len(ref), len(sel)))))
                        print('in top: ', dists_)
                    
                    
                    np.save(dist_file, dists_)
                    
                else:
                    
                    dists_=np.empty(len(np.arange(start,stop,stride)), (len(ref), len(sel)))
                    print('in empty: ', dists_)
                    
 
                        
            else:
                print(f'\tDistance {idx} found for {name}')
                dists_=np.load(dist_file)
                
                print('in load: ', dists_)
            clock_e=time.time()
            
            #print(f'distance {idx} of {name}: {np.shape(dists_)}')
            dists.append(dists_)
        
        #for idx, dists_ in enumerate(dists, 1):
            #print(f'\tDistance {idx}: \n\tShape: {np.shape(dists_)}, \n\tRanges: {np.min(dists_), np.max(dists_)} \n\tCalculation time: ({np.round(clock_e-clock_s, decimals=2)} s)')    
      
        
        if child:
            
            return np.asarray(dists) #return
        
        else:
            
            dist_array=np.asarray(dists[0])
            
            try:
                #TODO: Check behaviour for ref > 1
                #TODO: switch ref and sel for full.
                frames, sel, ref=np.shape(dist_array)
            
                print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}\n\t{np.min(dist_array)}, {np.max(dist_array)}')
            
                pairs=np.arange(1,sel+1)
                #print(pairs)
                indexes.append(pairs)
                #print(indexes)
                column_index=pd.MultiIndex.from_product(indexes, names=names)
                #print(column_index)
                dist_df_system=pd.DataFrame(dist_array, columns=column_index) #index=np.arange(start, stop, stride)
        
            except:
            
                print(f'Empty array for {name}.')
            
                dist_df_system=pd.DataFrame()
            
            
            return (dist_file, dist_df_system)

    @staticmethod
    def distance_calculation(system_specs, specs, child=False): #system_specs, measurements):
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
        (selection,  start, stop, timestep, stride)=specs   
        
        
        traj, status=Trajectory.Trajectory.loadTraj(topology, trajectory, name)
        
        if status == ('full' or 'incomplete'):
            
            
            
            ref, sel = traj.select_atoms(selection[0]), traj.select_atoms(selection[1]) 
            
            #print(ref)
            print(sel)
            
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
        
# =============================================================================
#    
# 
#         if child:
#             print('in child')
#             return np.asarray(dists) #return
#         
#         else:
#             
#             dist_array=np.asarray(dists[0])
#             print(dist_array)
#             
#             try:
#                 #TODO: Check behaviour for ref > 1
#                 #TODO: switch ref and sel for full.
#                 frames, sel, ref=np.shape(dist_array)
#             
#                 print(f'{name} \n\tNumber of frames: {frames}\n\tSelections: {sel}\n\tReferences: {ref}\n\t{np.min(dist_array)}, {np.max(dist_array)}')
#             
#                 pairs=np.arange(1,sel+1)
#                 #print(pairs)
#                 indexes.append(pairs)
#                 #print(indexes)
#                 column_index=pd.MultiIndex.from_product(indexes, names=names)
#                 #print(column_index)
#                 dist_df_system=pd.DataFrame(dist_array, columns=column_index) #index=np.arange(start, stop, stride)
#         
#             except:
#             
#                 print(f'Empty array for {name}.')
#             
#                 dist_df_system=pd.DataFrame()
#             
#             
#             return (dist_file, dist_df_system)
#  
# =============================================================================

    @staticmethod
    def getMethod(method):
        
        methods_dict={'dNAC' : Featurize.nac_calculation,
                      'distance' : Featurize.distance_calculation,
                      'RMSD' : Featurize.rmsd_calculation,
                      'RMSF' : Featurize.rmsf_calculation,
                      'RDF' : Featurize.rdf_calculation}
        
        
        kind={'RMSD': 'global',
              'RMSF': 'by_element',
              'RDF' : 'by_element',
              'distance' : 'global'}
        
        try:
            method_function=methods_dict[method]
            print(f'Selected method: {method} using the function "{method_function.__name__}"')
           
            return method_function, kind[method] 
           
        except:
            
            print(f'Method {method} not supported.')
            print('Try: ', [method for method in methods_dict.keys()])




    
    def plot(self, 
             input_df=None, 
             method='RMSD',
             feature_name=None,
             level=2, 
             subplots_=True,
             stats=True):
        """
        

        Parameters
        ----------
        input_df : TYPE, optional
            DESCRIPTION. The default is None.
        method : TYPE, optional
            DESCRIPTION. The default is 'RMSD'.
        level : TYPE, optional
            DESCRIPTION. The default is 2.
        subplots_ : TYPE, optional
            DESCRIPTION. The default is True.
        stats : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        
        
        print('Plotting :', method)
        try:
            input_df=self.features[f'{method}-{feature_name}']

        except:
            input_df=input_df

        function, kind=self.getMethod(method)
        levels=input_df.columns.levels #might need to subselect here
        units=levels[-1].to_list()[0]
        
        #Set stats
        level_iterables=input_df.columns.get_level_values(f'l{level}').unique()
        sublevel_iterables=input_df.columns.get_level_values(f'l{level+1}').unique()  
        
        
        #plot level
        rows, columns, fix_layout=tools_plots.plot_layout(level_iterables)
        fig_,axes_=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        
        print(axes_)
        
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
                       legend=True, 
                       sort_columns=True,
                       ax=ax_it)

                title_=f'{method}: {df_it.index.values[0]} to {df_it.index.values[-1]} {df_it.index.name}'
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
              
        

            #fig_it.suptitle(title_)
            fig_it.show()
            fig_it.savefig(os.path.abspath(f'{self.results}/{method}_sub_l{level+1}-{sup_it}.png'), bbox_inches="tight", dpi=600)
         
        fig_.legend(sublevel_iterables)
        fig_.show()
        fig_.savefig(os.path.abspath(f'{self.results}/{method}_l{level}_stats.png'), bbox_inches="tight", dpi=600)  


#####LEGACY
# =============================================================================
# 
#     def dist(self,  
#                 distances,
#                 feature_name='feature',
#                 start=0,
#                 stop=-1,
#                 stride=1,
#                 n_cores=-1):
#             
#             
#             
#         #self, distances, start=0, stop=-1, stride=1)->dict:
#         
#         """
#         Calculates distances between pairs of atom groups (dist) for each set of selections "dists" 
#         using MDAnalysis D.distance_array method.
#         Stores the distances in results_dir. Returns the distances as dataframes.
# 	    Uses the NAC_frames files obtained in extract_frames(). 
# 	    NOTE: Can only work for multiple trajectories if the number of atoms is the same (--noWater)
#         
#         
# 
#         Parameters
#         ----------
#         distances : list
#             DESCRIPTION.
#         start : int, optional
#             DESCRIPTION. The default is 0.
#         stop : int, optional
#             DESCRIPTION. The default is -1.
#         stride : int, optional
#             DESCRIPTION. The default is 1.
# 
#         Returns
#         -------
#         dict
#             A dictionary of distance.
# 
#         """
#         
#         # Define system specifications and measurement instructions
#         systems_specs=[]
#         
#         for name, system in self.systems.items(): #system cannot be sent as pickle for multiproc, has to be list
#             
#             trajectory=system.trajectory
#             topology=system.topology
#             results_folder=system.results_folder
#             timestep=system.timestep
#             
#             systems_specs.append((trajectory, topology, results_folder, name))
#             
#         measurements=(distances, start, stop, timestep, stride)
#         #print(system_specs)
#         #print(measurements)
#         
#         
#         data_t=tools.Tasks.parallel_task(Featurize.distance_calculation, 
#                                              systems_specs, 
#                                              measurements, 
#                                              n_cores=n_cores)
#         
#         # concat dataframes from multiprocessing into one dataframe
#         #Update state of system and features with the feature df.
#         
#         distance_df=pd.DataFrame()
#         
#         for data_df, system in zip(data_t, self.systems.values()):
#           
#             system.data[feature_name]=data_df[0]
#             system.features[feature_name]=data_df[1]
#             
#             distance_df=pd.concat([distance_df,  distance_df[1]], axis=1)
#                                        
#         #print(nac_df)
#         
#         self.features[feature_name]=distance_df
#         
#         return distance_df
#     
#     
#     
#        def nac(self,  
#                 distances,
#                 feature_name='feature',
#                 start=0,
#                 stop=-1,
#                 stride=1,
#                 n_cores=-1):
#         """
#         Calculates the d_NAC values from an arbitrary number of distance "sels" pairs. 
#         Increase "n_cores" to speed up calculations.
#         Retrieves the d_NAC array and corresponding dataframe.
#         Recieves instructions from multiprocessing calls.
#         Makes calls to "nac_calculation" method.
#         Operation is adjusted for multiprocessing calls, hence the requirement for tuple manipulation.
#         
#         
#         Parameters
#         ----------
# 
# 
#         distances : list of tuples
#             A list of distance tuples of the kind [(ref1-sel1), ..., (refN-selN)].
#         feature_name : string
#             The name given to the featurized object. The default is 'feature'.
#         start : int, optional
#             The starting frame used to calculate. The default is 0.
#         stop : int, optional
#             The last frame used to calculate. The default is -1.
#         stride : int, optional
#             Read every nth frame. The default is 1.
#         n_cores : int, optional
#             The number of cores to execute task. Set to -1 for all cores. The default is 2.
# 
#         Returns
#         -------
#         dataframe
#             A dataframe containing all the d_NAC values across the list of iterables*replicas*pairs.
# 
#         """
#            
#         # Define system specifications and measurement instructions
#         systems_specs=[]
#         
#         for name, system in self.systems.items(): #system cannot be sent as pickle for multiproc, has to be list
#             
#             trajectory=system.trajectory
#             topology=system.topology
#             results_folder=system.results_folder
#             timestep=system.timestep
#             
#             systems_specs.append((trajectory, topology, results_folder, name))
#             
#         measurements=(distances, start, stop, timestep, stride)
#         #print(system_specs)
#         #print(measurements)
#         
#         
#         data_t=tools.Tasks.parallel_task(Featurize.nac_calculation, 
#                                              systems_specs, 
#                                              measurements, 
#                                              n_cores=n_cores)
#         
#         # concat dataframes from multiprocessing into one dataframe
#         #Update state of system and features with the feature df.
#         
#         nac_df=pd.DataFrame()
#         
#         for data_df, system in zip(data_t, self.systems.values()):
#           
#             system.data[feature_name]=data_df[0]
#             system.features[feature_name]=data_df[1]
#             
#             nac_df=pd.concat([nac_df,  data_df[1]], axis=1)
#                                        
#         #print(nac_df)
#         
#         self.features[feature_name]=nac_df
#         
#         return nac_df
# =============================================================================
