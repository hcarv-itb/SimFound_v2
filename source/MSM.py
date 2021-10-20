# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:52:02 2021

@author: hcarv
"""

import os
import pyemma
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pickle
import nglview
import Trajectory
import tools_plots
import tools
import warnings
import mdtraj as md
import seaborn as sns
from matplotlib import cm


class MSM:
    """
    Base class to create a pyEMMA *features* object.
    
    
    """
    
    skip=0
    c_stride=1
    subset='not water'
    superpose='protein and backbone'
    var_cutoff=0.95
    msm_var_cutoff = 0.95
    
    
    def __init__(self,
                 project,
                 regions,
                 timestep,  
                 chunksize=0, 
                 stride=1,
                 skip=0,
                 confirmation=False,
                 _new=False,
                 warnings=False,
                 heavy_and_fast=False,
                 pre_process=True):
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
        self.systems = self.project.systems
        self.regions=regions
        #TODO: make storage specific to system if multiple systems are within project path.
        
        



        self.features={}
        self.data = {}
        self.chunksize=chunksize
        self.stride=stride
        self.skip=skip
        self.timestep=timestep
        self.unit='ps'
        self.warnings=warnings
        self.heavy_and_fast=heavy_and_fast
        self.pre_process=pre_process
         #os.path.abspath(f'{self.systems.results}/MSM')
        result_paths = []
        for name, system in self.systems.items(): 
            result_paths.append(system.project_results)
        if len(list(set(result_paths))) == 1:
            result_path = list(set(result_paths))[0]
            self.results = os.path.abspath(f'{result_path}/MSM')
            
        else:
            self.results=os.path.abspath(f'{self.project.results}/MSM')
            print('Warning! More than one project result paths defined. reverting to default.')
        self.stored=os.path.abspath(f'{self.results}/MSM_storage')
        print('Results will be stored under: ', self.results)
        print('PyEMMA calculations will be stored under: ', self.stored)
        tools.Functions.fileHandler([self.results, self.stored], confirmation=confirmation, _new=_new)
 

    def filterModel(self, name, data_tuple, filters):
   
        (msm, n_states_msm, lag) = data_tuple

        fraction_state = np.around(msm.active_state_fraction, 3)
        fraction_count = np.around(msm.active_count_fraction, 3)
        resolved_processes = self.spectral_analysis(name, data_tuple, plot=False)
        
        
        keep = 0
        for filter_ in filters:
            
        
            if filter_ == 'connectivity':
                if fraction_state == 1.000:
                    keep +=1
                else:
                    print(f'\tWarning! Model {name} ({n_states_msm}states @ {lag*self.timestep}) is disconnected: {fraction_state} states, {fraction_count} counts')

            if filter_ =='time_resolved':
                if resolved_processes > 1:
                    keep += 1
                else:
                    print(f'\tWarning! Model {name} ({n_states_msm}states @ {lag*self.timestep}) has no processes resolved above lag time.')
        
        
        if keep == len(filters):
            return True
        else:
            return False


    def analysis(self, 
                 inputs, 
                 disc_lags=[200, 1000], 
                 method='StatDist', 
                 features=['torsions', 'positions'], 
                 dim=-1,
                 filters=['connectivity', 'time_resolved']):
        
        #TODO: make self.regions and inputs cross check.
        #ft_names = self.regions
        self.features=features
        self.overwrite=False
        
        df = pd.DataFrame()
        n_models=0
        #TODO: selection inputs can also be shown here.
        for ft_name, value in inputs.items(): # ft_names.keys():
            print(ft_name)
            if isinstance(value, tuple):
                (n_states, lag, metastates) = value
            else:
                raise TypeError('Input dictionary values have to be a tuple of integers (number of states, lag, macrostates)')
            self.ft_name= ft_name
            self.discretized={f'{feature}@{disc_lag*self.timestep}' : None for feature in features for disc_lag in disc_lags}
            self.bayesMSM_calculation(n_states, lag)
            
            for name, data_tuple in self.msm.items():
                keep_model = self.filterModel(name, data_tuple, filters)
                if keep_model:
                    print('\tModel: ', name)
                    n_models+=1
                    (msm, n_states_msm, lag) = data_tuple
                    if method == 'CKtest':
                        self.data['CKtest'] = self.CKTest_calculation(name, data_tuple, metastates)
                    elif method == 'PCCA':
                        self.TICA_calculation(disc_lags, dim=dim)
                        self.data ['PCCA'] = self.PCCA_calculation(name, data_tuple, metastates)
                    elif method == 'Statdist':
                        statdist = msm.stationary_distribution
                        column_index=pd.MultiIndex.from_tuples([(ft_name, name, n_states_msm, lag)], 
                                                               names=['Region', 'Discretization', 'states', 'lag'])
                        df = pd.concat([df, pd.DataFrame(statdist, index=msm.active_set, columns=column_index)], axis=1)
                        self.plot_statdist()
                    elif method == 'Spectral':
                        self.spectral_analysis(name, data_tuple, plot=True)
                    elif method == 'MFPT_cg':
                        self.cg_MFPT(name, data_tuple, metastates)
                    else:
                        raise SyntaxError('Analysis method not defined')
     
        print(f'{n_models} models filtered')

            
        

    def plot_statdist(self):
        
        print('Plotting')
        
        df=self.data['StatDist']
        
        for region in df.columns.get_level_values(0).unique():
            print('data', region)
            region_df = df.loc[:,region]

            fig = region_df.plot(kind='bar', sharex=True, figsize=(7,5))
            fig.set_title(f'Stationary distributions of {region} MSMs', weight='bold')
            fig.set_ylabel(r'Stationary distribution $\pi$')
            fig.set_xlabel('State index')
            plt.savefig(f'{self.results}/StationaryDistributions_{region}.png', dpi=600, bbox_inches="tight")
        
        
#        rows, columns=tools_plots.plot_layout(self.msm)
#        fig_statdist, axes_statdist = plt.subplots(rows, columns, figsize=(9, 6), constrained_layout=True, sharex=True)
# =============================================================================
#         for model in self.models.items()
#                     ax.bar(msm.active_set, statdist)
#                     ax.title(fr'{name}, {n_states} states @ $\tau$ = {lag*self.timestep}')
#             fig_statdist.suptitle(f'{ft_name} Stationary distributions', weight='bold')      
#             fig_statdist.text(0.5, -0.04, 'State index', ha='center', va='center', fontsize=12)
#             fig_statdist.text(-0.04, 0.5, r'$\pi$', ha='center', va='center', rotation='vertical', fontsize=12)
#             fig_statdist.tight_layout() 
#             fig_statdist.savefig(f'{self.results}/StationaryDistributions.png', dpi=600, bbox_inches="tight")
# =============================================================================

        
    def calculate(self,
                  inputs=None,
                  method=None,
                  evaluate=[None], 
                  opt_feature={}, 
                  features=['torsions', 'positions', 'distances'],
                  TICA_lags=[1], 
                  dim=-1,
                  equilibration=False,
                  production=True,
                  def_top=None,
                  def_traj=None,
                  overwrite=False):
        """
        Wrapper function to calculate VAMP2 scores using pyEMMA. 
        


        inputs : list or tuple
            Either a list of 1 selection or atuple of 2 selections.
        evaluate : string
            How to calculate VAMP scores. The default is None.
        ft_name: string, optional.
            The optional name of the inputs. The default is None.
        lags : list, optional.
            A list of lag values to calculate VAMP2 scores. The defaults is [1].
        dim : int, optional.
            The number of dimensions to use for VAMP2 calculation in some methods. 
            If fractionary and < 1, will calculate the corresponding precentage of kinetic variance.
            More details in pyEMMA pyemma.vamp.


        Returns
        -------
        dataframe
            A dataframe containing all featurized values across the list of iterables*replicas*pairs.

        """

        # Define system definitions and measurement hyperparameters

        systems_specs=[]
        for name, system in self.systems.items():        

            results_folder=system.results_folder
            timestep=system.timestep
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
        
        #Gather all trajectories and handover to pyEMMA
        trajectories_to_load = []
        tops_to_load = []
        for system in systems_specs: #trajectory, topology, results_folder, name
            #TODO: check if only trj_list[0] is always the enough.
            trj_list, top=system[0], system[1]
            if len(trj_list):
                trajectories_to_load.append(trj_list[0])
                tops_to_load.append(top)
        self.tops_to_load=tops_to_load[0]
        self.trajectories_to_load=trajectories_to_load
        self.features=features
        self.overwrite=overwrite

        #Use optional feature not defined at __init__
        try:
            ft_names = self.regions
        except:
            if not isinstance(ft_names, dict):
                raise TypeError('Provide a dictionary of type {region : inputs}')
            ft_names = opt_feature
 
        #Loop through features
        data = {}
        for ft_name, ft_inputs in ft_names.items():
            
            print(ft_name)
            self.inputs=ft_inputs
            self.ft_name=ft_name

            
            #VAMPs
            if method == 'VAMP':
                for ev in evaluate:
                    if ev == 'features':
                        vamps=self.VAMP2_features(TICA_lags, dim)
                    elif ev == 'dimensions': 
                        vamps = self.VAMP2_dimensions(TICA_lags)   
                    else:
                        print('No evaluation defined.')
                
                    out_data = vamps
            else:
                
                #TODO: flexibilize here for other alternative schemes
                self.discretized=self.TICA_calculation(TICA_lags, dim=dim)
                
                #TICAS
                if method == 'TICA':
                    out_data = self.discretized # warning! remove if above changes
                elif method == 'Clustering':
                    out_data = self.cluster_calculation(lags=TICA_lags, method='kmeans')
                    
                #MSMs    
                elif method == 'ITS':
                    if isinstance(inputs, dict):
                        out_data = self.ITS_calculation(inputs[ft_name])
                    else:
                        raise TypeError('Input values have to be a integer "number of states"')
                elif method == 'bayesMSM':
                    if isinstance(inputs[ft_name], tuple) and len(inputs[ft_name]) == 2:
                        (n_states, msm_lag) = inputs[ft_name] 
                        out_data = self.bayesMSM_calculation(n_states, msm_lag)
                    else:
                        raise TypeError('Input values have to be a tuple of integers (number of states, lag)')


                else:
                    raise ValueError('No method defined')
            
            data[f'{ft_name}-{method}'] = out_data
        
        self.data = data
        return self.data
    
    
    
    def plot_MSM_TICA(self, name, data_tuple, tica_concat):
        
        (msm, n_states, lag) = data_tuple
        fig, axes = plt.subplots(1,2, figsize=(8,6), constrained_layout=True)
        pyemma.plots.plot_free_energy(*tica_concat[:, :2].T, ax=axes[0], legacy=False)
        pyemma.plots.plot_free_energy(*tica_concat[:, :2].T, weights=np.concatenate(msm.trajectory_weights()), ax=axes[1], legacy=False)
        axes[0].set_title(f'TICA of {name}')
        axes[1].set_title(f'MSM with {n_states} states @ {lag*self.timestep}')
        
        fig.text(0.5, -0.04, 'IC 1', ha='center', va='center', fontsize=12)
        fig.text(-0.04, 0.5, 'IC 2', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.suptitle(f'{self.ft_name} free energy landscape in TICA space', weight='bold', fontsize=12)
        fig.savefig(f'{self.results}/FreeEnergyTICA_{self.ft_name}_{name.replace(" ", "")}_stride{self.stride}.png', dpi=600, bbox_inches="tight")
        
        plt.show()
    
    def cg_MFPT(self, name, data_tuple, metastates):
        (msm, n_states, lag) = data_tuple
        msm.pcca(metastates)
        assignments=msm.metastable_assignments
        statdist = msm.stationary_distribution
        mpg = sns.load_dataset("mpg")
        #print(mpg)
        
        colors= pl.cm.Accent(np.linspace(0,1,metastates))
        fig=plt.figure(figsize=(12,8), constrained_layout=True)
        gs = gridspec.GridSpec(nrows=2, ncols=2)
            
        #plot histo
        ax0 = fig.add_subplot(gs[1, 0]) #states statdist
        ax1 = fig.add_subplot(gs[1,1], sharey=ax0) #
        ax2 = fig.add_subplot(gs[0,0], sharex=ax0)

        #fig, axes = plt.subplots(2,1, figsize=(8,6), sharex=True, constrained_layout=True)
        for x, stat in zip(range(1, len(statdist)+1), statdist):
            ax0.scatter(x, assignments[x-1]+1, s=300*stat, color=colors[assignments[x-1]])
            
            #for t in assignments:
                #axes[1].plot(range(1, len(statdist)+1), assignments[t])
        
# =============================================================================
#         state_labels = range(metastates)
#         for idx, meta in enumerate(assignments, 1):
#             print(f"Macrostate {idx}: {meta} ", assignments)
# =============================================================================

# =============================================================================
#         groups=[[] for f in np.arange(metastates)]
#         print(groups)
#         print(state_labels)
#         for k,v in enumerate(assignments):
#             groups[v].append(state_labels[k])
#         print(groups)
# =============================================================================

        index_cg=pd.MultiIndex.from_product([[self.ft_name], [name], [s for s in np.arange(1, metastates+1)]], names=['region', 'data', 'Macrostate'])

        statdist_cg=[]
        for i, s in enumerate(msm.metastable_sets):
            statdist_m = msm.pi[s].sum()
            ax1.barh(i+1, statdist_m, color=colors[i])
            print(f'Metastate {i+1} = {statdist_m}')
            statdist_cg.append(statdist_m)
        
        
        for idx, met_dist in enumerate(msm.metastable_distributions):
                #print(idx, met_dist)
                ax2.bar(range(1, len(statdist)+1), met_dist, color=colors[idx], label=f'MS {idx+1}')
        
        ax0.set_xlabel('State index')
        ax0.set_yticks(range(1, metastates+1))
        ax0.set_ylabel('MS assignment')
        #ax1.set_xticks([])
        #ax1.set_yticks([])
        #ax1.legend()
        ax2.set_ylabel('MS distributions')
        ax2.legend()
        plt.show()
        pi_cg=pd.DataFrame(statdist_cg, columns=[r'$\pi$'],  index=index_cg)

        mfpt_cg_table=pd.DataFrame([[msm.mfpt(msm.metastable_sets[i], msm.metastable_sets[j]) for j in range(metastates)] for i in range(metastates)], 
                                   index=index_cg, columns=[c for c in np.arange(1, metastates+1)])
        #mfpt_cg_c=pd.concat([mfpt_cg_c, mfpt_cg_table], axis=0, sort=True)
        print(mfpt_cg_table)
        #sns.heatmap(mfpt_cg_table, linewidths=0.1, cmap='rainbow', cbar_kws={'label': r'rates (log $s^{-1}$)'})
        #mfpt_cg_table.plot()
# =============================================================================
#         #rates_cg_table=mfpt_cg_table.apply(lambda x: (1 / x) / 1e-12)
#         rates_cg_table.replace([np.inf], np.nan, inplace=True)
#         rates_cg_table.fillna(value=0, inplace=True)
# 
#         #rates_cg_table=pd.concat([rates_cg_table, pi_cg], axis=1)
#         #rates_cg_c=pd.concat([rates_cg_c, rates_cg_table], axis=0, sort=True)
#         print(rates_cg_table)
#         rates_cg_table.plot()
# =============================================================================

    def plot_MFPT(self, mfpt_df, scheme, feature, parameters, error=0.2, regions=None, labels=None):
        """Function to plot heatmap of MFPTS between all states."""
            
        # Taken from matplotlib documentation. Make images respond to changes in the norm of other images (e.g. via the
        # "edit axis, curves and images parameters" GUI on Qt), but be careful not to recurse infinitely!
        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())     
            
        images = []
        
        
        rows, columns=tools_plots.plot_layout(parameters)
        fig, axes = plt.subplots(rows, columns, constrained_layout=True, figsize=(9,6))
        fig.suptitle(f'Discretization: {scheme}\nFeature: {feature} (error tolerance {error:.1%})', fontsize=14)  
            
        cmap_plot=plt.cm.get_cmap("gist_rainbow")
        #cmap_plot.set_under(color='red')
    
        #cmap_plot.set_over(color='yellow')
        cmap_plot.set_bad(color='white')
            
            
        vmins, vmaxs=[], []
                
        for plot, parameter in zip(axes.flat, parameters):
                
            means=self.mfpt_filter(mfpt_df, scheme, feature, parameter, error) 
                
            try:
                if scheme == 'combinatorial':        
                    contour_plot=plot.pcolormesh(means, edgecolors='k', linewidths=1, cmap=cmap_plot) 
                    label_names=tools.sampledStateLabels(regions, sampled_states=means.index.values, labels=labels)
                    positions=np.arange(0.5, len(label_names)+0.5)                    
                    plot.set_xticks(positions)
                    plot.set_xticklabels(label_names, fontsize=7, rotation=70)
                    plot.set_yticks(positions)
                    plot.set_yticklabels(label_names, fontsize=7)
                else:
                    contour_plot=plot.pcolormesh(means, cmap=cmap_plot)
                    ticks=plot.get_xticks()+0.5
                    plot.set_xticks(ticks)
                    plot.set_xticklabels((ticks+0.5).astype(int))
                    plot.set_yticks(ticks[:-1])
                    plot.set_yticklabels((ticks+0.5).astype(int)[:-1])
                    
                plot.set_facecolor('white')
                plot.set_title(parameter) #, fontsize=8)
                plot.set_xlabel('From state', fontsize=10)
                plot.set_ylabel('To state', fontsize=10)
                images.append(contour_plot)
    
            except:
                print('No values to plot')
    
            
        # Find the min and max of all colors for use in setting the color scale.   
        vmins=[]
        vmaxs=[]
        for image in images:
            array=image.get_array()
            try:
                vmin_i=np.min(array[np.nonzero(array)])
            except:
                vmin_i=1
            try:
                vmax_i=np.max(array[np.nonzero(array)])
            except:
                vmax_i=1e12
            vmins.append(vmin_i)
            vmaxs.append(vmax_i)
            
        vmin=min(vmins)
        vmax=max(vmaxs)
        #vmax = max(image.get_array().max() for image in images)
    
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    
        for im in images:
            im.set_norm(norm)
                 
        print(f'limits: {vmin:e}, {vmax:e}')
             
        cbar=fig.colorbar(images[-1], ax=axes)
        cbar.set_label(label=r'MFPT (ps)', size='large')
        cbar.ax.tick_params(labelsize=12)
        for im in images:
            im.callbacksSM.connect('changed', update)
                            
        return images    
    
    
    #from pyEMMA notebook
    def spectral_analysis(self, name, data_tuple, plot=False):
        """
        

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        data_tuple : TYPE
            DESCRIPTION.
        nits : TYPE, optional
            DESCRIPTION. The default is 15.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        def its_separation_err(ts, ts_err):
            """
            Error propagation from ITS standard deviation to timescale separation.
            """
            return ts[:-1] / ts[1:] * np.sqrt(
                (ts_err[:-1] / ts[:-1])**2 + (ts_err[1:] / ts[1:])**2)
        
        (msm, n_states, lag) = data_tuple
        lag_=str(lag*self.timestep).replace(" ", "")
        dt_scalar=int(str(self.timestep*self.stride).split(' ')[0])
        
        timescales_mean = msm.sample_mean('timescales')
        timescales_std = msm.sample_std('timescales')



        #Filter number os states above 0.95 range of ITS
        state_cutoff=-1
        factor_cutoff =  (1 - MSM.msm_var_cutoff) * (max(timescales_mean)-min(timescales_mean))
        for idx, t in enumerate(timescales_mean, 1):
            if t < factor_cutoff:
                #print(f'Factor {factor_cutoff}, {t} {idx}')
                state_cutoff = idx
                break
            
        imp_ts=range(1, len(timescales_mean)+1)
        if factor_cutoff > dt_scalar*lag:
            timescales_mean_=timescales_mean[:state_cutoff]
            timescales_std_=timescales_std[:state_cutoff]

            imp_ts_factor=imp_ts[:state_cutoff]
            factor = timescales_mean_[:-1]/timescales_mean_[1:]
            factor_labels = [f'{k}/{k+1}' for k in imp_ts_factor[:-1]]
            prop_error=its_separation_err(timescales_mean_, timescales_std_)

            if plot:
                print(f'\t{len(imp_ts_factor)} processes with {MSM.msm_var_cutoff*100}% ITS resolved above lag time ({self.timestep*lag})')
                fig, axes = plt.subplots(1, 2, figsize=(8,6))
                
                axes[0].plot(imp_ts, timescales_mean, marker='o')
                axes[0].fill_between(imp_ts, timescales_mean-timescales_std, timescales_mean+timescales_std, alpha=0.8)
                #axes[0].set_xticks(imp_ts)
                #axes[0].axvline(state_cutoff, lw=1.5, color='green', linestyle='dotted')
                axes[0].axhline(factor_cutoff, lw=1.5, color='red', linestyle='dashed')
                axes[0].axhline(dt_scalar*lag, lw=1.5, color='red', linestyle='solid')
                axes[0].axhspan(0, dt_scalar*lag, alpha=0.5, color='red')
                axes[0].axhspan(dt_scalar*lag, factor_cutoff, alpha=0.2, color='red')
                axes[0].set_xlabel('implied timescale index')
                axes[0].set_ylabel(self.unit)
                axes[0].ticklabel_format(axis='y', style='sci')
                axes[0].set_yscale('log')
                axes[0].set_ylim(bottom=1)
                axes[0].set_title('Implied timescales')
                
                axes[1].errorbar(imp_ts_factor[:-1], factor, yerr=prop_error, marker='o')
                #axes[1].fill_between(imp_ts_factor[:-1], factor-prop_error, factor+prop_error, alpha=0.5)
                axes[1].grid(True, axis='x')
                axes[1].set_xticks(imp_ts_factor[:-1])
                axes[1].set_xticklabels(factor_labels, rotation=45)
                axes[1].set_xlabel('implied timescale indices')
                axes[1].set_ylabel('Factor')
                axes[1].set_title('Timescale separation')
                fig.suptitle(f'ITS decompositon of {name} ({n_states} states@{self.timestep*lag})', weight='bold', fontsize=12)
                fig.tight_layout()
    
                fig.savefig(f'{self.results}/SpectralITS_{name.replace(" ", "")}_{n_states}states@{lag_}_stride{self.stride}.png', dpi=600, bbox_inches="tight")
                plt.show()
            
            return len(imp_ts_factor)

        else:
            print(f'\tModel discarded. Not enough processes resolved above lag time')
            return 0

    def extract_states(self, msm_file_name, msm, macrostate, n_samples=500, save_structures=True):
        """
        

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        msm : TYPE
            DESCRIPTION.
        met_dist : TYPE
            DESCRIPTION.
        macrostate : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        state_samples= {}
        topology=self.tops_to_load
        trajectories=self.trajectories_to_load
        
        superpose_indices=md.load(topology).topology.select(MSM.superpose)
        subset_indices=md.load(topology).topology.select(MSM.subset)
        md_top=md.load(topology) #, atom_indices=ref_atom_indices )
        
        if self.pre_process:
            trajectories=Trajectory.Trajectory.pre_process_MSM(trajectories, md_top, superpose_to=superpose_indices)
            md_top.atom_slice(subset_indices, inplace=True)
        else:
            trajectories = md.load(self.trajectories_to_load)
        
        for idx, idist in enumerate(msm.sample_by_distributions(msm.metastable_distributions, n_samples), 1):
            print(f'\tGenerating samples for macrostate {idx}/{macrostate}', end='\r')
            file_name=f'{self.stored}/{msm_file_name}_{idx}of{macrostate}macrostates.dcd'
            
            if save_structures:
                pyemma.coordinates.save_traj(trajectories, idist, outfile=file_name, top=md_top)
                
            state_samples[idx] = file_name
        
        return state_samples
            
    def RMSD_source_sink(self, 
                         file_names, 
                         selection='backbone'):
        
        df = pd.DataFrame()
        for idx, file_name in file_names.items():
        
            indexes_source=[['RMSD'], [idx], ['source']]
            indexes_sink=[['RMSD'], [idx], ['sink']]
            names=['name', 'macrostates', 'reference']
    
            
            
            source_traj = md.load(self.project.input_topology)
            sink_traj = md.load(f'{self.project.def_input_struct}/SETD2.pdb')
            traj= md.load(file_name, top=self.project.input_topology)
            
            atom_indices=traj.topology.select(selection)
            source_atom_indices=source_traj.topology.select(selection)
            sink_atom_indices=source_traj.topology.select(selection)
            
            common_atom_indices=list(set(atom_indices).intersection(source_atom_indices))
            common_atom_indices_s=list(set(atom_indices).intersection(sink_atom_indices))
            
            traj.atom_slice(common_atom_indices, inplace=True)
            source_traj.atom_slice(common_atom_indices, inplace=True)
            sink_traj.atom_slice(common_atom_indices_s, inplace=True)
            
            rmsd_source=md.rmsd(traj,
                             source_traj,
                             frame=0,
                             precentered=True,
                             parallel=True)
            
            rmsd_sink=md.rmsd(traj,
                             sink_traj,
                             frame=0,
                             precentered=True,
                             parallel=True)
            
            
            rows=pd.Index(np.arange(1, len(rmsd_sink)+1), name='sample')
            df_macro=pd.DataFrame(rmsd_source, 
                                  index=rows, 
                                  columns=pd.MultiIndex.from_product(indexes_source, names=names))
            df_macro=pd.concat([df_macro, pd.DataFrame(rmsd_sink, 
                                                       index=rows, 
                                                       columns=pd.MultiIndex.from_product(indexes_sink, names=names))], axis=1)
            df_macro.hist(grid=False, sharex=True, sharey=True, bins=10, figsize=(8,6))
            df=pd.concat([df, df_macro], axis=1)
        
        
        #print(df)
        return df


        #cmap = mpl.cm.get_cmap('viridis', nstates)
        



    def PCCA_calculation(self, name, data_tuple, macrostates, dims_plot=10) -> dict:
        """
        

        Parameters
        ----------
        macrostates : TYPE
            DESCRIPTION.

        Returns
        -------
        pccas : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.mute = True
        
        #TODO: consider creating meta/macro/hidden distinction
        if isinstance(macrostates, tuple):
            macrostates_=range(macrostates[0], macrostates[1])
        elif isinstance(macrostates, int):
            macrostates_=[macrostates]
        else:
            raise TypeError('Macrostates has to be either a integer or a tuple (min, max) number of macrostates')
        
    
        (msm, n_states, lag) = data_tuple
         
        msm_file_name=f'bayesMSM_{self.ft_name}_{name.replace(" ", "")}_{n_states}states_stride{self.stride}'
        tica=self.tica[name]
        tica_concat = np.concatenate(tica.get_output())
        clusters=self.clusterKmeans_standalone(tica, msm_file_name, n_states)
        dtrajs_concatenated = np.concatenate(clusters.dtrajs)
        #print(*clusters.clustercenters.T)
        self.plot_MSM_TICA(name, data_tuple, tica_concat, self.ft_name)

        #loop through sets of macrostates
        for macrostate in macrostates_:   
            file_name=f'PCCA_{self.ft_name}_{name.replace(" ", "")}_{macrostate}macrostates_stride{self.stride}'                     
            pcca=msm.pcca(macrostate)
            rows, columns=tools_plots.plot_layout(macrostate)
            pcca_plot, axes = plt.subplots(rows, columns, figsize=(8, 6), sharex=True, sharey=True)

            #loop through individual states
            for idx, met_dist in enumerate(msm.metastable_distributions):
                print(len(met_dist[dtrajs_concatenated]))
                try:
                    ax=axes.flat[idx]
                except AttributeError:
                    ax=axes[0]

# =============================================================================
#                 _=pyemma.plots.plot_contour(*tica_concat[:, :2].T,
#                                             met_dist[dtrajs_concatenated],
#                                             ax=ax,
#                                             cmap='rainbow')
# =============================================================================

                _=pyemma.plots.plot_free_energy(*tica_concat[:, :2].T,
                                           met_dist[dtrajs_concatenated],
                                           legacy=False,
                                           ax=ax,
                                           method='nearest') #cbar_label = f'metastable distribution {idx+1}', cmap='rainbow',

                #ax.scatter(*clusters.clustercenters.T)
                ax.set_xlabel('IC 1')
                ax.set_ylabel('IC 2')
                ax.set_title(f'Macrostate {idx+1}')

            pcca_plot.suptitle(f'PCCA+ of {self.ft_name}: {n_states} -> {macrostate} macrostates @ {lag*self.timestep}\nTICA of {name}', 
                               ha='center', 
                               weight='bold', 
                               fontsize=12)
            pcca_plot.tight_layout()
            pcca_plot.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
            plt.show()
                
            #samples=self.extract_states(msm_file_name, msm, macrostate)
            #rmsds = self.RMSD_source_sink(samples)
                
                #print(rmsds)


            
        return (pcca, pcca_plot)


    
    def CKTest_calculation(self, name, data_tuple, metastates, mlags=7) -> dict:
        """
        

        Parameters
        ----------
        metastates : TYPE
            DESCRIPTION.
        mlags : TYPE, optional
            DESCRIPTION. The default is 7.

        Returns
        -------
        ck_tests : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.show_progress_bars = False
        
        (msm, n_states, lag) = data_tuple #(bayesMSM, n_states, lag) 
        lag_ =str(lag*self.timestep).replace(" ", "")        
        file_name = f'CKtest_{self.ft_name}_{name.replace(" ", "")}@{lag_}_{metastates}metastates_stride{self.stride}'
        
        if not os.path.exists(f'{self.stored}/file_name.npy') or self.overwrite:
            print(f'\tPerforming CK test for {name}, {n_states} states -> {metastates} metastates @ {lag*self.timestep}.')
            cktest=msm.cktest(metastates)
            cktest.save(f'{self.stored}/{file_name}.npy', overwrite=True)
        else:
            print('\tCK test found for: ', name)
            cktest=pyemma.load(f'{self.stored}/file_name.npy')

        dt_scalar=int(str(self.timestep*self.stride).split(' ')[0])
        ck_plot, axes=pyemma.plots.plot_cktest(cktest, 
                                               dt=dt_scalar, 
                                               layout='wide', 
                                               marker='.',  
                                               y01=False, 
                                               units=self.unit)
            
            #TODO (maybe?): play around with subplot properties
# =============================================================================
#             for ax in axes:
#                 for subax in ax:
#                     #subax.set_xscale('log')
# =============================================================================
        ck_plot.suptitle(f'CK test: {n_states} -> {metastates} macrostates (TICA of {self.ft_name} {name})', va='top', ha='center', weight='bold', fontsize=12)
        ck_plot.tight_layout()
        ck_plot.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
        
        plt.show()

        return (cktest, ck_plot)

    def bayesMSM_calculation(self, states, lags,  statdist=None) -> dict:
        """

        Parameters
        ----------
        n_states : TYPE
            DESCRIPTION.
        lag : TYPE
            DESCRIPTION.
        variant : TYPE, optional
            DESCRIPTION. The default is False.
        statdist : TYPE, optional
            DESCRIPTION. The default is None.
        overwrite : TYPE, optional
            DESCRIPTION. The default is True.
        c_stride : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        msm_features : TYPE
            DESCRIPTION.

        """


        def check_inputs(input_):
            

            if isinstance(input_, int):
                input_ =[input_]
            elif isinstance(input_, tuple) and len(input_) == 2:
                input_ =range(input_[0], input_[1])
            elif isinstance(input_, list):
                pass
            else:
                raise TypeError(input_+'should be either a single integer, a tuple (min, max) for range or a list of integers')
            
            return input_
        
        states = check_inputs(states)
        lags = check_inputs(lags)
            
        for n_states in states:
            
            for lag in lags:

                msm_features= {}
                for name, data in self.discretized.items():
                    lag_ =str(lag*self.timestep).replace(" ", "")
                    file_name=f'bayesMSM_{self.ft_name}_{name.replace(" ", "")}_{n_states}states@{lag_}_stride{self.stride}'
                    bayesMSM=None
                    msm_stride=1
                    while bayesMSM == None and msm_stride < 10000:
                        msm_name=os.path.abspath(f'{self.stored}/{file_name}_{msm_stride}sits.npy')
                        if not os.path.exists(msm_name) or self.overwrite:
#                            try:
                            print(f'\tGenerating Bayesian MSM for {name} {n_states} states @ {lag*self.timestep} with stride {msm_stride}')
    
                            tica=data.get_output()
                            clusters=self.clusterKmeans_standalone(tica, file_name, n_states)
                            disc_trajs=clusters.dtrajs #np.array(np.load(self.values['discretized'][0]))
                            #data_i=np.ascontiguousarray(data[0::msm_stride])
                            bayesMSM=pyemma.msm.bayesian_markov_model(disc_trajs[0::msm_stride], 
                                                                      lag=lag, 
                                                                      dt_traj=str(self.timestep), 
                                                                      conf=0.95)
                            bayesMSM.save(msm_name, overwrite=True) 
# =============================================================================
#                             except:
#                                 print('Could not generate bayesMSM. Increasing stride.')
#                                 msm_stride=msm_stride*2
# =============================================================================
                        else:
                            bayesMSM=pyemma.load(msm_name)
                            print(f'\tFound Bayesian MSM of {name} ({n_states} states @ {lag*self.timestep})')
        
                    msm_features[name] = (bayesMSM, n_states, lag)
            
        self.msm = msm_features
        return msm_features
        
    def ITS_calculation(self, 
                        n_states,
                        lags=[1, 2, 5, 10, 20, 40, 100, 250, 500, 750, 1000, 1500, 2000],
                        c_stride=1) -> dict:
        """
        

        Parameters
        ----------
        n_states : TYPE
            DESCRIPTION.
        c_stride : TYPE, optional
            DESCRIPTION. The default is 1.
        overwrite : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        pyemma.config.show_progress_bars = False
        
        its_features= {}
        for name, data in self.discretized.items():                      
            file_name = f'ITS_{self.ft_name}_{name.replace(" ", "")}_{n_states}states_stride{self.stride}'            
            print(f'Generating ITS profile for {name}')
        
            tica=data.get_output()

            #TODO: make overwrite cluster, its specific?
            its=None
            its_stride=1
            while its == None and its_stride < 10000:
                #data_i=np.ascontiguousarray(disc_trajs[0::its_stride])
                try:
                    its_name=f'{self.stored}/{file_name}_{its_stride}sits.npy'
                    if not os.path.exists(its_name) or self.overwrite:
                        clusters=self.clusterKmeans_standalone(tica, file_name, n_states)
                        #TODO: make n_centres read from dictionary of input.                
                        disc_trajs = clusters.dtrajs
                        print(f'\tCalculating ITS with trajectory stride of {its_stride}', end='\r')
                        its=pyemma.msm.its(disc_trajs[0::its_stride], lags=lags, errors='bayes')
                        its.save(f'{self.stored}/{file_name}_{its_stride}sits.npy', overwrite=self.overwrite)  
                    else:
                        print('ITS profile found for: ', name)
                        its=pyemma.load(its_name)
                except:
                    print('\tWarning!Could not generate ITS. Increasing stride.')
                    its_stride=its_stride*2
                  
            dt_scalar=int(str(self.timestep*(self.stride*its_stride)).split(' ')[0])
            its_plot=pyemma.plots.plot_implied_timescales(its, units=self.unit, dt=dt_scalar)
            its_plot.set_title(f'ITS of {self.ft_name}: {n_states} states\nTICA of {name}', ha='center', weight='bold', fontsize=10)
            plt.tight_layout()
            plt.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
            plt.show()
        
            its_features[name] = (its, lags, its_plot)
        
        self.its = its_features
        return its_features
    
    
    def TICA_calculation(self,  
                         lags,
                         dim=-1):
        """
        

        Parameters
        ----------
        features : TYPE
            DESCRIPTION.
        lags : TYPE
            DESCRIPTION.
        ft_name : TYPE, optional
            DESCRIPTION. The default is None.
        dim : TYPE, optional
            DESCRIPTION. The default is -1.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        ticas : TYPE
            DESCRIPTION.

        """
        
        if isinstance(lags, str):
            lags = [lags]
        
        if isinstance(dim, float):  
            dim = -1
            
        if isinstance(self.features, str):
            self.features=[self.features]
               
        ticas = {}    
        for feature in self.features:            
            for lag in lags:
                lag_ =str(lag*self.timestep).replace(" ", "")
                file_name = f'{self.stored}/TICA_{self.ft_name}_{feature}@{lag_}_stride{self.stride}.npy'
                if not os.path.exists(file_name) or self.overwrite:                  
                    data = self.load_features(self.tops_to_load, 
                                              self.trajectories_to_load, 
                                              [feature], 
                                              inputs=self.inputs,
                                              name=self.ft_name)[0] #[0] because its yielding a list.
                    print(f'\tCalculating TICA of {feature} with lag {lag*self.timestep}')
                    try:
                        tica = pyemma.coordinates.tica(data, 
                                                       lag=lag, 
                                                       dim=dim,
                                                       var_cutoff=MSM.var_cutoff,
                                                       skip=self.skip, 
                                                       stride=self.stride, 
                                                       chunksize=self.chunksize)
                        tica.save(file_name, save_streaming_chain=True)
                        self.plot_TICA(tica, lag, dim, feature) #TODO: Consider pushing opt_dim up. 
                    except ValueError:
                        print(f'Warning! Failed for {feature}. Probably trajectories are too short for selected lag time and stride')
                        break
                else:
                    print(f'\tFound TICA of {feature}@{lag*self.timestep}')
                    tica = pyemma.load(file_name)
                
                                            
                ticas[f'{feature}@{lag*self.timestep}'] = tica       #name here is feature on cluster calculation.      
        self.tica=ticas

        return ticas

    def plot_TICA(self, 
                  tica,
                  lag,
                  dim,  
                  name, 
                  opt_dim=10):
        """
        

        Parameters
        ----------
        tica : TYPE
            DESCRIPTION.
        dim : TYPE
            DESCRIPTION.
        lag : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        opt_dim : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        Notes
        -----
        
        tica_concat is the trajectories concatenated.
        ticas is the individual trajectories
        """
        lag_ =str(lag*self.timestep).replace(" ", "")

        if tica.dimension() > 10 and dim == -1:
            print(f'\tWarning: TICA for {MSM.var_cutoff*100}% variance cutoff yields {tica.dimension()} dimensions.')
            print(f'Reducing to {opt_dim} dimensions for visualization only.')
            dims_plot=opt_dim
        elif tica.dimension() > dim and dim > 0:
            dims_plot=tica.dimension()
        
        tica_concat = np.concatenate(tica.get_output())
        
        fig=plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(nrows=1, ncols=2)
            
        #plot histogram
        ax0 = fig.add_subplot(gs[0, 0])
        pyemma.plots.plot_feature_histograms(tica_concat[:, :dims_plot], ax=ax0) #, ylog=True)
        ax0.set_title('Histogram')
        
            
        #plot projection along main components
        ax1 = fig.add_subplot(gs[0, 1])
        pyemma.plots.plot_free_energy(*tica_concat[:, :2].T, ax=ax1, legacy=False) #, logscale=True)
        ax1.set_xlabel('IC 1')
        ax1.set_ylabel('IC 2')
        ax1.set_title('IC density')
        

        fig.suptitle(fr'TICA: {self.ft_name} {name} @ $\tau$ ={self.timestep*lag}', weight='bold')
        fig.tight_layout()   
        fig.savefig(f'{self.results}/TICA_{self.ft_name}_{name}@{lag_}_stride{self.stride}.png', dpi=600, bbox_inches="tight")
        plt.show()


        #plot discretized trajectories
        ticas=tica.get_output()
        
        rows, columns=tools_plots.plot_layout(ticas)
        fig_trajs,axes_=plt.subplots(columns, rows, sharex=True, sharey=True, constrained_layout=True, figsize=(10,10))
        fig_trajs.subplots_adjust(hspace=0)

        try:
            flat=axes_.flat
        except AttributeError:
            flat=[axes_]     

        #loop for the trajectories
        for (idx_t, trj_tica), ax in zip(enumerate(ticas), flat): #len trajs?
            
            
            x = self.timestep * np.arange(trj_tica.shape[0])
    
            for idx, tic in enumerate(trj_tica.T):
                
                if idx <= dims_plot:
                    ax.plot(x, tic, label=f'IC {idx}')
                    #ax.set_aspect('equal')
                    #ax.set_ylabel('Feature values')
        
                    #ax.set_xlabel(f'Total simulation time  (in {self.unit})')
                    #ax_trj.set_title('Discretized trajectory: ', idx)
        
        #fig_trajs.legend()
        #fig_trajs.subplots_adjust(wspace=0)
        fig_trajs.suptitle(fr'TICA: {self.ft_name} {name} @ $\tau$ ={self.timestep*lag}', weight='bold')
        handles, labels = ax.get_legend_handles_labels()
        
        def_locs=(1.1, 0.6)
        fig_trajs.legend(handles, labels, bbox_to_anchor=def_locs)
        fig_trajs.tight_layout()
        fig_trajs.text(0.5, -0.04, f'Trajectory time ({self.unit})', ha='center', va='center', fontsize=12)
        fig_trajs.text(-0.04, 0.5, 'IC value', ha='center', va='center', rotation='vertical', fontsize=12)
        fig_trajs.savefig(f'{self.results}/TICA_{self.ft_name}_{name}_lag{self.timestep*lag}_stride{self.stride}_discretized.png', dpi=600, bbox_inches="tight")
        plt.show()

    
    def VAMP2_dimensions(self, lags):
        """
        

        Parameters
        ----------
        lags : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        rows, columns=tools_plots.plot_layout(self.features)
        fig, axes = plt.subplots(rows, columns, figsize=(9,6))

        try:
            flat=axes.flat
        except AttributeError:
            flat=axes
            flat=[flat]

        for ax, feature in zip (flat, self.features):
            
            data, dimensions = self.load_features(self.tops_to_load, 
                                      self.trajectories_to_load, 
                                      [feature], 
                                      inputs=self.inputs,
                                      name=self.ft_name,
                                      get_dims=True)[0] #[0] because its yielding a list.

                        
            dims = [int(dimensions*i) for i in np.arange(0.1, 1.1, 0.1)]
            legends = []

            for lag in lags:
                print(f'\tLag: {lag*self.timestep}\n' )
                scores_dim= []
                #scores_=np.array([self.score_cv(data, dim, lag)[1] for dim in dims])

                for dim in dims:
                    try:
                        scores_dim.append(self.score_cv(data, dim, lag, number_of_splits=10)[1]) #discard vamps [0]

                    except MemoryError:
                        print('\tWarning! Failed due to memory error.')
                        scores_dim.append(np.full(10, 0))

                    except ValueError:
                        print('\tWarning! Failed due to too large lag, stride combination.')
                        scores_dim.append(np.full(10, 0))

                scores_ = np.asarray(scores_dim)
                scores = np.mean(scores_, axis=1)
                errors = np.std(scores_, axis=1, ddof=1)
                #color = 'C{lags.index(lag)}'
                ax.fill_between(dims, scores - errors, scores + errors, alpha=0.3) #, facecolor=color)
                ax.plot(dims, scores, '--o')
                legends.append(r'$\tau$ = '+f'{self.timestep*lag}')
            #ax.legend()
            ax.set_title(feature)
            ax.set_xlabel('number of dimensions')
            ax.set_ylabel('VAMP2 score')
        
        fig.suptitle(f'VAMP2 of {self.ft_name}\nstride {self.stride*self.timestep}', ha='center', weight='bold')
        fig.legend(legends, ncol=1, bbox_to_anchor=(1.03, 0.5), loc='center left')
        
        #if not axes[-1].lines: 
        #    axes[-1].set_visible(False)

        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/VAMP2_evalDimensions_{self.ft_name}_stride{self.stride}.png', dpi=600, bbox_inches="tight")
    
        
    def VAMP2_features(self, lags, dim):
        """
        

        Parameters
        ----------
        lags : TYPE
            DESCRIPTION.
        dim : TYPE
            DESCRIPTION.

        Returns
        -------
        vamps : TYPE
            DESCRIPTION.

        """
        
        vamps={}
        rows, columns=tools_plots.plot_layout(self.features)
        fig, axes = plt.subplots(rows, columns, figsize=(8,6), constrained_layout=True, sharex=True, sharey=True)
        try:
            flat=axes.flat
        except AttributeError:
            flat=axes
            flat=[flat]

        legends = []
        colors= pl.cm.Set2(np.linspace(0,1,len(self.features)))
        
        for feature, ax, color in zip(self.features, flat, colors):
            print('Feature: ', feature)
            data = self.load_features(self.tops_to_load, 
                                      self.trajectories_to_load, 
                                      [feature], 
                                      inputs=self.inputs,
                                      name=self.ft_name)[0] #[0] because its yielding a list.
            
            legends.append(feature)
            scores=[]
            errors=[]
            
            lags_scalar=[]
            for lag in lags:
                try:
                    vamps[feature], ft_scores=self.score_cv(data, lag=lag, dim=dim)
                    scores.append(ft_scores.mean())
                    errors.append(ft_scores.std())
                except ValueError:
                    print('\tWarning! Failed due to too large lag, stride combination.')
                lags_scalar.append(int(str(self.timestep*lag).split(' ')[0]))
                
            errors_=np.asarray(errors)
            ax.plot(lags_scalar, scores, color=color, marker='o')
            ax.fill_between(lags_scalar, scores-errors_, scores+errors_, color=color, alpha=0.5)
            #colors=[f'C{c}' for c in range(len(self.features[ft_name]))]
                
            #ax.bar(labels, scores, yerr=errors, color='green') #, color=colors)
            ax.set_xlabel(f'Lag time (in {self.unit})')
            ax.set_ylabel('VAMP2 score')
            ax.set_xscale('log')
            ax.set_title(feature)
            #ax.tick_params('x', labelrotation=45)
            #vamp_bars_plot = dict(labels=self.labels, scores=scores, errors=errors, dim=dim, lag=lag)
        
        #fig.legend(legends) #, ncol=1, bbox_to_anchor=(1.03, 0.5), loc='center left')
        fig.suptitle(f'VAMP2 of features for {self.ft_name} (stride {self.stride*self.timestep})', weight='bold')
        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/VAMP2_evalFeatures_{self.ft_name}_stride{self.stride}.png', dpi=600, bbox_inches="tight")
        
        return vamps
        

    def load_features(self, 
                      topology, 
                      trajectories, 
                      features,
                      inputs=None,
                      name=None,
                      get_dims=False):
        """
        

        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        trajectories : TYPE
            DESCRIPTION.
        features : TYPE
            DESCRIPTION.
        inputs : TYPE, optional
            DESCRIPTION. The default is None.
        name : TYPE, optional
            DESCRIPTION. The default is None.
        subset : TYPE, optional
            DESCRIPTION. The default is 'not water'.
        superpose : TYPE, optional
            DESCRIPTION. The default is 'protein and backbone'.
        get_dims : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        features_list : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.show_progress_bars = False
        
        features_list=[]
        
        superpose_indices=md.load(topology).topology.select(MSM.superpose)
        subset_indices=md.load(topology).topology.select(MSM.subset)
        md_top=md.load(topology) #, atom_indices=ref_atom_indices )
        
        if self.pre_process:
            trajectories=Trajectory.Trajectory.pre_process_MSM(trajectories, md_top, superpose_to=superpose_indices)
            md_top.atom_slice(subset_indices, inplace=True)
        
        for f in features:
            
            try:
                feat=pyemma.coordinates.featurizer(md_top)

                
                if type(inputs) is tuple:
                    if f == 'distances':
                        print(inputs[0], inputs[1])
                        print(feat.select(inputs[1]))
                        feat.add_distances(indices=feat.select(inputs[0]), indices2=feat.select(inputs[1]))
                    elif f == 'contacts':
                        feat.add_contacts(indices=feat.select(inputs[0]), indices2=feat.select(inputs[1]))
                    elif f == 'min_dist':
                        sel1 = feat.select(inputs[0])
                        sel2 = feat.select(inputs[1])
                        
                        for s1 in sel1:
                            for s2 in sel2:
                                #pairs_sel=np.concatenate((sel1, sel2))
                                res_pairs=feat.pairs([s1,s2], excluded_neighbors=2)
                                feat.add_residue_mindist(residue_pairs=res_pairs, 
                                                 scheme='closest-heavy', 
                                                 ignore_nonprotein=False, 
                                                 threshold=0.3)
                else:
                    
                    if f == 'torsions':
                        feat.add_backbone_torsions(selstr=inputs, cossin=True)
        
                    elif f == 'positions':
                        feat.add_selection(feat.select(inputs))
          
                    elif f == 'chi':
                        for idx in inputs:
                            try:
                                feat.add_chi1_torsions(selstr=f'resid {idx}', cossin=True)
                            except ValueError:
                                pass
# =============================================================================
#                     elif f == 'RMSD':
#                         feat.add_minrmsd_to_ref(inputs)
# =============================================================================


                if len(feat.describe()) > 0: 
                    
                    dimensions=feat.dimension()
                    feature_file = f'{self.stored}/feature_{name}_{f}_stride{self.stride}_dim{dimensions}.npy'

                    if not os.path.exists(feature_file):
                        data = pyemma.coordinates.load(trajectories, features=feat, stride=self.stride)
                        print(f'Loading {f} data with stride {self.stride} ({feat.dimension()} dimensions)')
                        ft_list=open(feature_file, 'wb')
                        pickle.dump(data, ft_list)
                    else:
                        print(f"Loading featurized {f} file: {feature_file}")
                        ft_list=open(feature_file, 'rb')
                        data=pickle.load(ft_list)                            
                    
                        
                    if not get_dims:
                        features_list.append(data)           
                    else:
                        features_list.append((data, feat.dimension()))
                else:
                    print(f'Warning! Failed to featurize {f}. Check inputs and features.')
        
            except OSError:
                print('PyEMMA could not load data. Check input topologies and trajectories.')
        

        return features_list
        
        
        
    #From pyEMMA notebook
    def score_cv(self,
                 data, 
                 dim, 
                 lag, 
                 number_of_splits=10, 
                 validation_fraction=0.5):
        """Compute a cross-validated VAMP2 score.

        We randomly split the list of independent trajectories into
        a training and a validation set, compute the VAMP2 score,
        and repeat this process several times.

        Parameters
        ----------
        data : list of numpy.ndarrays
            The input data.
        dim : int
            Number of processes to score; equivalent to the dimension
            after projecting the data with VAMP2.
        lag : int
            Lag time for the VAMP2 scoring.
        number_of_splits : int, optional, default=10
            How often do we repeat the splitting and score calculation.
        validation_fraction : int, optional, default=0.5
            Fraction of trajectories which should go into the validation
            set during a split.
        """
        #print(f'\tCalculating VAMP2 scores for {dim} dimensions \n')
        nval = int(len(data) * validation_fraction)
        scores = np.zeros(number_of_splits)
        
        
        if 0 < dim <= 1:
            string=f'({dim} kinetic var.)'
        else:
            string=f'({dim} dimensions)'
        for n in range(number_of_splits):
        
            print(f'\tCalculating VAMP2 scores @lag {lag*self.timestep} {string}: cycle {n+1}/{number_of_splits}', end='\r')
            ival = np.random.choice(len(data), size=nval, replace=False)
            vamp = pyemma.coordinates.vamp([d for i, d in enumerate(data) if i not in ival], 
                                       lag=lag, 
                                       dim=dim, 
                                       stride=self.stride, 
                                       skip=self.skip, 
                                       chunksize=self.chunksize)
        
            scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])
            
            #make pretty
            if n+1 == number_of_splits:
                print(' ')

        return vamp, scores


    def clusterKmeans_standalone(self, tica, file_name, n_states):

        pyemma.config.show_progress_bars = False
        cluster_name=f'{self.stored}/{file_name}_{n_states}clusters.npy'
        if not os.path.exists(cluster_name) or self.overwrite:
            print(f'\tClustering for {n_states} cluster centres')
            clusters=pyemma.coordinates.cluster_kmeans(tica, max_iter=50, k=n_states, stride=MSM.c_stride, chunksize=self.chunksize)
            clusters.save(cluster_name, save_streaming_chain=True)
        else:
            print(f'\tCluster found for {n_states} states: ', file_name)
            clusters=pyemma.load(cluster_name)
        
        return clusters

    #from pyEMMA notebook
    def cluster_calculation(self, 
                            lags=[1,2],
                            n_centres=[2, 5, 10, 30, 75, 200, 450], 
                            n_iter=5, 
                            method='kmeans',
                            overwrite=True):
        """
        

        Parameters
        ----------
        features : TYPE
            DESCRIPTION.
        lag : TYPE, optional
            DESCRIPTION. The default is [1,2].
        n_centres : TYPE, optional
            DESCRIPTION. The default is [1, 5, 10, 30, 75, 200, 450].
        n_iter : TYPE, optional
            DESCRIPTION. The default is 7.
        method : TYPE, optional
            DESCRIPTION. The default is 'kmeans'.

        Returns
        -------
        optk_features : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.show_progress_bars = False
            
        optk_features={}      
        
        rows, columns=tools_plots.plot_layout(lags)
        fig, axes=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        try:
            subplots=axes.flat
        except AttributeError:
            subplots=axes
            subplots=[subplots]
            
        for lag, ax in zip(lags, subplots):
      
            for feature in self.features:
                name = f'{feature}@{lag*self.timestep}'
                print(name)
                
                data = self.discretized[f'{feature}@{lag*self.timestep}'] #TODO: change if self.discretized gets more fancy
                #data.n_chunks(self.chunksize)
                tica=data.get_output()
                tica_concatenated = np.concatenate(tica)
                    
                scores = np.zeros((len(n_centres),n_iter))
                
                for n, k in enumerate(n_centres):                     
                    rows_, columns_ =tools_plots.plot_layout(n_iter)
                    fig_, axes_ = plt.subplots(rows_, columns_, sharex=True, sharey=True, constrained_layout=True, figsize=(8, 6))                    
                    try:
                        subplots_=axes_.flat
                    except AttributeError:
                        subplots_=axes_
                        subplots_=[subplots_]
                    
                    for m, subplot in zip(range(n_iter), subplots_):
                        print(f'\tCalculating VAMP2 score for {k} cluster centres ({m+1}/{n_iter})', end='\r')
                        if method == 'kmeans':
                            clusters=pyemma.coordinates.cluster_kmeans(tica, max_iter=50, k=k, stride=MSM.c_stride, chunksize=self.chunksize)
                        #TODO: more methods
                        
                        try:
                            msm=pyemma.msm.estimate_markov_model(clusters.dtrajs, lag=lag, dt_traj=str(self.timestep))
                            scores[n,m]=msm.score_cv(clusters.dtrajs, n=1, score_method='VAMP2', score_k=min(10, k))
                        except Exception:
                            print(f'\tWarning! could not estimate MSM at iteration {m}/{n_iter-1}')
                            pass
                        
                        pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=subplot, cbar=False, alpha=0.3)
                        subplot.scatter(*clusters.clustercenters[:, :2].T, s=5, c='C1')
                        #subplot.set_title(f'Iteration {m+1}/{n_iter}')
                    
                    if len(subplots_) > n_iter:
                        axes_.flat[-1].axis("off")

                    fig_.subplots_adjust(wspace=0, hspace=0)
                    fig_.suptitle(f'{self.ft_name} {feature} @{self.timestep*lag} : {k} centres')
                    fig_.text(0.5, -0.02, 'IC 1', ha='center', va='center')
                    fig_.text(-0.02, 0.5, 'IC 2', ha='center', va='center', rotation='vertical')
                    fig_.tight_layout()                        

                    fig_.savefig(f'{self.results}/ClusterTICA_{self.ft_name}_{feature}_{method}_lag{self.timestep*lag}_{k}centres.png', 
                                 dpi=600, 
                                 bbox_inches="tight")
                
                lower, upper=pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
                ax.plot(n_centres, np.mean(scores, axis=1), '-o', label=feature)
                ax.fill_between(n_centres, lower, upper, alpha=0.3)
                ax.semilogx()
                ax.set_xlabel('Number of cluster centres')
                ax.set_ylabel('VAMP2 score')

                optk=dict(zip(n_centres, np.mean(scores, axis=1)))
                #print(f'\tVAMP2 scores for feature {feature}', optk)
                optk_features[f'{feature}-{lag}']=optk
                print('\n')
            ax.legend()
            ax.set_title(r'$\tau$ = '+f'{self.timestep*lag}')
        
        fig.suptitle(f'Clustering in TICA space for {self.ft_name} using {method}', weight='bold')
        fig.tight_layout()
        fig.savefig(f'{self.results}/ClusterTICA_{self.ft_name}_{method}_lag{self.timestep*lag}_stride{self.stride}.png', dpi=600, bbox_inches="tight")

                
        plt.show()        
        
        self.clusters=optk_features
        
        return optk_features
        


class MSM_dep:
    """
    Base class to create Markov state models. 
    
    """
    
    def __init__(self, systems, data=None, discretized_name='discretized', results=os.getcwd()):
        """
        

        Parameters
        ----------
        systems : dict
            The dictionary of systems.
        discretized_name : array
            The array of feature dictionaries. The default is 'None'.
        results_folder : path
            The path were to store results from methods.

        Returns
        -------
        None.

        """
        
        self.systems=systems
        self.data=data
        self.discretized_name=discretized_name
        self.results=results
        self.msms={}
        



      
    @staticmethod
    def flux(name, model, parameter_scalar=None, regions=None, labels=None, A_source=None, B_sink=None, value=None, top_pathways=2):
        """Function to calculate flux of model. A and B need to provided."""


        def setsAandB(model, scheme):
            """Retrieve the set of A and B states sampled by the model. In *combinatorial* dicretization scheme, 
            the model may not sample all states defined in A and B.
            A_filter and B_filter correspond to the set of states sampled."""
            
            states=list(model.active_set)
            
            if scheme == 'combinatorial':
                
                state_names=Functions.sampledStateLabels(regions, labels=labels)
                state_labels=[]
                                
                for v in states:
                    state_labels.append(state_names[int(v)])
                
                #print(f'{parameter} states: {state_labels}')
                
                #Remove states in A or B that are not sampled at c. The set of states will become different.
                A_filter = list(filter(lambda k: k in state_labels, A_source))
                A=([state_labels.index(k) for k in A_filter])

                B_filter = list(filter(lambda k: k in state_labels, B_sink))
                B=([state_labels.index(k) for k in B_filter])              
                
                if len(A_filter) != len(A_source) or len(B_filter) != len(B_sink):
                    print("\tWarning: not all A (source) or B (sink) states were sampled at {}".format(parameter))
                    print("\tset A indexes: {} \n\tset B indexes: {}".format(A_filter, B_filter))               
		      
            else:

                A=[states.index(states[-1])]
                B=[states.index(states[0])]
                 
            return A, B, states


        scheme, feature, parameter=name.split('-')
                
        if parameter_scalar != None:
            parameter=parameter_scalar[parameter]
                       
        if model != None:

            A, B, states =setsAandB(model, scheme)
            #print("\tset A indexes: {} \n\tset B indexes: {}".format(A,B))
         
            if len(A) != None and len(B) != None:
                
                flux=pyemma.msm.tpt(model, A, B)
                
                
                index_row=pd.MultiIndex.from_product([[scheme], [feature], [parameter]], names=['Scheme', 'feature', 'parameter'])  
                index_col=pd.MultiIndex.from_product([['Forward', 'Backward'], [s for s in states]], names=['committor', 'states'])
                
                #conversion from ps to s
                net_flux=flux.net_flux*1e12 
                rate=flux.rate*1e12
                
                flux_model=pd.DataFrame({'Net flux': np.sum(net_flux), 'Rate':rate}, index=index_row)
                    
                
                
                #Calculate commmittors
                
                f_committor=model.committor_forward(A, B)
                b_committor=model.committor_backward(A, B)
                #print(f_committor)
                
                committor_model=pd.DataFrame(index=index_row, columns=index_col)
                
                for f, b, s in zip(f_committor, b_committor, states):
                    #print(s, states.index(s))
                    committor_model.loc[(scheme, feature, parameter), ('Forward', s)]=f
                    committor_model.loc[(scheme, feature, parameter), ('Backward', s)]=b
                
                
                #Calculate the pathways	
                
                paths, path_fluxes = flux.pathways()
                path_fluxes_s=[x*1e12 for x in path_fluxes]
                path_labels=[[] for _ in paths]
                
                if scheme == 'combinatorial':    
                    label_names=Functions.sampledStateLabels(regions, sampled_states=states, labels=labels)
                else:
                    label_names=[str(s) for s in states]
                
                    
                for k, v in enumerate(paths):
                    for p in v:
                        path_labels[k].append(label_names[p])   
                    
                path_labels=[' -> '.join(k) for k in path_labels]


                path_col=pd.MultiIndex.from_product([path_labels], names=['pathway'])
                pathway_model=pd.DataFrame(index=index_row, columns=path_col)
                    
                values_top=np.min(path_fluxes_s[:top_pathways]) #WARNING: this is only valid since array of fluxes is already sorted.
                
                for label, path_flux in zip(path_labels, path_fluxes_s):
                    if path_flux >= values_top: 
                        pathway_model.loc[(scheme, feature, parameter), label]= path_flux
                       
                
                return flux_model, committor_model, pathway_model
                
            else:
                print("No A or B states sampled")
             
        
        else:
            print("No model found.")
            
            return None, None, None
        

    def MFPT(self, model):
        """Calculation of mean first passage times for all models.
        TODO: Method requires nested loops (i,j) to fill each cell of the MFPT matrix. 
        TODO: Check deprecated for list compreehension method (does not work multiIindex df)."""
        
        if model == None:
            print('no model')
        
        else:
            states=model.active_set
            
            index_row=pd.MultiIndex.from_product([[self.scheme], [self.feature], [s for s in states]], 
                                                 names=['Scheme', 'feature', 'state_source'])
            index_col=pd.MultiIndex.from_product([[self.parameter], [s for s in states], ['mean', 'stdev']], names=['parameter', 'state_sink', 'values'])
            
            mfpt=pd.DataFrame(index=index_row, columns=index_col)
            
            for i, state_source in enumerate(states):
                for j, state_sink in enumerate(states):
                    mfpt.loc[(self.scheme, self.feature, state_source), (self.parameter, state_sink, 'mean')]= model.sample_mean("mfpt", i,j)
                    mfpt.loc[(self.scheme, self.feature, state_source), (self.parameter, state_sink, 'stdev')]= model.sample_std("mfpt", i,j)
            
            return mfpt
        
@staticmethod
def mfpt_filter(mfpt, error):
        """Function to filter out MFPT values whose standard deviations are above *error* value.
        Default value of *error* is 20%"""
        
        #mfpt=mfpt_df.loc[(scheme, feature), (parameter)]
        mfpt.dropna(how='all', inplace=True)
        mfpt.dropna(how='all', axis=1, inplace=True)
            
            
        means=mfpt.iloc[:, mfpt.columns.get_level_values(level='values')=='mean']
        stdevs=mfpt.iloc[:, mfpt.columns.get_level_values(level='values')=='stdev']
        
        #print('Before: ', means.isna().sum().sum())
        counts=0
        ####Filter the mean values to the error percentage of stdev
        for mean, stdev in zip(means, stdevs):
            ratio = mfpt[stdev] / mfpt[mean]
            for r, v in zip(ratio, means[mean]):
                if r >= error:
                    index_means=means.index[means[mean] == v]
                    means.loc[index_means, mean] =np.nan
                    counts+=1
                        
        means.dropna(how='all', inplace=True)
        means.dropna(how='all', axis=1, inplace=True)
            
        #print('After: ', means.isna().sum().sum())
        #print('Counts: ', counts)
            
        #minimum_notzero, maximum= means[means.gt(0)].min().min(), means.max().max()
        
        return means
    