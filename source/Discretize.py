# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:59:08 2021

@author: hcarv
"""

#SFv2 modules
try:
    import tools
    import tools_plots
except:
    print('Could not load SFv2 modules')

#python modules
import os
import numpy as np
import uncertainties.unumpy as unumpy 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import glob
import collections
from multiprocessing import Process, Pool


import itertools
from tqdm import tqdm as tq
import matplotlib.pylab as pl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
pd.options.plotting.backend = 'matplotlib' #'matplotlib'
from scipy.optimize import curve_fit

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import f_regression, mutual_info_classif

class Discretize:
    
    
    
    
    
    """Base class to discretize features. Takes as input a dictionary of features, 
    each containing a dictionary for each parameter and raw_data"""
    
    
    def_locs=(1.25, 0.6)
    def_locs_multi=(1, 0.5)
    #headers= [0,1,2,3,4,5] #might not be 7
    headers = {'shellprofile' : [0,1,2,3,4,5], 'combinatorial' :  [0,1,2,3,4,5], 'dG' : [0,1,2,3,4,5,6,7]} #might not be 7
    #TODO: featurize module also needs to have this. move to tools or main for unique setup
    level = 2
    rep_level = 3
    ref_level = 4
    sel_level = 5
    
    
    def __init__(self, 
                 project,  
                 results=None, 
                 systems=None, 
                 scalars=[], 
                 method=None,
                 feature_name='undefined',
                 mode='pipeline',
                 overwrite=False):
        """
        

        Parameters
        ----------
        project : TYPE
            DESCRIPTION.
        data : TYPE, optional
            DESCRIPTION. The default is None.
        results : TYPE, optional
            DESCRIPTION. The default is None.
        systems : TYPE, optional
            DESCRIPTION. The default is None.
        scalars : TYPE, optional
            DESCRIPTION. The default is [].
        feature_name : TYPE, optional
            DESCRIPTION. The default is 'undefined'.
        mode : TYPE, optional
            DESCRIPTION. The default is 'pipeline'.

        Returns
        -------
        None.

        """

        self.project = project
        self.feature_name=feature_name
        if results != None:
            self.results = tools.Functions.pathHandler(self.project, results)
        else:
            self.results = self.project.results
        self.discretized={}
        self.scalars=scalars
        self.timestep_=''.join(str(self.project.timestep).split(' ')) 
        
        if systems is not None:
           self.systems =systems 
        else:
            self.systems=project.systems

        #self.data = data
        self.overwrite = overwrite
        self.method = method
        self.multi_process = True
        print('Results will be stored under: ', self.results)

        
        #print(f'Input feature: {self.feature_name} \nData: \n {self.data}')

    # =============================================================================
    #             if stride != 1:
    #                 stride=int((it_df.index[-1]-it_df.index[0])/(len(it_df.index.values)-1))
    # =============================================================================  

    
    def combinatorial(self, 
                      shells,
                      start=0,
                      stop=-1,
                      stride=1,
                      subset=None,
                      labels=None,
                      plot_refs=True,
                      plot_trajectories=False,
                      get_sampled_states=False,
                      reconstruct=False):
        """
        
        Function to generate combinatorial encoding of shells into states. Discretization is made based on combination of shells. 
        Not sensitive to which shell a given molecule is at each frame. Produces equal-sized strings for all parameters.
        Static methods defined in *tools.Functions*  are employed here.

        Parameters
        ----------
        shells : array
            The array of shell boundaries.
        level : TYPE, optional
            DESCRIPTION. The default is 3.
        start : TYPE, optional
            DESCRIPTION. The default is 0.
        stop : TYPE, optional
            DESCRIPTION. The default is -1.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.
        labels : TYPE, optional
            The shell labels (+1 shells). Method will resolve inconsistencies by reverting to numeric label.

        Returns
        -------
        feature_df : dataframe
            Dataframe of discretized features.

        """
        #TODO: fix this to also allow loading from input_df. this was the original behaviour
        self.start = start
        self.stop = stop
        self.stride = stride
        self.disc_scheme = 'combinatorial'
        if subset == None:
            iterables = self.project.parameter
        else:
            iterables = subset     
        
        shells_name=''.join(labels)
    
        supra_df = pd.DataFrame()
        
        state_labels = []
        state_indexes = []
        for idx, iterable in enumerate(iterables):
            
            if reconstruct:
                df_name=f'discTraj_{self.method}_{self.feature_name}_{start}_{stop}_{self.timestep_}_{stride}.csv' 
            else:
                df_name=f'{self.method}_{self.feature_name}*_{start}_{stop}_{self.timestep_}_{stride}.csv' 
            
            it_name = f'{self.results}/combinatorial_{self.feature_name}_{iterable}_{shells_name}.csv'
            
            if not os.path.exists(it_name) or self.overwrite:
                it_df = pd.DataFrame()
                
                for rep in range(self.project.initial_replica,self.project.replicas+1):
                    rep_disc = pd.DataFrame()
                    try:
                        headers = Discretize.headers[self.disc_scheme]
                        input_rep_df = self.get_featurized_data(iterable, df_name, header= headers, replicate=rep, reconstruct=reconstruct)
                    except pd.errors.ParserError as v:
                        print(v)
                        headers = Discretize.headers[self.disc_scheme][:-1]
                        input_rep_df = self.get_featurized_data(iterable, df_name, header= headers, replicate=rep, reconstruct=reconstruct)


                    if plot_refs:
                        
                        refs = input_rep_df.columns.get_level_values(Discretize.ref_level).unique()
                        colors = pl.cm.tab10(np.linspace(0,1,len(refs)))
                        for idx_r, ref in enumerate(refs):
                            label_ref = 'chain_'+chr(ord('@')+idx_r+1)
                            print(f'\tCalculating {iterable}:{rep}-{label_ref} ', end='\r')
                            input_ref_df = input_rep_df.loc[:, input_rep_df.columns.get_level_values(Discretize.ref_level) == ref] 
                            if reconstruct:
                                state_series = input_ref_df
                                ref_disc = input_ref_df
                            else:
                                state_series=tools.Functions.state_mapper(shells, input_ref_df, labels=labels)
                                levels = [input_ref_df.columns.get_level_values(x).unique() for x in range(Discretize.ref_level+1)]
                                #levels.append(pd.Index([label_ref], name='Reference'))
                                sel = input_ref_df.columns.get_level_values(Discretize.sel_level).unique()
                                
                                label_sel = f'sel_1-{len(sel)}'
                                levels.append(pd.Index([label_sel], name=sel.name))
                                index_col = pd.MultiIndex.from_arrays(levels)

                                ref_disc=pd.DataFrame(state_series, index=input_ref_df.index, columns=index_col)
                            rep_disc=pd.concat([rep_disc, ref_disc], axis=1)

                    elif not plot_refs and not reconstruct:
                        print(f'\tCalculating {iterable}:{rep}', end='\r')
                        state_series=tools.Functions.state_mapper(shells, input_rep_df, labels=labels)
                        levels = [input_rep_df.columns.get_level_values(x).unique() for x in range(Discretize.rep_level)]
                        
                        index_col = pd.MultiIndex.from_arrays(levels)
                        rep_disc =pd.DataFrame(state_series, index=input_rep_df.index, columns=index_col)
                        #rep_disc=pd.concat([rep_disc, rep_disc_], axis=1)

                    it_df = pd.concat([it_df, rep_disc], axis=1)
                
            else:
                print('Loading ', it_name, end='\r')
                try:
                    it_df = pd.read_csv(it_name, index_col=0, header=Discretize.headers[self.disc_scheme])
                except Exception as v:
                    print(v.__class__, v)
                    #pd.errors.ParserError:
                    it_df = pd.read_csv(it_name, index_col=0, header=Discretize.headers['shellprofile'])
            self.discretized[f'combinatorial_{self.feature_name}_{iterable}_{shells_name}']= it_df
            
            supra_df = pd.concat([supra_df, it_df], axis=1)
            it_df.to_csv(it_name)
        
        
        state_indexes = set(supra_df.values.flat)
        state_labels = tools.Functions.sampledStateLabels(shells, labels, sampled_states=state_indexes)
        states = {j : i for i, j in zip(state_labels, state_indexes)}
        
        if get_sampled_states is True:
            
            return supra_df, states
        
        elif get_sampled_states == 'remap':
            
            return tools.Functions.remap_states(supra_df, states)
        
        else:
            return supra_df


    def get_featurized_data(self, iterable, df_name, header=None, reconstruct=False, replicate=None):      


        if replicate != None:
            file = [glob.glob(f'{system.results_folder}/{df_name}')[0] for system in self.systems.values() if system.replicate == replicate and system.parameter == iterable]
            if len(file) == 1:
                out_df = pd.read_csv(file[0], index_col=0, header=header)
            else:
                print('Warning! ', file)
        
              
        else:      
            out_df = pd.DataFrame()
            df_list = [glob.glob(f'{system.results_folder}/{df_name}')[0] for system in self.systems.values() if system.parameter == iterable]
            #TODO: make robust to start/stop frame definitions. Currently looking for index, not value (ps)
            for file in df_list:
                if not isinstance(file, str):
                    print(file)
                    file = file[input('Which one (index)?')]
                
                out_df = pd.concat([out_df, pd.read_csv(file, index_col=0, header=header)], axis=1)

        if not reconstruct:
            out_df = out_df.iloc[self.start:self.stop:self.stride, :]
 
        return out_df


    def shell_profile(self,  
                    thickness=0.5,
                    limits=(0,150),
                    start=0,
                    stop=-1,
                    stride=1,
                    subset=None,
                    n_cores=-1,
                    chunksize = 1000,
                    multi_process=True,
                    reconstruct=False):
        """
        Generate the discretization feature into shells=(min("limits"), max("limits"), "thickness"). 

        Parameters
        ----------
        thickness : TYPE, optional
            DESCRIPTION. The default is 0.5.
        limits : TYPE, optional
            DESCRIPTION. The default is (0,150).
        level : int, optional
            The level for data agreggation. The default is 2 (molecule).
        labels : TYPE, optional
            DESCRIPTION. The default is None.
        shells : TYPE, optional
            DESCRIPTION. The default is None.
        n_cores : int
            The number of processes.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        self.start = start
        self.stop = stop
        self.stride = stride
        
        if n_cores == 1:
            multi_process = False

        feature_range=np.arange(limits[0], limits[1],thickness)
        feature_center_bin=feature_range+(thickness/2)                
            
        df_name=f'{self.method}_{self.feature_name}_{start}_{stop}_{self.timestep_}_{stride}.csv' 
        
        if subset == None:
            subset = self.project.parameter
        else:
            for s in subset:
                if not s in self.project.parameter:
                    raise ValueError('Subset not defined under Project parameters')
        
        for iterable in subset:
            shell_profile = f'{self.results}/shellProfile_{self.feature_name}_{iterable}_d{thickness}_b{start}_e{stop}_s{stride}.csv'
            if not os.path.exists(shell_profile) or self.overwrite:
                
                if reconstruct:
                    print(f'Reconstructuing shell profile of {iterable} dS = {thickness} Angstrom')
                    iterable_df_disc = self.get_featurized_data(iterable, df_name, reconstruct=reconstruct)
                else:
                    print(f'Generating shell profile of {iterable} dS = {thickness} Angstrom')
                    iterable_df = self.get_featurized_data(iterable, df_name) #will retrieve proper start stop
                    iterable_df_disc=pd.DataFrame(index=feature_center_bin[:-1], columns=iterable_df.columns) #iterable_df
                    values=[(iterable_df[value], value) for value in iterable_df.columns] # iterable_df[value] values_]
                    fixed=(feature_range, feature_center_bin[:-1])
        
                    if multi_process:
                        value_list=[values[i:i + chunksize] for i in range(0, len(values), chunksize)]
                        for values in value_list:
                            shells=tools.Tasks.parallel_task(Discretize.shell_calculation, 
                                                         values, 
                                                         fixed, 
                                                         n_cores=n_cores)
                            for shell in shells:
                                iterable_df_disc=pd.concat([iterable_df_disc, shell], axis=1)
                    else:
                        for idx, series_value in enumerate(values, 1):
                            iterable_df_disc=[pd.concat([iterable_df_disc, shell], axis=1) for shell in self.shell_calculation(series_value, fixed)]
                print('\n')   
                iterable_df_disc.dropna(axis=1, inplace=True)
                iterable_df_disc.to_csv(shell_profile)
                
            else:
                print('Loading :', shell_profile)
                iterable_df_disc = pd.read_csv(shell_profile, index_col=0, header=Discretize.headers)
        self.discretized[f'ShellProfile_{self.feature_name}_{iterable}']=iterable_df_disc
        
        print('Discretization updated: ', [discretized for discretized in self.discretized.keys()])
        
        return self.discretized
    
    
    @staticmethod
    def shell_calculation(series_value, specs=()):
        """
        The workhorse function for shell calculation.
        Returns a 1D histogram as a Series using parameters handed by specs (ranges, center of bins).

        Parameters
        ----------
        series_value : TYPE
            DESCRIPTION.
        specs : TYPE, optional
            DESCRIPTION. The default is ().

        Returns
        -------
        hist_df : TYPE
            DESCRIPTION.

        """
        
        (feature_range, index)=specs
        (series, value)=series_value
        
        names=[f'l{i}' for i in range(1, len(series.name)+1)] 
        column_index=pd.MultiIndex.from_tuples([value], names=names)
        
        hist, _=np.histogram(series, bins=feature_range)
        hist_df=pd.DataFrame(hist, index=index, columns=column_index)
        
        return hist_df


    def dG_calculation(self,
                       start=0,
                       stop=-1,
                       stride=1,
                       n_cores=-1, 
                       bulk=(30, 41), 
                       subset=None,
                       feature_name=None, 
                       describe='mean',
                       plot_multi = True,
                       plot_replicas = True,
                       quantiles=[0.01,0.5,0.75,0.99],
                       plot_all=False,
                       bulk_solvent=None,
                       sel_range=False):
        """
        Function to get free energy values from multiple shell profiles (1D histogram). 
        NOTE: The "ranges_bulk" should be carefully chosen for each system.
        
        
        NOTES:
        Calculates the theoretical NAC values in "bulk" in a spherical shell of size given by "ranges". 
        The output is N - 1 compared to N values in ranges, so an additional value of "resolution" is added for equivalence.
        N_ref= lambda ranges, bulk_value: np.log(((4.0/3.0)*np.pi)*np.diff(np.power(np.append(ranges, ranges[-1] + resolution), 3.0))*6.022*bulk_value*factor)

        

        Parameters
        ----------
        input_df : TYPE, optional
            DESCRIPTION. The default is None.
        start : TYPE, optional
            DESCRIPTION. The default is 0.
        stop : TYPE, optional
            DESCRIPTION. The default is -1.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.
        n_cores : TYPE, optional
            DESCRIPTION. The default is -1.
        bulk : TYPE, optional
            DESCRIPTION. The default is (30, 41).
        level : TYPE, optional
            DESCRIPTION. The default is 2.
        resolution : TYPE, optional
            DESCRIPTION. The default is 0.5.
        shells : TYPE, optional
            DESCRIPTION. The default is [].
        feature_name : TYPE, optional
            DESCRIPTION. The default is None.
        describe : TYPE, optional
            DESCRIPTION. The default is 'mean'.
        quantiles : TYPE, optional
            DESCRIPTION. The default is [0.01,0.5,0.75,0.99].

        Returns
        -------
        None.

        """  

        self.describe = describe
        self.quantiles = quantiles
        self.start = start
        self.stop = stop
        self.stride = stride
        self.subset = subset
        self.bulk_solvent = bulk_solvent
        self.bulk = bulk
        self.disc_scheme = 'dG'
        
        self.binding_profile_path = f'{self.results}/binding_profile'
        

        if feature_name != None:
            
            self.feature_name = feature_name

        iterables = self.get_iterable_data()

        rows, columns=tools_plots.plot_layout(iterables)
        
        name_original=r'N$_{(i)}$reference (initial)'
        name_fit =r'N$_{(i)}$reference (fit)'
        
        fig_fits,axes_fit=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(10,8))
        try:
            axes_fit =axes_fit.flat
        except AttributeError:
            axes_fit=[axes_fit]
        
        dG_fits=pd.DataFrame()
        
        if plot_all:
            fig_dG, axes_dG = plt.subplots(1,1, sharex=True, sharey=True, constrained_layout=True, figsize=(8,6))
            colors_all = pl.cm.YlGnBu(np.linspace(0.2,1,len(iterables)))
        else:
            fig_dG,axes_dG=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(8,6))
            try:
                axes_dG=axes_dG.flat
            except AttributeError:
                axes_dG=[axes_dG]
            
        
        if plot_replicas:
            reps = iterables[0][1].columns.get_level_values(3).unique()
            fig_rep_dG,axes_rep_dG = plt.subplots(len(reps), 
                                                  len(iterables), 
                                                  sharex=True, 
                                                  sharey=True, 
                                                  constrained_layout=True, 
                                                  figsize=(int(3*len(iterables)),int(1*len(reps))))
            #fig_rep_fits,axes_rep_fit=plt.subplots(len(reps), len(iterables), sharex=True, sharey=True, constrained_layout=True, figsize=(14,10))
            
        
        def _fit(df):
            N_enz, N_error_p, N_error_m  = self.get_sim_counts(df, iterable, quantiles, describe)            
            N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit = self.get_fittings(iterable, 
                                                                                         N_enz, 
                                                                                         ranges, 
                                                                                         bulk_range, 
                                                                                         resolution, 
                                                                                         name_fit)
            return N_enz, N_error_p, N_error_m, N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit
            
        
        def bulk_fitting(iterable_df, ax, label=None, transp=0.5, color='green'):
            if label != None:
                iterable_ = f'{iterable}-{label}'
            else:
                iterable_ = iterable
               
            N_enz, N_error_p, N_error_m, N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit = _fit(iterable_df)
            
            ax.plot(ranges, N_t, label=r'N$_{(i)}$reference (initial)', color='red', ls='--')
            ax.plot(ranges, N_opt, label=r'N$_{(i)}$reference (fit)', color='black')
            if describe == 'mean': 
                
                #err = [N_error_m.values, N_error_p.values]
                #ax.errorbar(ranges, N_enz.values.flat, yerr=err, fmt='.', color=color, label='N$_{(i)}$enzyme')
                ax.plot(ranges, N_enz, '.', label='N$_{(i)}$enzyme', color=color) #[:len(ranges)]
                ax.fill_betweenx([np.min(N_enz), np.max(N_enz_fit)], bulk_range[0], bulk_range[-1], label='Bulk', color='grey', alpha=0.5)
                ax.fill_between(ranges, N_error_m, N_error_p, color=color, alpha=transp)
            elif describe == 'quantile': 
                ax.plot(ranges, N_enz, label='N$_{(i)}$enzyme', color='orange')
                
                for idx, m in enumerate(N_error_m, 1):
                    ax.plot(ranges, N_error_m[m], alpha=1-(0.15*idx), label=m, color='green')
                for idx, p in enumerate(N_error_p, 1):
                    ax.plot(ranges, N_error_p[p], label=p, alpha=1-(0.15*idx), color='red') 
        
            ax.set_yscale('log') #, base=2)
            ax.set_xscale('log') #, base=2)

            
            ax.set_title(f'{iterable_}\n({np.round(fitted_bulk, decimals=1)} {unit})')
            
            return ax

        
        def dG_plot(iterable_df, dG_fits, ax, title=None, label=None, transp= 0.5, color='green', multi_plot=False):

            if title is None:
                title = iterable
            if label is None:
                label = iterable
                
                
            N_enz, N_error_p, N_error_m, N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit = _fit(iterable_df)
                
            #Calculate dG (kT)
            (a, a_p, a_m, b) = [np.log(i) for i in [N_enz, N_error_p, N_error_m, N_opt]]                
            dG=pd.DataFrame({f'$\Delta$G {title}': np.negative(a - b)}) 
            dG_err_m=np.negative(a_m.subtract(b, axis='rows'))
            dG_err_p=np.negative(a_p.subtract(b, axis='rows')) 
            
            theoretic_df=pd.DataFrame({f'{name_original} {title}':N_t, 
                                       f'{name_fit} {title}':N_opt}, 
                                       index=N_enz.index.values)                
            dG_fit=pd.concat([dG, dG_err_m, dG_err_p, N_enz, theoretic_df], axis=1)   
            
            dG_, dG_err_m_, dG_err_p_ = dG.loc[:bulk_max+10], dG_err_m.loc[:bulk_max+10], dG_err_p.loc[:bulk_max+10]

            if describe == 'mean':
                
                ax.plot(dG_.index, dG_.values, '.', color=color, label=label)
                if not multi_plot:
                    ax.fill_between(dG_.index, dG_err_m_.values, dG_err_p_.values, alpha=transp, color=color)

                
            #Fix quantile to multi ref plotting.
            elif describe == 'quantile':
                
                ax.plot(ranges, dG, color='orange')
                
                for idx, m in enumerate(dG_err_m, 1):
                    ax.plot(ranges, dG_err_m[m], alpha=1-(0.15*idx), label=m, color='green')
                for idx, p in enumerate(dG_err_p, 1):
                    ax.plot(ranges, dG_err_p[p], alpha=1-(0.15*idx), label=p, color='red') 
                    
            ax.axhline(y=0, ls='--', color='darkgray')
            ax.set_xscale('log') #, base=2)
            if multi_plot:
                ax.set_title(title, y=1.0, pad=-14) 
            else:
                ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
                ax.xaxis.set_major_formatter(ScalarFormatter())

            return dG_fit, ax


        #TODO: check scalars
        #scalars = tools.Functions.setScalar(iterables, ordered=False, get_uniques=True)
        for idx_, (it, ax_fit) in enumerate(zip(iterables, axes_fit)):
            (iterable, it_df) = it
            dG_it_name = f'{self.binding_profile_path}/binding_profile_{self.feature_name}_{describe}_b{start}_e{stop}_s{stride}_{iterable}.csv'
            if not os.path.exists(dG_it_name):
            
                ranges=it_df.index.values
                resolution = float(ranges[1]) - float(ranges[0])  
                if isinstance(bulk, tuple):
                    (bulk_min, bulk_max) = bulk
                elif isinstance(bulk, dict):
                    try:
                        (bulk_min, bulk_max) =  bulk[iterable]
                    except KeyError:
                        raise KeyError('No bulk range provided for ', iterable)
                bulk_range=np.arange(min(ranges, key=lambda x:abs(x-bulk_min)), min(ranges, key=lambda x:abs(x-bulk_max)), resolution)
                #bulk_range=np.arange(bulk_min, bulk_max, resolution)
                
                #TODO: get shell labels and limits from shell_profile. also in dG_plot.
                state_labels=['A', 'P', 'E', 'S', 'B']
                iterable_df = self.get_descriptors(it_df, iterable, sel_range=sel_range)
                bulk_fitting(iterable_df, ax_fit)
                
                if plot_all:
                    ax_dG = axes_dG
                    color = colors_all[idx_]
                else:
                    ax_dG = axes_dG[idx_]
                    color = 'green'
                
                dG_fit, ax_dG = dG_plot(iterable_df, dG_fits, ax_dG, color=color, transp=0.6)
    
                self.discretized[f'dGProfile_{self.feature_name}_{iterable}_{describe}']=dG_fit
                if plot_replicas:
                    
                    rep_dfs = pd.DataFrame()
                    
                    for idx, rep in enumerate(reps): 
                        rep_df = it_df.loc[:, it_df.columns.get_level_values(Discretize.rep_level) == rep]
                        iterable_df_rep = self.get_descriptors(rep_df, iterable, sel_range=sel_range) 
                        rep_dfs = pd.concat([rep_dfs, iterable_df_rep], axis=1)
    
                        if plot_multi:
                            try:
                                ax_dG_multi = axes_rep_dG[idx, idx_]
                            except IndexError:
                                ax_dG_multi = axes_rep_dG[idx]
                            except TypeError:
                                ax_dG_multi = axes_rep_dG
                            refs = rep_df.columns.get_level_values(Discretize.ref_level).unique()
                            colors = pl.cm.tab10(np.linspace(0,1,len(refs)))
                            for idx_r, ref in enumerate(refs):
                                
                                label = 'chain_'+chr(ord('@')+idx_r+1)
                                print(f'Processing {iterable}:{rep}-{label}', end='\r')
                                ref_df = rep_df.loc[:, rep_df.columns.get_level_values(Discretize.ref_level) == ref]
                                iterable_df_multi = self.get_descriptors(ref_df, iterable, sel_range=sel_range)
                                #N_enz, N_error_p, N_error_m, N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit = _fit(iterable_df_multi)
                                #bulk_fitting(iterable_df_multi, axes_ref_fit[idx, idx_], label=f'Replica {idx_r+1}', transp=0.2, color='darkgrey')
                                dG_fit_multi, ax_dG_rep = dG_plot(iterable_df_multi, 
                                                                  dG_fits, 
                                                                  ax_dG_multi,
                                                                  title=f'{iterable}-{rep}',
                                                                  label=label, 
                                                                  transp=0.2, 
                                                                  color=colors[idx_r],
                                                                  multi_plot=True)
                                
                                self.discretized[f'dGProfile_{self.feature_name}_{iterable}_{rep}_{ref}_{describe}']=dG_fit_multi
    # =============================================================================
    #                     else:
    #                         print(f'Processing {iterable}:{rep}-{label}', end='\r')
    #                         bulk_fitting(iterable_df_multi, ax_fit, label=rep, color=colors[rep])
    #                         dG_fit_rep,ax_dG_rep = dG_plot(iterable_df_rep, 
    #                                                        dG_fits, 
    #                                                        axes_rep_dG[idx, idx_], 
    #                                                        label=rep, 
    #                                                        color='black')    
    #                         self.discretized[f'dGProfile_{self.feature_name}_{iterable}_{rep}_{describe}']=dG_fit_rep
    # =============================================================================
    
                dG_fit.to_csv(dG_it_name)
            else:
                print('Loading DG of ', iterable)
                dG_fit = pd.read_csv(dG_it_name, index_col=0, header=[0])
        
            dG_fits=pd.concat([dG_fits, dG_fit], axis=1)
        
            
        dG_fits.to_csv(f'{self.binding_profile_path}/binding_profile_{self.feature_name}_{describe}_b{start}_e{stop}_s{stride}.csv')
        
        if not axes_fit[-1].lines: 
            axes_fit[-1].set_visible(False)
        try: 
            if not axes_dG[-1].lines: 
                axes_dG[-1].set_visible(False) 
        except:
            if not axes_dG.lines: 
                axes_dG.set_visible(False)     
        
        handles, labels = ax_fit.get_legend_handles_labels()
        fig_fits.subplots_adjust(wspace=0, hspace=0)
        fig_fits.legend(handles, labels, bbox_to_anchor=Discretize.def_locs)
        try:
            fig_fits.supxlabel(r'$\itd$$_{NAC}$ ($\AA$)')
            fig_fits.supylabel(r'ln $\itN$')
        except:
            fig_fits.text(0.5, 0.00, r'Shell $\iti$ ($\AA$)')
        fig_fits.suptitle(f'Feature: {self.feature_name}\n{describe}')
        fig_fits.tight_layout()
        fig_fits.show()
        fig_fits.savefig(f'{self.binding_profile_path}/binding_profile_{self.feature_name}_{describe}_b{start}_e{stop}_s{stride}_fittings.png', dpi=600, bbox_inches="tight")
        if plot_replicas:
            #fig_ref_dG.legend(bbox_to_anchor=Discretize.def_locs)
            fig_rep_dG.suptitle(f'Feature: {self.feature_name}\n{describe}')
            try:
                fig_rep_dG.supxlabel(r'$\itd$$_{NAC}$ ($\AA$)')
                fig_rep_dG.supylabel(r'$\Delta$G ($\it{k}$$_B$T)')
            except:
                fig_rep_dG.text(0.5, 0.00, r'$\itd$$_{NAC}$ ($\AA$)')
            fig_rep_dG.suptitle(f'Feature: {self.feature_name}\n{describe}')
            fig_rep_dG.subplots_adjust(wspace=0, hspace=0)
            #handles_, labels_ = ax_dG_rep.get_legend_handles_labels()
            #fig_rep_dG.legend(handles_, labels_, loc='center right') #, ncol=4) #, bbox_to_anchor=Discretize.def_locs_multi)
            #fig_rep_dG.tight_layout()
            fig_rep_dG.show()
            fig_rep_dG.savefig(f'{self.binding_profile_path}/binding_profile_{self.feature_name}_refs_{describe}_b{start}_e{stop}_s{stride}.png', dpi=600, bbox_inches="tight")

        #handles_dG, labels_dG = ax_dG.get_legend_handles_labels()
        #fig_dG.legend(handles_dG, labels_dG, bbox_to_anchor=Discretize.def_locs)
        
        #fig_dG.subplots_adjust(wspace=0, hspace=0)
        #plt.tick_params(axis='y', which='minor')

        try:
            fig_dG.supxlabel(r'$\itd$$_{NAC}$ ($\AA$)')
            fig_dG.supylabel(r'$\Delta$G ($\it{k}$$_B$T)')
        except:
            fig_dG.text(0.5, 0.0, r'$\itd$$_{NAC}$ ($\AA$)')
        #fig_dG.suptitle(f'Feature: {self.feature_name}\n{describe}')
        if plot_all:
            fig_dG.legend(title=self.project.ligand[0]) #loc='upper right')
        fig_dG.show()
        fig_dG.savefig(f'{self.binding_profile_path}/binding_profile_{self.feature_name}_{describe}_b{start}_e{stop}_s{stride}.png', dpi=600, bbox_inches="tight")
        
        
        print(' ', end='\r')
        return dG_fits


        
        



    def get_iterable_data(self):
        #TODO: fix this to also allow loading from input_df. this was the original behaviour

        if self.subset != None:
            for s in self.subset:
                if not s in self.project.parameter:
                    print('Subset not defined under Project parameters')
                    raise ValueError() 
            
            iterables = self.subset
        else:
            iterables = self.project.parameter
            
        iterables_to_process = []
        for iterable in iterables:
            shell_df_name = f'{self.results}/shellProfile_{self.feature_name}_{iterable}_d*_b{self.start}_e{self.stop}_s{self.stride}.csv'
            file = glob.glob(shell_df_name)
            if len(file) > 1:
                print(file)
                file = file[int(input(f'Multiple files found for {iterable}. Which one (index)?'))]
            elif len(file) == 0:
                raise FileNotFoundError(f'ERROR: {shell_df_name} not found')
            else:
                file = file[0]
                print('Loading:', file, end='\r')
            input_df = pd.read_csv(file, index_col=0, header=Discretize.headers[self.disc_scheme])
            input_df.iloc[self.start:self.stop:self.stride, :]
            iterables_to_process.append((iterable, input_df))

        
        return iterables_to_process


    
    def get_descriptors(self, df_it, name, sel_range=False):
        """
        

        Parameters
        ----------
        df_it : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.

        Returns
        -------
        descriptor : TYPE
            DESCRIPTION.

        """


        levels = df_it.columns.levels
        replicas = len(levels[3])
        references = len(levels[4])
        if sel_range:
            n_sels = levels[5].values
            selections = np.mean([int(n.split('-')[1]) for n in n_sels])
        else:
            selections = len(levels[5])
        counts = max(df_it.sum(axis=0)) 
        frames = counts / selections
        #print(max(df_it.sum())/40000, max(df_it.sum(axis=1))/40000)
        #print(replicas, references, selections, counts, frames)
        

        if self.describe == 'single':
            descriptor=df_it.quantile(q=0.5) / frames

        elif self.describe == 'mean':
            descriptor=pd.DataFrame()
            mean=df_it.mean(axis=1) / frames 
            mean.name=name
            sem=df_it.sem(axis=1)  / frames
            sem.name=f'{name}-St.Err.'
            descriptor=pd.concat([mean, sem], axis=1)
        
        elif self.describe == 'quantile':
            descriptor=pd.DataFrame()                    
            for quantile in self.quantiles:        
                descriptor_q=df_it.quantile(q=quantile, axis=1)/ frames
                descriptor_q.name=f'Q{quantile}'
                descriptor=pd.concat([descriptor, descriptor_q], axis=1)
                
    
        else: #TODO: make more elif statements if need and push this to elif None.
            descriptor=df_it
    
        #descriptor.name=name           
        
        return descriptor
    
    
    
    
       
    def get_fittings(self, iterable, N_enz, ranges, bulk_range, resolution, name_fit):
        
        
        
        bulk_min, bulk_max=bulk_range[0], bulk_range[-1]
        #ranges[np.abs(ranges - bulk_range[0]).argmin()], ranges[np.abs(ranges - bulk_range[-1]).argmin()]

        if self.bulk_solvent != None:
            iterable = self.bulk_solvent
        #get the value (in M) of the concentration from the string
        try:
            bulk_value=float(str(iterable).split('M')[0]) 
            factor=1e-4
            unit='M'
        except:
            bulk_value=float(str(iterable).split('mM')[0])
            factor=1e-7
            unit='mM'
        #print(f'Bulk value: {bulk_value} {unit}')
        
        
        def N_ref(ranges, bulk_value):

            spherical=np.log(((4.0/3.0)*np.pi))
            sphere=np.diff(np.power(np.append(ranges, ranges[-1] + resolution), 3.0))
            c_normal=6.022*bulk_value*factor            
            out=spherical*sphere*c_normal

            return out*3 # The 3 factor Niels Hansen was talking about
        
        
        #Region of P_nac values to be used for fitting. 
        N_enz_fit=N_enz.loc[bulk_min:bulk_max].values
        #N_enz_fit=N_enz.iloc[ranges.index(bulk_min):ranges.index(bulk_max)].values
        #original theoretical distribution with predefined bulk
        
        N_t=N_ref(ranges, bulk_value)   
        
        #Optimize the value of bulk with curve_fit of nac vs N_ref. 
        #Initial guess is "bulk" value. Bounds can be set as  +/- 50% of bulk value.
        try:

            c_opt, c_cov = curve_fit(N_ref, 
                                 bulk_range, #[:-1], 
                                 N_enz_fit, 
                                 p0=bulk_value)
                                 #bounds=(bulk_value - bulk_value*0.5, bulk_value + bulk_value*0.5))
            stdev=np.sqrt(np.diag(c_cov))[0]
        except Exception as v:
            print('Warning! ', v.__class__, v)
            c_opt = bulk_value
            stdev = 0
        
        
        fitted_bulk=c_opt[0]
        #print(f'The fitted bulk value is: {np.around(fitted_bulk, decimals=3)} +/- {np.around(stdev, decimals=3)} {unit}  ({np.around(fitted_bulk/bulk_value, decimals=3)} error)')  
        
        #Recalculate N_ref with adjusted bulk concentration.
        

        
        if len(ranges) != len(N_enz.index.values):
            print('Warning! ', iterable, name_fit)
            
        N_opt=pd.Series(N_ref(ranges, fitted_bulk), index=N_enz.index.values[:len(ranges)], name=f'{name_fit} {iterable}')
        N_opt.replace(0, np.nan, inplace=True)
    
        return N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit
    
    @staticmethod            
    def get_sim_counts(iterable_df, iterable, quantiles, describe):

        if describe == 'mean':
        
            #Define what is N(i)_enzyme
            N_enz=iterable_df[iterable]
            N_enz.name=f'N$_{{(i)}}$enzyme {iterable}'
          
            #Define what is N(i)_enzyme error
            N_enz_err=iterable_df[f'{iterable}-St.Err.']
            N_enz_err.name=f'N$_{{(i)}}$enzyme {iterable} St.Err.'
            
            #Note: This is inverted so that dG_err calculation keeps the form.
            N_error_p, N_error_m=N_enz-N_enz_err, N_enz+N_enz_err 
            
        elif describe == 'quantile':
            
            N_error_m=pd.DataFrame()
            N_error_p=pd.DataFrame()
                            
            for quantile in quantiles:
                
                #Define what is N(i)_enzyme
                if quantile == 0.5:
                    
                    N_enz=iterable_df['Q0.5']     
                    N_enz.replace(0, np.nan, inplace=True)
                
                #Define what is N(i)_enzyme error
                elif quantile < 0.5:                        
                    N_error_m=pd.concat([N_error_m, iterable_df[f'Q{quantile}']], axis=1)
                    #N_error_m.name=f'N$_{{(i)}}$enzyme {iterable} Q{quantile}'
                    #print(N_error_m)
                    
                elif quantile > 0.5:
                    N_error_p=pd.concat([N_error_p, iterable_df[f'Q{quantile}']], axis=1)
                    #N_error_p.name=f'N$_{{(i)}}$enzyme {iterable} Q{quantile}'
                    #print(N_error_p)
                    #N_error_p=df[f'{iterable}-Q{quantiles[-1]}']
        #N_enz.replace(0, np.nan, inplace=True)
        #N_error_m.replace(0, np.nan, inplace=True)
        #N_error_p.replace(0, np.nan, inplace=True)
        
        if N_error_p.min() < 1e-10:
            N_error_p= N_error_p.where(N_error_p > 1e-10, np.nan)
        return N_enz, N_error_p, N_error_m

    
    def plot(self, 
             input_df=None,
             level=2, 
             subplots_=True):
        
        dfs= []        
        for k, v in self.discretized.items():
            if re.search('shellProfile', k):
                print(f'Discretization {k} found.')
                dfs.append((k,v, 'ShellProfile'))
                
            if re.search('combinatorial', k):
                print(print(f'Discretization {k} found.'))
                dfs.append((k,v, 'Combinatorial'))
                
        if not len(dfs):
            
            df=[('external', input_df)]
            print('Feature not found. Using input DataFrame.')
            
            
        
        
        for name_df in dfs:
            (name, df, kind) = name_df

            
            level_iterables=df.columns.levels #Exclude last, the values of states
            rows, columns=tools_plots.plot_layout(level_iterables[level])
            fig_, axes_it =plt.subplots(rows, columns, sharex=True, sharey=False, constrained_layout=True, figsize=(12,9))
            
            try:
                axes_it.flat
            except:
                axes_it=np.asarray(axes_it)
        
            for iterable, ax_it in zip(level_iterables[level], axes_it.flat): 
                
                df_it=df.loc[:,df.columns.get_level_values(f'l{level+1}') == iterable] #level +1 due to index starting at l1
                if kind == 'ShellProfile':
                    df_it.plot(kind='line', 
                               ax=ax_it, 
                               subplots=subplots_,  
                               title=f'{name} @{iterable}', 
                               figsize=(9,6), 
                               legend=False,
                               sort_columns=True,
                               linewidth=1,
                               loglog=True,
                               xlabel=f'{level_iterables[-1].to_list()[0]}',
                               ylabel='counts')
                elif kind == 'Combinatorial':
                    df_it.plot(x=self.data.index.name, y=np.arange(0,len(df_it.columns)), 
                               kind='scatter',
                               ax=ax_it,
                               subplots=subplots_, 
                               sharex=True, 
                               sharey=True, 
                               #layout=(5,5), # (int(len(df_it)/2), int(len(df_it)/2)),
                               title=f'{name} @{iterable}',
                               xlabel=f'Trajectory time ({self.data.index.name})',
                               ylabel='State Index',
                               figsize=(9,6))

                plt.savefig(f'{self.results}/discretized_{kind}_{name}.png', bbox_inches="tight", dpi=600)
        
        return plt.show()


    @classmethod
    def get_combinatorial_traj(cls):
        
        return cls()



    @staticmethod
    def data_loader(combinatorial_traj,
                    supra_project, 
                    projects, 
                    project_type,
                    state_boundaries,
                    state_labels,
                    state_name = 'ATESB',
                    water_mols = {'normal' : ['BeOH', 'BuOH'], 'inhibition' : ['BeOH', 'BuOH']}):
        
        
        
        def loader():
            supra_df = pd.DataFrame()
            
            combinatorial = {}
            if project_type == 'water':
                project = projects['H2O']
                
                supra_df = combinatorial_traj
                if isinstance(supra_df, tuple):
                    (supra_df, states) = supra_df
                print(np.unique(supra_df.dropna().values.flat).astype(int))
                for p_type, mols_w in water_mols.items():
                    if p_type == 'normal' :
                        for mol_w in mols_w:
                            its_w = supra_project[p_type][mol_w].parameter
                            for it_w in its_w:
                                df_mol = pd.read_csv(f'{project.results}/combinatorial_acylSer-His_{mol_w}_water_{it_w}_{state_name}.csv', index_col=0, header=[0,1,2,3,4,5])
                                supra_df = pd.concat([supra_df, df_mol], axis=1)
                                print(mol_w, it_w, np.unique(supra_df.dropna().values.flat).astype(int))
                    else:
                        for mol_w in mols_w:
                            if mol_w == 'BeOH':
                                mol2 = 'BeAc'
                            else:
                                mol2 = 'ViAc'
                            print(mol_w, mol2)
                            df_name = f'{project.results}/combinatorial_acylSer-His_{mol_w}-{mol2}_water_100mM_{mol2}_5mM_{state_name}.csv'
                            supra_df = pd.concat([supra_df, pd.read_csv(df_name, index_col=0, header=[0,1,2,3,4,5])], axis=1)
                            print(mol_w, mol2, it_w, np.unique(supra_df.dropna().values.flat).astype(int))
                            
                combinatorial['H2O'] = supra_df
                
            elif project_type == 'normal':
                
                supra_df = pd.DataFrame()
                for mol, project in projects.items():  
                    
        
                    df_mol = combinatorial_traj
                    supra_df = pd.concat([supra_df, df_mol], axis=1)
                    combinatorial[mol] = df_mol
            
            else:
                
                supra_df = pd.DataFrame()
                for mol, project in projects.items():                                                                                    
                    df_mol = pd.read_csv(f'{project.results}/combinatorial_acylSer-His_{mol}_100mM_{state_name}.csv', index_col=0, header=[0,1,2,3,4,5])
                    if mol == 'BeOH':
                        mol2 = 'BeAc'
                    else:
                        mol2 = 'ViAc'
                    df_mol_in_mol2 = pd.read_csv(f'{project.results}/combinatorial_acylSer-His_{mol}_100mM_{mol2}_5mM_{state_name}.csv', index_col=0, header=[0,1,2,3,4,5])
                    df_mol2 = pd.read_csv(f'{project.results}/combinatorial_acylSer-His_{mol2}_100mM_{mol2}_5mM_{state_name}.csv', index_col=0, header=[0,1,2,3,4,5])
                    df_mol_ternary = pd.concat([df_mol, df_mol_in_mol2, df_mol2], axis=1) 
                    supra_df = pd.concat([supra_df, df_mol_ternary], axis=1) 
                    combinatorial[mol] = df_mol_ternary
                    
                    
            return combinatorial, supra_df
        


            
            
        print(project_type)
    
        
    
        combinatorial, supra_df = loader()
    
        state_indexes = np.unique(supra_df.dropna().values.flat).astype(int) 
        state_labels = tools.Functions.sampledStateLabels(state_boundaries, state_labels, sampled_states=state_indexes) 
        states = {j : i for i, j in zip(state_labels, state_indexes)}
    
        original_values, replaced_values, labels = [], [], []
        for idx,(index, label) in enumerate(states.items()):
            original_values.append(index)
            replaced_values.append(idx)
            labels.append(label)
        remap_states = {r : l for r, l in zip(replaced_values, labels)} 
    
        sampled_states = (states, remap_states)

        
        
        return combinatorial, supra_df, sampled_states
        
        
    @staticmethod
    def run_metric(inputs, corr_mode='single', method='correlation'):
        
        def run_metric_parallel():
            
            fixed = (ref, ref2, method)
            values=state_combs # iterable_df[value] values_]
            
            _metric = tools.Tasks.parallel_task(Discretize.metric_calculation, values, fixed, n_cores=n_cores, run_silent=True)

            return _metric
        
        (df_mol, df_mol2, mol1, mol2, _states, state_combs, n_cores) = inputs

        states1, states2 = _states
        traj_mol = pd.DataFrame()
        metrics = [] #np.empty((len(states1)*len(states2))*len(df_mol.columns))
        for idx, col in enumerate(df_mol.columns):
            ref = df_mol[col]
            ref2 = df_mol2.iloc[:, idx]
            traj_mol = pd.concat([traj_mol, pd.DataFrame({ mol1 : ref.values, mol2 : ref2.values})], axis=0)
            
            if corr_mode == 'all_median':

                print(ref.name, ref2.name, end='\r')
                metrics.append(run_metric_parallel())
        if corr_mode == 'all_median':
            out_corr = np.nanmedian(np.asarray(metrics).reshape((len(states1),len(states2), len(df_mol.columns))), axis=2)

            
            metric_table_mol = out_corr #correlation_tables[mol1] = out_corr
        
            return np.asarray(metric_table_mol).reshape(len(states1), len(states2)), traj_mol
        
        elif corr_mode == 'single':         
            ref = traj_mol.iloc[:,0]
            ref2 = traj_mol.iloc[:,1]

            metric_table_mol  = run_metric_parallel() #correlation_tables[mol1]
            
            #X = mol, its
            #target = water
            #mi = 
            
            #f_test, _ = f_regression(X, y)
            #f_test /= np.max(f_test)
            #cross_corr_trajectory[idx] = i[1]
            #out_cross_corr = cross_corr_trajectory.reshape((9,9,len(ref)))
            return np.asarray(metric_table_mol).reshape(len(states1), len(states2)), traj_mol
        
        elif corr_mode == 'all':
            X = traj_mol.iloc[:,0]
            y = traj_mol.iloc[:,1]
            
            return mutual_info_classif(X.values.reshape(-1,1), y, discrete_features=True), traj_mol


    @staticmethod
    def metric_calculation(series_value, specs=()):

        (ref, ref2, method)=specs
        (state_combs)=series_value
        
        s = state_combs[0]
        s2 = state_combs[1]
        ref_s = ref == s
        ref2_s2 = ref2 == s2
    
    
        if method == 'correlation':
        #cross_corr = np.correlate(ref_s.values, ref2_s2.values, mode='valid')
        #conv = np.convolve(ref_s.values, ref2_s2.values, mode='valid')
            out = ref_s.corr(ref2_s2)
        
        elif method == 'mutualInformation':
            X = ref_s
            y = ref2_s2

            #mi = mutual_info_classif(X.values.reshape(-1,1), y, discrete_features=False)
            out = mutual_info_classif(X.values.reshape(-1,1), y, discrete_features=True)

        else:
            out = None

        #pbar.update(1)
        return out #corr_coeff


    @staticmethod
    def get_metricsInhibition(combinatorial, 
                                   mol_ternary, 
                                   sampled_states, 
                                   n_cores=10, 
                                   corr_mode='single', 
                                   remap_traj = False,
                                   method='mutualInformation'):
        
        
        
        states_mol = sampled_states['inhibition'][0]
        states_list = list(states_mol.keys()) #list(states.keys())
        #state_combs = itertools.product(states_list, states_list)
        #_states = (states_list, states_list)
        
        correlation_tables = {}
        trajectories = {}
        

        for mol1, mol2 in mol_ternary.items():
            print(mol1)
           
            df_mol = combinatorial['inhibition'][mol1].loc[:, ('acyl_octamer', mol1, f'100mM_{mol2}')] 
            df_mol2 = combinatorial['inhibition'][mol1].loc[:, ('acyl_octamer', mol2, '5mM')] #f'100mM_{mol2}')]  
            
            states_mol_list = states_list# list(np.unique(df_mol2.values.flat))
            #states_mol_list = {s : states_mol[s] for s in _states_mol_list}  
            _states = (states_list, states_mol_list)
            state_combs = itertools.product(states_list, states_mol_list) #this needs to be here since its not reinitializing
            _inputs = (df_mol, df_mol2, mol1, mol2, _states, state_combs, n_cores)
            
            table, traj = Discretize.run_metric(_inputs, corr_mode, method=method)
            if remap_traj:
                traj = tools.Functions.remap_trajectories_states(traj, states_mol)[0]

            #table = table.reshape(len(states_mol), len(states_mol))
            correlation_tables[mol1], trajectories[mol1] = table, traj
            
            
            #TODO:  single corr from traj_mol or median of all columns beware of vmax vmin changes
            
            
        return correlation_tables, trajectories
        
        
    
    
    @staticmethod
    def double_combinatorial_calculation(series_value, specs=()):
        
        (df_mol, df_mol2, encodings) = specs
        (idx, col)=series_value
        
        def state_evaluator(x):
            x= tuple(x)
            if x in encodings: #.values():
                
                if np.isnan(encodings.index(x)):
                    print(' is nan', x)
                
                
                return encodings.index(x)
        
        ref = df_mol[col]
        ref2 = df_mol2.iloc[:, idx]
        
        
        
        
        discretized_df=pd.concat([ref, ref2], axis=1).apply(state_evaluator, raw=True, axis=1)

        return discretized_df
    
    @staticmethod
    def run_double_combinatorial(inputs):
        
        
        
        (df_mol, df_mol2, mol1, mol2, _states, state_combs, n_cores) = inputs

        states1, states2 = _states
        traj_mol = pd.DataFrame() #TODO: make new columns
        metrics = [] #np.empty((len(states1)*len(states2))*len(df_mol.columns))
        
        fixed = (df_mol, df_mol2, state_combs)
        values=[(i, j) for i,j in enumerate(df_mol.columns)] # iterable_df[value] values_]
        
        traj_mols = tools.Tasks.parallel_task(Discretize.double_combinatorial_calculation, values, fixed, n_cores=n_cores, run_silent=True)
        
        for t in traj_mols:
            traj_mol = pd.concat([traj_mol, t], axis=1)
        
        #print(traj_mol)
        return traj_mol
        


    @staticmethod
    def get_double_combinatorial(combinatorial, 
                                 mol_water, 
                                 sampled_states, 
                                 n_cores=10,  
                                 remap_traj=True,
                                 results=None,
                                 overwrite=False):
        
        
        base_water_df = combinatorial['water']['H2O']

        base_mol = sampled_states['normal'][0]
        base_water = sampled_states['water'][0]
        states_mol = list(base_mol.keys())
        states_water = list(base_water.keys())
        
        labels_mol = list(base_mol.values())
        labels_water = list(base_water.values())
        _states = (states_water, states_mol)
        
        
        state_combs = list(itertools.product(states_water, states_mol))
        label_combs = list(itertools.product(labels_water, labels_mol))

        
        
        water_l = []
        for idx, l in enumerate(label_combs):
            
            if l[0] == 'ATESB':
                
                label_combs[idx] = f'w_{l[1]}'
                water_l.append(idx)
            else:
                label_combs[idx] = l[1]
        
        water_labels = [label_combs[l] for l in water_l]
        comb_states = {k : v for k,v in enumerate(label_combs)}
        comb_water_states = {k : v for k,v in enumerate(water_labels)}
        comb_mol_states = {k : v for k,v in enumerate(label_combs) if v not in water_labels}
        
        
        print(comb_states, comb_water_states, comb_mol_states)
        
        trajectories = collections.defaultdict(dict)
        supra_traj = pd.DataFrame()
        sampled_comb_states = []
        fig=plt.figure(figsize=(9,6), constrained_layout=True)
        outer_grid = gridspec.GridSpec(1,len(mol_water))
        
        
        for idx_mol, (mol1, iterables_mol_w) in enumerate(mol_water.items()):
            print(idx_mol, mol1, iterables_mol_w)
            
            mol_traj = pd.DataFrame()
            (iterables, mol2) = iterables_mol_w
            
            #grid_mol = gridspec.GridSpecFromSubplotSpec(len(iterables),1 , subplot_spec = outer_grid[0,idx_mol])
            for idx_it, it in enumerate(iterables):
                print(idx_it, it)
                if it == f'100mM_{mol2}_5mM':
                    df_mol = combinatorial['inhibition'][mol1].loc[:, ('acyl_octamer', mol2, '5mM')] #f'100mM_{mol2}'
                    #TODO: make here entry for mol2
                else:
                    df_mol = combinatorial['normal'][mol1].loc[:, ('acyl_octamer', mol1, it)] 

                df_water =  base_water_df.loc[:, ('acyl_octamer', mol1, it)] #f'100mM_{mol2}')] 
                df_water = df_water.iloc[:-1, :]
                

                _inputs = (df_water, df_mol, 'water', mol1, _states, state_combs, n_cores)
                
                file_name = f'{results}/temp/traj_{mol1}_{it}.csv'
                if not os.path.exists(file_name) or overwrite:
                    traj = Discretize.run_double_combinatorial(_inputs) 
                    traj.to_csv(file_name)
                else:
                    traj = pd.read_csv(file_name, index_col=0, header=[0])

                trajectories[mol1][it] = traj

                mol_traj = pd.concat([mol_traj, traj], axis=1)
                
            mol_traj.columns = base_water_df.iloc[:, base_water_df.columns.get_level_values('l2') == mol1].columns
            
            
            supra_traj = pd.concat([supra_traj, mol_traj], axis=1)

        sampled_comb_states = {s : label_combs[idx] for idx, s in enumerate(list(set(supra_traj.values.flat)))}

        sampling_fractions = collections.defaultdict(dict)
        for idx_mol, (mol, it_trajs) in enumerate(trajectories.items()):
            print(mol)
            grid_mol = gridspec.GridSpecFromSubplotSpec(len(it_trajs),1 , subplot_spec = outer_grid[0,idx_mol])
            for idx_it, (it, traj) in enumerate(it_trajs.items()):
                print(it)
                if idx_it == 0:
                    ax = plt.subplot(grid_mol[idx_it,0])
                else:
                    ax = plt.subplot(grid_mol[idx_it,0], sharex=ax, sharey=ax)
                if remap_traj: 
                    traj, sampled_comb_states = tools.Functions.remap_trajectories_states(traj, sampled_comb_states)

                state_fraction =  np.asarray([np.count_nonzero(traj == i) / traj.size for i in comb_states]) 
                state_fraction_w = np.asarray([np.count_nonzero(traj == i) / traj.size for i, label in comb_states.items() if label in water_labels]).T 
                state_fraction_mol = np.asarray([np.count_nonzero(traj == i) / traj.size for i, label in comb_states.items() if label not in water_labels]).T 
               
                water_vs_mol = state_fraction_w + state_fraction_mol

                df = pd.DataFrame(data=np.column_stack((state_fraction_w / water_vs_mol, state_fraction_mol / water_vs_mol)), index=comb_mol_states.values(), columns=['ATESB', 'TESB'])
                sampling_fractions[mol][it] = df
            
                df.plot.bar(stacked=True, ax=ax)
                ax.set_xticks(list(comb_mol_states.keys()))
                ax.set_xticklabels(list(comb_mol_states.values()), rotation=90)
                ax.set_title(it)
            
        return (supra_traj, trajectories, sampling_fractions)
        


    

    @staticmethod
    def get_metricsNormal(combinatorial : dict, 
                         mol_water, 
                         sampled_states, 
                         n_cores=10, 
                         corr_mode='single', 
                         remap_traj=False,
                         method='correlation'):
        

        
        base = combinatorial['normal']
        
        mols = list(base.keys())

        states_mol = sampled_states['normal'][0]
        states_mol_list = list(states_mol.keys())
       

        _states = (states_mol_list, states_mol_list)

        trajectories = collections.defaultdict(dict)
        correlation_tables = collections.defaultdict(dict)
        
        
        def correlation_map(x):
            
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0
        
        
        ref_map = {'A-B' : (0,1), 'C-H' : (2, 7), 'E-F' : (4,5), 'D-G' : (3,6)}
        
        from scipy import stats
        
        for idx_mol, mol in enumerate(mols):

            df_mol = base[mol].iloc[:, base[mol].columns.get_level_values('l2') == mol]
            iterables = df_mol.columns.get_level_values('l3').unique()
            references = df_mol.columns.get_level_values('l3').unique()

            print(idx_mol, mol)

            for it in iterables:

                df_it = df_mol.iloc[:, df_mol.columns.get_level_values('l3') == it]
                references = df_it.columns.get_level_values('l5').unique()
                
                it_table = []
                
                for pair, index in ref_map.items():
                    
                    idx, idx2 = index[0], index[1]
                    ref, ref2 = references[idx], references[idx2]
                    df_ref = df_it.iloc[:, df_it.columns.get_level_values('l5') == ref]
                    df_ref2 = df_it.iloc[:, df_it.columns.get_level_values('l5') == ref2]
                    state_combs = itertools.product(states_mol_list, states_mol_list)
                    _inputs = (df_ref, df_ref2, 'source', 'target', _states, state_combs, n_cores)
                    table, _ = Discretize.run_metric(_inputs, corr_mode, method=method)
                    it_table.append(table)
                _it_table = np.asarray(it_table) 
                    
                #TODOTODOTODOTODOTO means only for MI, not for CORR COEFF!!!!!!!
                if method == 'mutualInformation':
                    out = _it_table.sum(axis=0)
                else:
                    mask = np.vectorize(correlation_map)(_it_table)
                    out = stats.mode(mask, axis=0).mode[0]
                print(np.shape(out))
                correlation_tables[mol][it], trajectories[mol][it] = out, pd.DataFrame()

        
        
        return correlation_tables, trajectories
        
        
        
        pass
    
        


    @staticmethod
    def get_metricsWater(combinatorial, 
                         mol_water, 
                         sampled_states, 
                         n_cores=10, 
                         corr_mode='single', 
                         remap_traj=False,
                         method='correlation'):
        

        
        base_water_df = combinatorial['water']['H2O']

        states_mol = sampled_states['normal'][0]
        states_water = sampled_states['water'][0]
        states_mol_list = list(states_mol.keys())
        states_water_list = list(states_water.keys())
       

        _states = (states_water_list, states_mol_list)

        trajectories = collections.defaultdict(dict)
        correlation_tables = collections.defaultdict(dict)
        for idx_mol, (mol1, iterables_mol_w) in enumerate(mol_water.items()):
            print(idx_mol, mol1, iterables_mol_w)
            
            (iterables, mol2) = iterables_mol_w
            for it in iterables:
                if it == f'100mM_{mol2}_5mM':
                    df_mol = combinatorial['inhibition'][mol1].loc[:, ('acyl_octamer', mol2, '5mM')] #f'100mM_{mol2}' 
                else:
                    df_mol = combinatorial['normal'][mol1].loc[:, ('acyl_octamer', mol1, it)] 

                df_water =  base_water_df.loc[:, ('acyl_octamer', mol1, it)] #f'100mM_{mol2}')] 
                df_water = df_water.iloc[:-1, :]
                
                state_combs = itertools.product(states_water_list, states_mol_list)
                _inputs = (df_water, df_mol, 'water', mol1, _states, state_combs, n_cores)
                table, traj = Discretize.run_metric(_inputs, corr_mode, method=method)
                if remap_traj:
                    traj_w = tools.Functions.remap_trajectories_states(traj.loc[:,'water'], states_water)[0]
                    traj_mol = tools.Functions.remap_trajectories_states(traj.loc[:,mol1], states_mol)[0]

                    traj = pd.concat([traj_w, traj_mol], axis=1)

                correlation_tables[mol1][it], trajectories[mol1][it] = table, traj
        
        
        return correlation_tables, trajectories






    # =============================================================================
#     def minValue(self, state_shell):
#         """Discretization is made directly on raw_data using the numpy digitize function.
#         Minimum value per frame (axis=2) is found using the numpy min function. 
#         Discretization controled by state_shell"""
#         
#         msm_min_f={}
#         for feature, parameters in self.features.items():
#             raw_data=[] #data_dict may contain one or more .npy objects
#             for parameter, data in parameters.items():
#                 if os.path.exists(data):
#                     i_arr=np.load(data)
#                     raw_data.append(i_arr)
#                 else:
#                     print(f'\tWarning: file {data} not found.')
#             rep, frames, ref, sel=np.shape(raw_data)
#             raw_reshape=np.asarray(raw_data).reshape(rep*frames, ref*sel)
#             discretize=np.min(np.digitize(raw_reshape, state_shell), axis=1)
# 
#             disc_path=f'{self.results_folder}/discretized-minimum-{self.name}-{parameter}.npy'
#             np.save(disc_path, discretize)
#             msm_min_f[parameter]={'discretized':[disc_path]}
#         return msm_min_f
#     
#     def single(self, state_shell):
#         """Discretization is made directly on raw_data using the numpy digitize function.
#         For each ligand, the digitized value is obtained. Discretization controled by state_shell.
#         WARNING: size of output array is N_frames*N_ligands*N_replicates. Might crash for high N_frames or N_ligands"""
#         
#         msm_single_f={}
#         for parameter, data_dict in self.features.items():
#             raw_data=[] #data_dict may contain one or more .npy objects
#             for i in data_dict:
#                 if os.path.exists(i):
#                     i_arr=np.load(i)
#                     raw_data.append(i_arr)
#                 else:
#                     print(f'\tWarning: file {i} not found.')
#             rep, frames, ref, sel=np.shape(raw_data)
#             raw_reshape=np.asarray(raw_data).reshape(rep*frames*ref*sel)
#             
#             discretize=np.digitize(raw_reshape, state_shell)
#             disc_path='{}/discretized-single-{}-{}.npy'.format(self.results, self.name, parameter)
#             np.save(disc_path, discretize)
#             msm_single_f[parameter]={'discretized':[disc_path]}
#         return msm_single_f
# =============================================================================    
    

    

    
    
    
    