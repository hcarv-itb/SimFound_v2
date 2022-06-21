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
            handles_, labels_ = ax_dG_rep.get_legend_handles_labels()
            fig_rep_dG.legend(handles_, labels_, loc='center right') #, ncol=4) #, bbox_to_anchor=Discretize.def_locs_multi)
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
    def run_correlation(inputs, corr_mode='single'):
        
        #TODO: BuOH is not being calculated. 
        def run_correlation_parallel():
            
            fixed = (ref, ref2)
            values=state_combs # iterable_df[value] values_]
            
            _correlation = tools.Tasks.parallel_task(Discretize.correlation_calculation, values, fixed, n_cores=n_cores, run_silent=True)

            return _correlation
        
        (df_mol, df_mol2, mol1, mol2, _states, state_combs, n_cores) = inputs
        (states1, states2) = _states
        traj_mol = pd.DataFrame()
        correlations = [] #np.empty((len(states1)*len(states2))*len(df_mol.columns))
        for idx, col in enumerate(df_mol.columns):
            ref = df_mol[col]
            ref2 = df_mol2.iloc[:, idx]
            traj_mol = pd.concat([traj_mol, pd.DataFrame({ mol1 : ref.values, mol2 : ref2.values})], axis=0)
            
            if corr_mode == 'all_median':

                print(ref.name, ref2.name, end='\r')
                correlations.append(run_correlation_parallel())
        if corr_mode == 'all_median':
            out_corr = np.nanmedian(np.asarray(correlations).reshape((len(states1),len(states2), len(df_mol.columns))), axis=2)

            
            correlation_table_mol = out_corr #correlation_tables[mol1] = out_corr
        
        elif corr_mode == 'single':         
            ref = traj_mol.iloc[:,0]
            ref2 = traj_mol.iloc[:,1]

            correlation_table_mol  = run_correlation_parallel() #correlation_tables[mol1]
            
            #X = mol, its
            #target = water
            #mi = 
            
            #f_test, _ = f_regression(X, y)
            #f_test /= np.max(f_test)
            #cross_corr_trajectory[idx] = i[1]
            #out_cross_corr = cross_corr_trajectory.reshape((9,9,len(ref)))

        #trajectories[mol1] = traj_mol
        
        return np.asarray(correlation_table_mol), traj_mol

    @staticmethod
    def correlation_calculation(series_value, specs=()):

        (ref, ref2)=specs
        (state_combs)=series_value
        
        s = state_combs[0]
        s2 = state_combs[1]
        ref_s = ref == s
        ref2_s2 = ref2 == s2
    
        #cross_corr = np.correlate(ref_s.values, ref2_s2.values, mode='valid')
        #conv = np.convolve(ref_s.values, ref2_s2.values, mode='valid')
        corr_coeff = ref_s.corr(ref2_s2)
        
        X = ref_s
        y = ref2_s2
        
        
        
        
        #mi = mutual_info_classif(X.values.reshape(-1,1), y, discrete_features=False)
        mi_disc = mutual_info_classif(X.values.reshape(-1,1), y, discrete_features=True)
        #print(s, s2, mi_disc)
        #print(corr_coeff)
        #pbar.update(1)
        return mi_disc #corr_coeff


    @staticmethod
    def get_correlationsInhibition(combinatorial, mol_ternary, sampled_states, n_cores=10, corr_mode='single', remap_traj = False):
        
        import itertools
        
        states_mol = sampled_states['inhibition'][0]
        states_list = list(states_mol.keys()) #list(states.keys())
        #state_combs = itertools.product(states_list, states_list)

        _states = (states_list, states_list)
        
        correlation_tables = {}
        trajectories = {}
        

        for mol1, mol2 in mol_ternary.items():
            print(mol1)
            state_combs = itertools.product(states_list, states_list) #this needs to be here since its not reinitializing
            df_mol = combinatorial['inhibition'][mol1].loc[:, ('acyl_octamer', mol1, f'100mM_{mol2}')] 
            df_mol2 = combinatorial['inhibition'][mol1].loc[:, ('acyl_octamer', mol2, '5mM')] #f'100mM_{mol2}')]  
            
            _inputs = (df_mol, df_mol2, mol1, mol2, _states, state_combs, n_cores)
            
            table, traj = Discretize.run_correlation(_inputs, corr_mode)
            if remap_traj:
                traj = tools.Functions.remap_trajectories_states(traj, states_mol)[0]

            table = table.reshape(len(states_mol), len(states_mol))
            correlation_tables[mol1], trajectories[mol1] = table, traj
            
            
            #TODO:  single corr from traj_mol or median of all columns beware of vmax vmin changes
            
            
        return correlation_tables, trajectories
        
        
    

    
    @staticmethod
    def get_correlationsWater(combinatorial, mol_water, sampled_states, n_cores=10, corr_mode='single', remap_traj=False):
        
        from multiprocessing import Process, Pool
        import itertools
        import collections
        
        base_water_df = combinatorial['water']['H2O']

        states_mol = sampled_states['normal'][0]
        states_water = sampled_states['water'][0]
        states_mol_list = list(states_mol.keys())
        states_water_list = list(states_water.keys())
        state_combs = list(itertools.product(states_mol.keys(), states_water.keys()))

        _states = (states_water_list, states_mol_list)

        trajectories = collections.defaultdict(dict)
        correlation_tables = collections.defaultdict(dict)
        for idx_mol, (mol1, iterables_mol_w) in enumerate(mol_water.items()):
            print(idx_mol, mol1, iterables_mol_w)
            
            (iterables, mol2) = iterables_mol_w
            for it in iterables:
                if it == f'100mM_{mol2}_5mM':
                    df_mol = combinatorial['inhibition'][mol1].loc[:, ('acyl_octamer', mol1, f'100mM_{mol2}')] 
                    #TODO: make here entry for mol2
                else:
                    df_mol = combinatorial['normal'][mol1].loc[:, ('acyl_octamer', mol1, it)] 

                df_water =  base_water_df.loc[:, ('acyl_octamer', mol1, it)] #f'100mM_{mol2}')] 
                df_water = df_water.iloc[:-1, :]
                
                _inputs = (df_water, df_mol, 'water', mol1, _states, state_combs, n_cores)
                table, traj = Discretize.run_correlation(_inputs, corr_mode)
                table = table.reshape(len(states_mol), len(states_water))
                print(np.sum(table))
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
    
#ad hoc dG comparison
# =============================================================================
#df_benzyl = pd.read_csv('/media/dataHog/hca/msAcT-acylOctamer/results/acyl_octamer-BeOH-100mM_BeAc-5mM/binding_profile_acylSer-His_BeAc_mean_b0_e-1_s1.csv', index_col=0)
#df_benzyl_normal = pd.read_csv('/media/dataHog/hca/msAcT-acylOctamer/results/binding_profile_acylSer-His_BeOH_mean_b0_e-1_s1.csv', index_col=0)

#df_vinyl = pd.read_csv('/media/dataHog/hca/msAcT-acylOctamer/results/acyl_octamer-BuOH-100mM_ViAc-5mM/binding_profile_acylSer-His_ViAc_mean_b0_e-1_s1.csv', index_col=0)
#df_vinyl_normal = pd.read_csv('/media/dataHog/hca/msAcT-acylOctamer/results/binding_profile_acylSer-His_BuOH_mean_b0_e-1_s1.csv', index_col=0)

#df_inib_benzyl = df_benzyl_normal[df_benzyl_normal.columns[6:9]]
#df_inib_vinyl = df_vinyl_normal[df_vinyl_normal.columns[6:9]]
# import matplotlib.pyplot as plt
# import numpy as np
# colors = ['grey', 'black'] #, 'black']
# labels = {'BeAc' : ['100mM BeOH', '100mM BeOH + 5mM BeAc'], 'ViAc' : ['100mM BuOH', '100mM BuOH + 5mM ViAc']}#, '5mM BeAc']
# #labels_vinyl = ['100mM BuOH', '100mM BuOH + 5mM ViAc'] #, '5mM ViAc
# 
# 
# mols = ['ViAc', 'BeAc']
# inibs = {'BeAc' : df_benzyl,
#       'ViAc' : df_vinyl}
# dfs = {'BeAc' : [df_benzyl_normal, df_inib_benzyl],
#       'ViAc' : [df_vinyl_normal, df_inib_vinyl]}
# plot, axes = plt.subplots(1,2, sharey=True, figsize=(10,6))
# symbols = ['--', '-']
# for mol, ax in zip(mols, axes.flat):
#     df_mol = dfs[mol] #= (f'df_{mol}', f'df_{mol}_normal') #, f'df_inib_{mol}')
#     labels_mol = labels[mol]
#     lns= []
#     for df, label, color, symbol in zip(df_mol, labels_mol, colors, symbols):
#         ranges = df.index.values
#         dg = df[df.columns[0]]
#         dg_m = df[df.columns[1]]
#         dg_p = df[df.columns[2]]
#         ln = ax.plot(ranges, dg, symbol, color=color, label=label)
#         ax.fill_between(ranges, dg_m, dg_p, alpha=0.5, color=color)
#         lns.append(ln)
#     ax.set_xlim(1, 95)
#     ax.set_ylim(-4,8)
#     ax.set_xscale('log')
#     ax.axhline(y=0, ls='--', color='black')    
#     #ax.legend()
#     ax2 = ax.twiny()
#     inib_df = inibs[mol]
#     ranges2 = inib_df.index.values
#     dg2 = inib_df[inib_df.columns[0]]
#     dg2_m = inib_df[inib_df.columns[1]]
#     dg2_p = inib_df[inib_df.columns[2]]
#     ln2 = ax2.plot(ranges2, dg2,'.', color='green', label=f'5mM {mol}')
#     ax2.fill_between(ranges2, dg2_m, dg2_p, alpha=0.2, color='green')
#     ax2.set_xlim(1, 95)
#     ax2.set_ylim(-4,7)
#     ax2.set_xscale('log')
#     #ax2.legend()
#     ax2.tick_params(axis='x', labelcolor='green')
#     lns.append(ln2)
#     labs = [p[0].get_label() for p in lns]
#         
#     ax.legend([ln[0] for ln in lns], labs, loc='lower right')
# plot.text(0.01, 0.5, r'$\Delta$G ($\it{k}$$_B$T)', rotation='vertical', fontsize=12) 
# plot.text(0.5, 0.01, r'$\itd$$_{NAC}$ ($\AA$)', fontsize=12)
# plot.text(0.5, 0.99, r'$\itd$$_5$ ($\AA$)', color='green', fontsize=12)
# plot.tight_layout()
# #plt.legend()
# plt.show()
# plot.savefig('/media/dataHog/hca/msAcT-acylOctamer/results/BindingProfile_Inibitors_all.png', dpi=600, bbox_inches='tight')
# =============================================================================
    
#ad hoc water comparison
# =============================================================================
# import pandas as pd
# 
# df_water = pd.read_csv(f'{project.results}/acyl_octamer-H2O-55.56M/binding_profile_acylSer-His_H2O_water_mean_b0_e-1_s1.csv', index_col=0)
# df_beoh_water = pd.read_csv(f'{project.results}/binding_profile_acylSer-His_BeOH_water_mean_b0_e-1_s1.csv', index_col=0)
# df_buoh_water = pd.read_csv(f'{project.results}/binding_profile_acylSer-His_BuOH_water_mean_b0_e-1_s1.csv', index_col=0)
# df_water_combinatorial_BeOH = pd.read_csv(f'{project.results}/acyl_octamer-H2O-55.56M/combinatorial_acylSer-His_H2O_water_55.56M_AT1T2T3ESB.csv', index_col=0, header=[0,1,2,3,4,5])
# df_water_combinatorial_BuOH = pd.read_csv(f'{project.results}/acyl_octamer-H2O-55.56M/combinatorial_acylSer-His_H2O_water_asBuOH_55.56M_AT1T2T3ESB.csv', index_col=0, header=[0,1,2,3,4,5])
# 
# iterables = { 'BeOH' : [['10mM', '20mM', '40mM', '100mM', '300mM'], df_beoh_water, df_water_combinatorial_BeOH], # ('o')], 
#             'BuOH' : [['11mM', '22mM', '100mM', '500mM'], df_buoh_water, df_water_combinatorial_BuOH],
#             'water' : [['55.56M'], df_water]} #, ('s')]}
# 
# supra_df = pd.DataFrame()
# for name, specs in iterables.items():
#     its = specs[0]
#     df_all = pd.DataFrame()
#     
#     if name != 'water':
#         df_all = pd.concat([df_all, specs[2]], axis=1)
#         for it in its:
#             df_it = pd.read_csv(f'{project.results}/combinatorial_acylSer-His_{name}_water_{it}_AT1T2T3ESB.csv', index_col=0, header=[0,1,2,3,4,5])
#             df_all = pd.concat([df_all, df_it], axis=1)
#         specs.append(df_all)
# 
#         supra_df = pd.concat([supra_df, df_all], axis=1)    
# state_indexes = np.unique(supra_df.dropna().values.flat).astype(int)
# state_labels = tools.Functions.sampledStateLabels([4,6,10,12,20,80], ['A', 'T1', 'T2', 'T3', 'E', 'S', 'B'], sampled_states=state_indexes)
# states = {j : i for i, j in zip(state_labels, state_indexes)}
# 
# original_values, replaced_values, labels = [], [], []
# for idx,(index, label) in enumerate(states.items()):
#     original_values.append(index)
#     replaced_values.append(idx)
#     labels.append(label)
# remap_states = {r : l for r, l in zip(replaced_values, labels)} 
# print(remap_states)
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pl
# import numpy as np
# import matplotlib.gridspec as gridspec
# fig=plt.figure(figsize=(10,10))
# gs = gridspec.GridSpec(nrows=3, ncols=2)
# plot_values = fig.add_subplot(gs[0, 0])
# loc1, loc2 =0, 250
# 
# scalars = {'BeOH': [10, 20, 40, 100, 300], 'BuOH' : [11, 22, 100, 500]}
# markers = {'BeOH': 's', 'BuOH' : 'o'}
# def plot_dg(dg):
#     
#     ranges = dg.iloc[loc1:loc2].index.values
#     ax.plot(ranges, dg.iloc[loc1:loc2], '.', marker=marker, color=color, label=label)
#     ax.fill_between(ranges, dg_m.iloc[loc1:loc2], dg_p.iloc[loc1:loc2], alpha=0.5, color=color)
#     ax.set_ylim(-1,14)
#     ax.set_xlabel(r'$\itd$$_{NAC}$ ($\AA$)')
#     ax.set_ylabel(r'$\Delta$G ($\it{k}$$_B$T)')
#     ax.set_xscale('log') 
#     ax.axhline(y=0, ls='--', color='darkgray')
#     ax.legend(ncol=2)
# def plot_dg_diff(dg):
#     ranges = dg.iloc[loc1:loc2].index.values
#     ax.plot(ranges, dg.iloc[loc1:loc2], '.', marker=marker, color=color, label=label_diff)
#     ax.set_xlabel(r'$\itd$$_{NAC}$ ($\AA$)')
#     ax.set_ylabel(r'$\Delta$$\Delta$G ($\it{k}$$_B$T)')
#     ax.set_xscale('log') 
#     ax.axhline(y=0, ls='--', color='black')
#     #ax.set_title(name, y=1.0, pad=-14)
#     ax.legend(title=name)
# counts_it_mol = {}
# for name, specs in iterables.items():
#     print(name)
#     its = specs[0]
#     df_profile = specs[1]
#     if name == 'water':
#         dg_water = df_profile['$\Delta$G 55.56M']
#         dg_m = df_profile['0']
#         dg_p = df_profile['1']
#         label = name
#         color = 'black'
#         marker = '^'
#         ax = plot_values
#         plot_dg(dg_water)
#     else:
#         remaped_df = specs[3].replace(to_replace=original_values, value=replaced_values)
#         counts_it = np.empty([len(its), len(remap_states)])
#         ref_df = remaped_df.loc[:, remaped_df.columns.get_level_values('l3') == '55.56M']
#         count_ref = np.asarray([np.count_nonzero(ref_df.values.flat == i) for i in remap_states])
#         marker = markers[name]
#         colors = pl.cm.Greens(np.linspace(0.2,1,len(its))) #(scalars[name]/np.max(scalars[name])) #
#         ax = plot_dff
#         if name == 'BeOH':
#             plot_diff = fig.add_subplot(gs[1, 0], sharex=plot_values)
#             plot_combinatorial = fig.add_subplot(gs[2, 0])
#         else:
#             plot_diff = fig.add_subplot(gs[1, 1], sharex=plot_diff, sharey=plot_diff)
#             plot_combinatorial = fig.add_subplot(gs[2, 1], sharey=plot_combinatorial)
#         for idx, it in enumerate(its): 
#             
#             dg = df_profile[f'$\Delta$G {it}']
#             if idx == 0:
#                 dg_m = df_profile['0']
#                 dg_p = df_profile['1']
#             else:
#                 dg_m = df_profile[f'0.{idx}']
#                 dg_p = df_profile[f'1.{idx}']
#             label = f'{name} {it}'
#             color= colors[idx]
#             ax = plot_values
#             plot_dg(dg)
#             
#             dg_diff = dg - dg_water
#             label_diff = it
#             ax = plot_diff
#             plot_dg_diff(dg_diff)
#             
#             df = remaped_df.loc[:, remaped_df.columns.get_level_values('l3') == it]
#             count = np.asarray([np.count_nonzero(df.values.flat == i) for i in remap_states])
#             counts_it[idx] = count/ count_ref
# 
#         colors_comb = pl.cm.turbo(np.linspace(0,1,len(remap_states)))
#         for idx_plot, it_s in enumerate(counts_it.T):
#             plot_combinatorial.plot(scalars[name], it_s, '-', marker='o', color=colors_comb[idx_plot], label=remap_states[idx_plot])
#         #plot_combinatorial.set_title(name)
#         plot_combinatorial.set_yscale('log')
#         plot_combinatorial.set_xlabel(f'[{name}] (mM)')
#         plot_combinatorial.set_ylabel(f'{name}/{name}$_0$')
# plot_combinatorial.legend(title='Sampled Fraction', ncol=2)            
# fig.tight_layout()
# plt.show()
# fig.savefig('/media/dataHog/hca/msAcT-acylOctamer/results/BindingProfile_combinatorial_water.png', dpi=600, bbox_inches='tight')
# =============================================================================
    
    
    
    