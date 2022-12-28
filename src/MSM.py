# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:52:02 2021

@author: hcarv
"""

import os
import pyemma
import re
import collections
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import ticker as ticker
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FormatStrFormatter, AutoLocator, PercentFormatter, MaxNLocator, ScalarFormatter, LogLocator, LinearLocator, AutoMinorLocator
from matplotlib import cm
from matplotlib import colors as ml_colors
import numpy as np
import pandas as pd
import hvplot
import holoviews as hv
#hv.extension('bokeh')
#pd.options.plotting.backend = 'holoviews'
import pickle
import itertools
import glob
import nglview
try:
    import Discretize
    import Trajectory
    import tools_plots
    import tools
    import plots
except Exception as v:
    print(v)
    pass
import mdtraj as md
from mdtraj.utils import in_units_of
from simtk.unit import angstrom, picosecond
from itertools import compress, product
from functools import wraps
from sklearn.impute import SimpleImputer




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
    report_units = 'angstrom'
    report_flux_units = 'second'
    pathway_connect_symbol = ' -> '
    ITS_lags=[1, 2, 5, 10, 20, 40, 100, 250, 500, 750, 1000, 1500, 2000, 4000]
    ref_samples=2000
    visual_samples=100
    movie_samples=200
    generate_samples=40
    pathway_fraction = 0.95
    

    plot_specs = {'normal' : {'ViAc' : pl.cm.YlOrRd(np.linspace(0.2,1,3)),
                          'BuOH' : pl.cm.YlGnBu(np.linspace(0.2,1,4)),
                          'BeOH' : pl.cm.BuPu(np.linspace(0.2,1,5))},
                  'inhibition' : {'BuOH' : pl.cm.YlGnBu(np.linspace(0.2,1,4))[2],
                                  'BeAc' : 'darkorange', 
                                  'ViAc' : 'darkorange',
                                  'BeOH' : ('darkorange', pl.cm.BuPu(np.linspace(0.2,1,5))[3])},
                  'water' : {'H2O' : ['red'],
                              'BuOH' : np.vstack([pl.cm.YlGnBu(np.linspace(0.2,1,4)), pl.cm.Oranges(np.linspace(0.5,0.5,1))]),
                             'BeOH' : np.vstack([pl.cm.BuPu(np.linspace(0.2,1,5)), pl.cm.Oranges(np.linspace(0.5,0.5,1))])}}


    def __init__(self,
                 supra_project,
                 regions=None,
                 timestep=None,  
                 chunksize=0, 
                 stride=1,
                 skip=0,
                 warnings=False,
                 pre_process=True,
                 equilibration=False,
                 production=True,
                 def_top=None,
                 def_traj=None,
                 filter_water=True,
                 results=None,
                 overwrite=False):
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
        
        self.supra_project = supra_project
        
        self.regions=regions
        self.features={}
        self.chunksize=chunksize
        self.stride=stride
        self.skip=skip
        self.timestep = timestep
        
        self.warnings=warnings
        self.pre_process=pre_process
        self.w_equilibration = equilibration
        self.w_production = production

        self.filter_water=filter_water
        self.overwrite = overwrite
        self.results_path = results
        
        #print('Results will be stored under: ', self.results)
        #print('PyEMMA calculations will be stored under: ', self.stored)
        
        #self.def_top = def_top
        #self.def_traj = def_traj
        #if def_traj != None or def_top != None:
        #    print(f'Using pre-defined trajectory {self.def_traj} and/or topology {self.def_top}')
        

    
    def execute_or_load(func):
        """Decorator for object calculation of loading"""
        
        @wraps(func)
        def execute_or_load(self, *args, **kwargs):

            if not os.path.exists(self.file_name) or self.overwrite:
                print(f'Executing {func.__name__}', end='\r')
                try:
                    out = func(self, *args, **kwargs)
                    
                    if isinstance(out, tuple):
                        out[0].save(self.file_name, overwrite=True)
                    else:
                        out.save(self.file_name, overwrite=True)
                except Exception as v:
                    print(v.__class__, v)
                    out=None
                     
            else:
                #print(f'Loading data : {file_name}')
                out=pyemma.load(self.file_name)
            
            setattr(self, func.__name__, out)
            return out
        return execute_or_load

    
    #def plot_flux_lite(self, )



    def set_specs_lite(self, inputs):
        """Decorator for setting attributes"""


        (method, mol, idx, it, lag, disc_traj) = inputs
        self.method = method
        
                
        self.mol = mol
        self.systems = self.project.systems
        self.it = it
        self.idx = idx
        self.lag = lag
        if self.timestep is None:
            self.timestep= self.project.timestep
        self.unit=self.timestep.unit.get_symbol()
        if self.results_path != None:
            base_results = tools.Functions.pathHandler(self.project, self.results_path)
        else:
            base_results = self.project.results
        self.results =os.path.abspath(f'{base_results}/MSM')
        self.stored=os.path.abspath(f'{self.results}/MSM_storage')
        tools.Functions.fileHandler([self.results, self.stored])
        self.dt_scalar = int(str(self.timestep*(self.stride)).split(' ')[0])
        self._lag = str(lag*self.timestep).replace(" ", "")
        
        if method == 'MFPT':
            ext  = 'csv'
        else:
            ext = 'npy'
        
        if method == 'iTS':
            self.file_name = f'{self.stored}/{self.method}_{self.project_type}_{mol}_{it}_b{self.start}_e{self.stop}_s{self.msm_stride}.{ext}' 
        else:
            
            self.file_name = f'{self.stored}/{self.method}_{self.project_type}_{mol}_{it}_@{self._lag}_b{self.start}_e{self.stop}_s{self.msm_stride}.{ext}'
        self.name = f'{it} @{self._lag} ({self.start}:{self.stop}:{self.msm_stride})'
        self.disc_trajs = disc_traj 
        





    
    
    def run(self, measurements):

        data = collections.defaultdict(dict)

        for m in measurements:
            #print(f'Processing :  {m[:-1]} \n', end='\r')
            
            for method in self.methods:
                self.method = method
                self.set_specs_lite([method] + m)
                try:
                    func = self.method_functions[method]
                except KeyError:
                    raise ValueError('Method not defined: ', method)
                out = func()
                data[method][self.name] = out
        
        return data



    def lite(self, 
             supra_project, 
             lags, 
             input_data=None,
             extra_input_data = {},
             regions={}, 
             input_states={}, 
             extra_states={}, 
             plot_figure=False, 
             plot_its = True,
             plot_cktest = True,
             methods=['bayesMSM'],
             paths_to_plot=-1,
             water_mode='combinatorial',
             macrostates=0):
        
        self.msm_stride = 1
        self.lags= lags
        self.extra_input_data = extra_input_data
        self.water_mode= water_mode
        self.macrostates = macrostates
        self.input_states = input_states
        self.methods = methods
        self.method_functions = {'bayesMSM' : self.bayesMSM,
                                 'bayesHMSM' : self.bayesHMSM,
                                 'iTS' : self.iTS,
                                 'CKTest' : self.CKTest,
                                 'flux' : self.TPT,
                                 'MFPT' : self.MFPT}

        
        output = collections.defaultdict(dict)
        
        for project_type, projects in self.supra_project.items():
            
            out = {}
            self.project_type = project_type
            self.projects = projects
            
            print(project_type)

            if water_mode == 'combinatorial' or project_type != 'water':
                self.regions = regions[project_type]
                self.states = input_states[project_type][0]
                self.remap_states = input_states[project_type][1]
                
                input_df = input_data[project_type]
                self.start=input_df.index.values[0]
                self.stop=input_df.index.values[-1]
    
                for mol, project in projects.items():
                    self.project = project
                    
                
                    
                    if project_type == 'normal':
                        mols = [mol]
                    elif project_type == 'inhibition':

                        if mol == 'BeOH':
                            mols = [mol] #, 'BeAc'] 
                        elif mol == 'BuOH':
                            mols = [mol] #, 'ViAc']
                        #continue
                    else:
                        mols = input_df.columns.get_level_values('l2').unique()
                        
                    for _mol in mols:
                        measurements = []
                        data = input_df.loc[self.start:self.stop:self.msm_stride, (self.project.protein[0], _mol)]
                        its = data.columns.get_level_values('l3').unique()
                        #if project_type != 'normal' and _mol != 'H2O':
                        #    its = its[:-1]
                        for idx, it in enumerate(its):
                            lag = lags[project_type][_mol][idx]
                            disc_traj = data.loc[:, it].values.T.astype(int).tolist()
                            measurements.append([_mol, idx, it, lag, disc_traj])
                        out[_mol] = self.run(measurements)
            
            elif water_mode != 'combinatorial' or project_type == 'water':  
                self.regions = regions['water_double']
                self.project = projects['H2O']
                self.states = extra_states
                self.remap_states = extra_states #tools.Functions.state_remaper(extra_states)[1]
                
                input_df = extra_input_data
                mols = input_df.columns.get_level_values('l2').unique()


                self.start = input_df.index.values[0]
                self.stop = input_df.index.values[-1]
                
                for _mol in mols:       
                    
                    data = input_df.iloc[self.start:self.stop:self.msm_stride, input_df.columns.get_level_values('l2') == _mol]
                    its = data.columns.get_level_values('l3').unique() #[:-1]
                    #its = list(data.keys())[:-1]
                    measurements = []
                    for idx, it in enumerate(its):
                        lag = lags[project_type][_mol][idx]

                        _data = data.iloc[:, data.columns.get_level_values('l2') == mol]
                        its = _data.columns.get_level_values('l3')
                        disc_traj = _data.iloc[:, _data.columns.get_level_values('l3') == it].values.T.astype(int).tolist() #data[it]

                        measurements.append([_mol, idx, it, lag, disc_traj])
                    out[_mol] = self.run(measurements)
    
                        
                        
      
                #if plot_figure: 
                #    self.plot_kinetics(out, paths_to_plot=paths_to_plot)
                        
                #if plot_its:
                    
                #    self.plot_implied_timescales(out)
        
            output[project_type] = out
        
        
        self.plot_kinetics_lite(output)
        
        
        
        return output
    
    
    def plot_kinetics_lite(self, input_data):
        
        colors = {'BeOH': [pl.cm.BuPu(np.linspace(0.2,1,5)), 3, 's'], 
                  'BuOH' : [pl.cm.YlGnBu(np.linspace(0.2,1,4)), 2, '^'],
                  'ViAc' : [pl.cm.YlOrRd(np.linspace(0.2,1,3)), 1,  'o'],
                  'H2O' : ['red', 0, '*'],
                  'BeAc' : ['darkorange', 's'],
                  }
        plt.rcParams['font.size'] = '8'
        fig = plt.figure(figsize=(7.48, 5.61))
        subfigs = fig.subfigures(1,2, width_ratios=[3,2])
        subfigs_flux_path = subfigs[0].subfigures(2,1, height_ratios=[3,2])
        subfig_flux = subfigs_flux_path[0].subplots(1,1)
        subfig_water = subfigs[1].subplots(3,1, sharex=True)
        ax_flux = subfig_flux
        ax_ratios_flux = subfig_water[0]
        ax_k = subfig_water[1]
        ax_kd = subfig_water[2]
        subfig_pathway = subfigs_flux_path[1].subplot_mosaic([['scheme','ViAc', 'BeOH', 'BuOH']], sharey=True, gridspec_kw = {'width_ratios' :[2,3,3,3]})
        ax_scheme = subfig_pathway['scheme']
        plots.Plots.draw_scheme(ax_scheme, 
                                self.input_states['normal'][1], 
                                offset= 0.8, 
                                mode = 'combinatorial')
    
        compare_flux_data = {}
        compare_inhib_flux_data = {}
        for project_type, project_data in input_data.items():
            print(project_type)
            
            for mol, data in project_data.items():
                print(mol)
                its = [d.split('@')[0] for d in data['bayesMSM'].keys()]
                
                iterable_scalars = [] #[0]
        
                for it in its:
                    
                    try:
                        bulk_value=float(str(it).split('M')[0]) 
                    except:
                        bulk_value=float(str(it).split('mM')[0])
                    iterable_scalars.append(bulk_value)

                _color, idx_color, marker = colors[mol]
                color_flux = _color[idx_color]

                flux_data = [l[0]['net flux'].values for l in data['flux'].values()] 
                committor_data = [l[1]['Forward'] for l in data['flux'].values()]    
                statdist = [d.stationary_distribution for d in data['bayesMSM'].values()]
                
                mfpt = [l for l in data['MFPT'].values()]
                
                #plot committors
                if project_type != 'water':
                    ax_committor = subfig_pathway[mol]
                    
                    _states = self.input_states[project_type][1]
                    
                    if project_type == 'inhibition':
                        
                        committor_data = [committor_data[-1]]
                        statdist = [statdist[-1]]
                
    
                    for idx_it, it_comm in enumerate(committor_data):
                        comms_to_plot = []
                        idx_to_plot = []

                        
                        for idx, s in _states.items():
                            try:
                                comm_value = it_comm.loc[it_comm.index.get_level_values('states') == s].values[0]
                                comms_to_plot.append(comm_value)
                                idx_to_plot.append(idx)
                            except:
                                pass

                        if project_type == 'inhibition':
                            _color_plot = 'darkorange'
                            _color_edge = 'none'
                            linestyle = 'dashed'
                            
                        else:
                            _color_plot = _color[idx_it]
                            _color_edge = _color[idx_it]
                            linestyle = 'solid'
                        
                        size = np.digitize(statdist[idx_it], bins = np.geomspace(1e-5,1, num=6)) * 5

                        ax_committor.plot(comms_to_plot, 
                                          idx_to_plot, 
                                          color=_color_plot,
                                          linestyle = linestyle)
                        ax_committor.scatter(comms_to_plot,
                                             idx_to_plot,
                                             color= _color_plot,
                                             linestyle = linestyle,
                                             s = size,
                                             facecolor = _color_edge,
                                             edgecolor = _color_plot
                                             )
                        ax_committor.set_yticks(range(len(_states)))
                        ax_committor.set_yticklabels(_states.values())
                        ax_committor.set_xticks(np.arange(0,1.5,0.5))
                        
                        if mol == 'BeOH':
                            ax_committor.set_xlabel('Forward committor probability')

                
                if project_type == 'normal':
                    ax_flux.plot(iterable_scalars, 
                                 flux_data, 
                                 color=color_flux, 
                                 marker=marker, 
                                 linestyle='solid', 
                                 label = mol)
                    compare_flux_data[mol] = flux_data
                    
                elif project_type == 'inhibition':
                    label = f'{mol} + 5 mM {its[-1].split("_")[1]}'
                    ax_flux.scatter(100, 
                                    flux_data[-1], 
                                    facecolor='darkorange', 
                                    edgecolor='darkorange', 
                                    marker=marker, 
                                    label = label)
                    compare_inhib_flux_data[mol] = flux_data[-1]
                elif project_type == 'water':
                    
                    if mol != 'H2O':
                        ax_flux.plot(iterable_scalars[:-1], 
                                     flux_data[:-1], 
                                     color=color_flux, 
                                     marker=marker, 
                                     linestyle='dashed', 
                                     label = f'water in {mol}')
                        ax_flux.scatter(100, 
                                        flux_data[-1], 
                                        facecolor='none', 
                                        edgecolor='darkorange', 
                                        marker=marker)
                        
                        
                        compare_data = np.array(compare_flux_data[mol]) / np.array(flux_data[:-1])
                        ax_ratios_flux.plot(iterable_scalars[:-1], 
                                            compare_data, 
                                            color=color_flux, 
                                            marker=marker)
                        ax_ratios_flux.scatter(100, 
                                               compare_inhib_flux_data[mol] / flux_data[-1], 
                                               facecolor='darkorange', 
                                               edgecolor='darkorange', 
                                               marker=marker)
                        #on = mfpt.iloc[0,1]
                        #off = mfpt.iloc[1,0]
                        ons = []
                        offs = []
                        for m in mfpt:
                            
                            ons.append((1 / m.iloc[0,1]) *1e12)
                            offs.append((1 / m.iloc[1,0]) * 1e12)
                            
                        kds = 1 / (np.array(ons)*55.56 / np.array(offs))
                        
                        ax_k.plot(iterable_scalars[:-1], 
                                  ons[:-1], 
                                  marker = marker, 
                                  markerfacecolor = color_flux, 
                                  color=color_flux,
                                  label = r'k$_{on}$ / 55.56 M')
                        ax_k.plot(iterable_scalars[:-1], 
                                  offs[:-1], 
                                  linestyle ='dashed', 
                                  marker = marker, 
                                  markerfacecolor = 'none', 
                                  color=color_flux,
                                  label = r'k$_{off}$')
                        ax_kd.plot(iterable_scalars[:-1],
                                   kds[:-1]*1000,
                                   color=color_flux,
                                   marker = marker)

                        
                        
                    else:
                        ax_flux.axhline(flux_data[0], color='red')
                        
                        on_0 = 1 / mfpt[0].iloc[0,1] * 1e12
                        off_0 = 1 / mfpt[0].iloc[1,0] * 1e12
                        
                        kd_0 = 1 / (on_0 *55.56 / off_0)
                        
                        ax_k.axhline(on_0, color='red')
                        ax_k.axhline(off_0, color='red', linestyle='dashed')
                        ax_kd.axhline(kd_0*1000, color='red')
                        ax_kd.text(200, kd_0*1000, f'{np.around(kd_0*1000, decimals=2)} mM', color='red')
                        
                
                from matplotlib.lines import Line2D    
                water_labels = [Line2D([0], [0], marker='o', color='black', linestyle='dotted', label=r'k$_{off}$', markerfacecolor='none'),
                                Line2D([0], [0], marker='o', color='black', linestyle='solid', label=r'k$_{on}$ / 55.56 M', markerfacecolor='black')]
                ax_k.legend(handles=water_labels)
                ax_k.set_xscale('log')
                ax_k.set_yscale('log')
                ax_k.set_ylabel(r'rate (s$^{-1}$)')
                
                ax_kd.set_ylabel(r'K$_D$ = k$_{on}$/k$_{off}$ (mM)')
                ax_kd.set_xscale('log')
                ax_kd.set_xlabel('Concentration (mM)')
                
                ax_flux.set_xscale('log')
                ax_flux.set_yscale('log')
                ax_flux.legend() #bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
                ax_flux.set_xticks([])
                ax_flux.set_xlabel('Concentration (mM)')
                ax_flux.set_ylabel(r'Flux ($s^{-1}$)')
                ax_flux.xaxis.set_major_locator(plots.Plots.scalars_uniques)
                ax_flux.xaxis.set_major_formatter(FormatStrFormatter("%d") ) 
                
                ax_ratios_flux.set_ylabel(r'Flux$_t$ / Flux$_h$ $\approx$ r$_t$ / r$_h$')
                ax_ratios_flux.set_ylim(0,1)
                ax_ratios_flux.set_xscale('log')
                
                ax_ratios_flux.xaxis.set_major_locator(plots.Plots.scalars_uniques)
                ax_ratios_flux.xaxis.set_major_formatter(FormatStrFormatter("%d") ) 
        
        fig.savefig(f'{self.project.results}/figures/kinetics_all.png', dpi=600, bbox_inches='tight')
    
    def plot_kinetics(self, input_data_mols, paths_to_plot=-1):
        
        states = self.states
        
        
        fig =  plt.figure(figsize=(9,9), constrained_layout=True)
        
        

        
        outer_grid = gridspec.GridSpec(2, 2, width_ratios=[2,3], height_ratios=[1,3])
        flux_grid = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer_grid[0,1])
        pathway_grid = gridspec.GridSpecFromSubplotSpec(len(input_data_mols), 3, subplot_spec = outer_grid[1,1], width_ratios=[1,6,1], hspace=0.3)
        mfpt_grid = gridspec.GridSpecFromSubplotSpec(len(input_data_mols), 1, subplot_spec = outer_grid[1,0], hspace=0.3)
        
        

        mol_labels, labels_all = self.defineSubsetLabels(input_data_mols)
        
        for idx_mol, (mol, input_data) in enumerate(input_data_mols.items()):
            axes = {'source' : plt.subplot(pathway_grid[idx_mol,0]), 
                    'sink' : plt.subplot(pathway_grid[idx_mol,2]), 
                    'intermediate' : plt.subplot(pathway_grid[idx_mol,1])}

            
            if idx_mol == 0:
                axes_mfpt = plt.subplot(mfpt_grid[idx_mol, 0])
            else:
                axes_mfpt = plt.subplot(mfpt_grid[idx_mol, 0], sharex=axes_mfpt)
            
            axes_flux = plt.subplot(flux_grid[0])    
            
            
            labels = mol_labels[mol]
            fluxes = []
            
            
            if self.water_mode != 'combinatorial':
                its = [d.split('@')[0] for d in input_data['bayesMSM'].keys()] #self.supra_project['normal'][mol].parameter
            else:
                if mol == 'H2O':
                    its = self.supra_project[self.project_type][mol].parameter
                else:
                    its = [d.split('@')[0] for d in input_data['bayesMSM'].keys()] #input_data['bayesMSM'].keys() #self.supra_project['normal'][mol].parameter 
            
            #print(self.supra_project['normal'][mol].parameter, self.supra_project[self.project_type][mol].parameter)
            
            data_elements = (input_data['flux'].items(), input_data['bayesMSM'].values(), input_data['MFPT'].values())
            for idx, ((name, data), model, mfpt) in enumerate(zip(*data_elements)):
                print(idx, name)

                
                
                
                statdist = model.stationary_distribution
                _states_model = model.active_set
                _state_labels = [states[s] for s in _states_model]
                
                
                if self.project_type != 'inhibition':
                    color = MSM.plot_specs[self.project_type][mol][idx]
                    _edgecolor = color
            
                    #color_mol = plot_specs['normal'][mol_w][1] 
                    #color_mol = np.vstack([color_mol, pl.cm.Oranges(np.linspace(0.5,0.5,1))])
            
                else:
                    if mol != ('water') and idx == 0:
                        break
                    elif mol != ('water') and idx != 0:                        
                        color =  'red' #'white'
                        _edgecolor = 'red' #MSM.plot_specs[self.project_type][self.mol]
                    else:
                        #break
                        color = MSM.plot_specs[self.project_type][mol]
                        _edgecolor = color
                
                
                self.plot_mfpt(idx_mol, mol, mfpt, idx, color, _state_labels, axes_mfpt)
                #axes_mfpt.set_xlim(1e3, 5e13)
                #axes_mfpt.set_ylim(0, axes_mfpt.get_ylim()[1]+0.2)
                
                
                (flux_df, committor_df, pathway_df) = data
                
                try:
                    fluxes.append(flux_df.loc[:, 'net flux'].values[0])
                except AttributeError as v:
                    print(v)
                    #fluxes.append(np.nan)
                    
                self.plot_pathways(idx_mol, data, labels, statdist, paths_to_plot, fig, axes, color, _edgecolor)
            
            
# =============================================================================
#             if self.project_type == 'water':
#                 it_scalars = [float(str(it).split('mM')[0]) for it in its_to_plot[:-1]]
#                 _it_scalar_inhib = float(str(its_to_plot[-1]).split('mM')[0])
#                 fluxes = fluxes[:-1]
#                 _flux_inhib = fluxes[-1]
#                 print(it_scalars)
#                 
#                 axes_flux.scatter(_it_scalar_inhib, _flux_inhib, color='darkorange')
# =============================================================================        
            #for i,j in zip(its, fluxes):
            #    print(i, j)

            if its[0] == '55.56M':
                its[0] = '0mM'
            
            #plot flux
            it_scalars = [float(str(it).split('mM')[0]) for it in its]
            
            if self.water_mode == 'combinatorial' and self.project_type == 'water':
                color_plot = MSM.plot_specs[self.project_type][mol][0]
            else:
                color_plot = MSM.plot_specs['normal'][mol][-2]
            axes_flux.plot(it_scalars, fluxes, 'o-', color=color_plot, label=mol)
            
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True) 
            formatter.set_powerlimits((-1,1)) 
            axes_flux.yaxis.set_major_formatter(formatter) 
            axes_flux.yaxis.set_minor_locator(AutoMinorLocator())
            axes_flux.xaxis.set_minor_locator(AutoMinorLocator())
            axes_flux.set_xlabel('concentration (mM)')
            axes_flux.set_ylabel(r'Net flux ($s^{-1}$)')
            axes_flux.legend()
            
            for name, ax in axes.items():
                #ax.set_title(name)
                #if name != 'intermediate':
                ax.set_ylim(ax.get_ylim()[0]-1, ax.get_ylim()[1]+1)
                for key, spine in ax.spines.items():
                    spine.set_visible(False)

        fig.tight_layout()
        fig.show()
        
        fig.savefig(f'{self.project.results}/figures/kinetics_{self.project_type}.png', dpi=600, bbox_inches='tight')
        
        

    def plot_implied_timescales(self, input_data_mols):
        
        for mol, data in input_data_mols.items():
            
            
            models = data['iTS']
            for model, its in models.items():
                print(mol, model, its)
            #data[method][self.name]
                its_plot=pyemma.plots.plot_implied_timescales(its, units=self.unit, dt=self.dt_scalar)
            
                #plt.savefig()
                plt.show()
    
    @staticmethod
    def draw_connect(path_fraction, labels, coords, fig, axes, color, line):

        (s1, x1, y1, s2, x2, y2) = coords
        width = 4-(3-3*path_fraction/100) # 1+ 4*path_fraction
        draw = False
        if s1 in labels['source']: 
        
            xyA = (x1,y1)
            xyB = (x2, y2)
            coordA = axes['source'].transData
            coordB = axes['intermediate'].transData
            axA = axes['source']
            axB = axes['intermediate']
            draw = True

        elif s2 in labels['sink']:
             
            xyA = (x1, y1)
            xyB = (x2, y2)
            coordA = axes['intermediate'].transData
            coordB = axes['sink'].transData
            axA = axes['intermediate']
            axB = axes['sink']
            draw = True
        else:
            axes['intermediate'].plot((x1,x2), 
                                      (y1,y2), 
                                      color=color, 
                                      linestyle=line, 
                                      linewidth=width,zorder=1)#  2*path_fraction[0]/100)
            
        if draw:
            con = ConnectionPatch(xyA=xyA, 
                                  xyB=xyB, 
                                  coordsA=coordA, 
                                  coordsB=coordB,
                                  axesA=axA, 
                                  axesB=axB, 
                                  color=color, 
                                  linewidth=width, 
                                  linestyle=line,zorder=1)
            fig.add_artist(con)
    
    
    def plot_pathways(self, idx_mol, data, labels, statdist, paths_to_plot, fig, axes, color, _edgecolor):
                
        (_, committor_df, pathway_df) = data
        
        if not isinstance(committor_df, list):

        
            
            comm_data = committor_df.loc[:,'Forward']
            data_points = []
            for idx_q, (l, x) in enumerate(zip(comm_data.index.get_level_values(-1).values, comm_data.values)):
                
                if x == 1.0:
                    state_set = 'sink'
                elif x == 0.0:
                    state_set = 'source'
                else:
                    state_set = 'intermediate'
                y = labels[state_set].index(l) #np.where(labels[state_set] == l)
                z = 25 + 400*(statdist[idx_q])
            
                _plot = False
                for path in pathway_df.index.get_level_values('pathway').values[:paths_to_plot]:
                    path_states = path.split(MSM.pathway_connect_symbol)    
                    if l in path_states:
                        _plot= True
            
                if _plot:
                    axes[state_set].scatter(x, y, s= z, facecolors=color, edgecolors=_edgecolor) #, alpha=0.8)
                    data_points.append((l,x,y))
            data_points = np.asarray(data_points)

            if idx_mol == len(self.projects) - 1: 
                axes['intermediate'].set_xlabel('Forward committor probability')
            elif idx_mol == 0:
                for name, ax in axes.items():
                    ax.set_title(name)
            
            #axes['intermediate'].axvline(0.5, color='black', linestyle='dashed')
            axes['source'].axvline(0, color='black', linestyle='dashed')
            axes['sink'].axvline(1, color='black', linestyle='dashed')
            axes['sink'].get_xaxis().set_visible(False) 
            axes['source'].get_xaxis().set_visible(False)
            axes['intermediate'].set_xlim(-0.1,1.1)
            axes['source'].set_xlim(-0.5, 0.5)
            axes['sink'].set_xlim(0.5, 1.5)
            axes['intermediate'].set_xticks(np.arange(0,1.2,0.2))
            

            _label_box = axes['intermediate'].yaxis.get_ticklabels()
            [l.set_bbox(dict(facecolor='grey', edgecolor='none')) for l in _label_box]
            
            
            
            #axes['intermediate'].set_xscale('logit')
            #axes['source'].set_ylim(-1, 2)
            #axes['intermediate'].set_ylim(-1,3)
            #axes['intermediate'].set_ylim(-1,3)
        
            for idx_path, (pathway, path_fraction) in enumerate(zip(pathway_df.index.get_level_values('pathway').values[:paths_to_plot], 
                                                                    pathway_df.values[:paths_to_plot])):
                
                if np.argmax(pathway_df.values[:paths_to_plot]) == idx_path:
                    line = 'solid'
                else:
                    line = 'dashed' #loosely dashed (0, (5, 10))
                
                
                path_states = pathway.split(MSM.pathway_connect_symbol)
                for s1, s2 in zip(path_states, path_states[1:]):
                    
                    index1, index2 = np.where(data_points[:,0] == s1)[0][0],  np.where(data_points[:,0] == s2)[0][0]
                    x1, x2 = float(data_points[index1, 1]), float(data_points[index2,1])
                    y1, y2 = float(data_points[index1, 2]), float(data_points[index2,2])
                    
                    coords = (s1, x1, y1, s2, x2, y2)
                    self.draw_connect(path_fraction, labels, coords, fig, axes, color, line)

            for subset in axes.keys():
                axes[subset].set_yticks(range(len(labels[subset])))
                axes[subset].set_yticklabels(labels[subset], ha='center', va='center')
                axes[subset].tick_params(axis='y', width=0)
                axes[subset].set_axisbelow(False)
                if subset == 'sink':
                    pad = -40
                elif subset == 'intermediate':
                    pad = -40
                else:
                    pad = 10
                axes[subset].tick_params(axis='y', pad=pad) 
                #if subset != 'source':
                axes[subset].spines['left'].set_position('center')

    
    
    def plot_mfpt(self, idx_mol, mol, mfpt, idx, color, _state_labels, axes_mfpt):
        

        
        mfpt_array = mfpt.values / 1e12
        mfpt_max = np.max(mfpt_array[mfpt_array > 0])
        mfpt_min = np.min(mfpt_array[mfpt_array > 0])
        _ind_max = np.unravel_index(np.argmax(mfpt_array, axis=None), mfpt_array.shape)
        ind_min = tuple([l[0] for l in np.where(mfpt_array == mfpt_min)])
        ind_max = tuple((_ind_max)) #reversed
        

        label_max = ' -> '.join([_state_labels[i] for i in ind_max])
        label_min = ' -> '.join([_state_labels[i] for i in ind_min])

        print(f'{mol} {idx_mol} {np.shape(mfpt_array)}, min {mfpt_min}, max {mfpt_max}')
        
        axes_mfpt.scatter(mfpt_min, idx, s=50, color=color, facecolor='none')
        axes_mfpt.scatter(mfpt_max, idx, s=50, color=color, facecolor='none')
        
        lag = self.lags[self.project_type][mol][idx]*self.timestep._value / 1e12
        axes_mfpt.scatter(lag, idx, s=100, marker='|', color='red')
        for x in mfpt_array.flat:
            axes_mfpt.scatter(x, idx, s=10, color=color)
        axes_mfpt.annotate(label_max, xy=(mfpt_max, idx+0.1))
        axes_mfpt.annotate(label_min, xy= (mfpt_min, idx+0.1))
        axes_mfpt.set_ylabel(mol)
        axes_mfpt.spines['top'].set_visible(False)
        axes_mfpt.spines['right'].set_visible(False)
        #axes_mfpt.set_xlim(-0.05, 1.05)
        axes_mfpt.set_xscale('log')
        if self.water_mode != 'combinatorial':
            all_its = self.supra_project['normal'][mol].parameter
        else:
            if mol == 'H2O':
                all_its = self.supra_project[self.project_type][mol].parameter
            else:
                all_its = self.supra_project['normal'][mol].parameter
        axes_mfpt.set_yticks(range(len(all_its)))
        axes_mfpt.set_yticklabels(all_its)
        
        if idx_mol == len(self.projects) - 1: 
            axes_mfpt.set_xlabel('MFPT (s)')


    
    #@get_model
    def TPT(self, 
            macrostate=None, 
            mode='by_region',
            between=([], [])):
        """
        Function to calculate flux of model. A and B need to provided.

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        data_tuple : TYPE
            DESCRIPTION.
        macrostate : TYPE, optional
            DESCRIPTION. The default is None.
        between : TYPE, optional
            DESCRIPTION. The default is ([], []).
        top_pathways : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        None.

        """

        msm = self.bayesMSM
        n_state = len(msm.active_set)
        
        #colors = MSM.plot_specs['normal'][self.mol]

        
        lag_ =str(self.lag*self.timestep).replace(" ", "")
        if mode == 'by_region':
            between = self.regions
        
        #if self.mol == 'BeAc' or (self.mol == 'ViAc' and self.project_type == 'inhibition'):
   
        (A, B, I) = self.get_source_sink(between, msm, mode, macrostate)
        A_index = list(A.keys())
        B_index = list(B.keys())
        
        
        label_names = []
        
        
        label_A = [a[0] for a in A.values()]
        label_B = [b[0] for b in B.values()]
        label_I = [i[0] for i in I.values()]
        
        
        
        
        label_names = label_A + label_I + label_B
        
        if self.project_type == 'water':
            #print(self.states)
            #print(label_names)
            label_names = [self.states[s] for s in msm.active_set]
        

        
        
        if len(A_index) and len(B_index):
        
            if macrostate != None:
                flux_ =pyemma.msm.tpt(msm, A_index, B_index)
                cg, flux = flux_.coarse_grain(msm.metastable_sets)
            else:
                flux=pyemma.msm.tpt(msm, A_index, B_index)

    
            _outA , _outB = "-".join(map(str,label_A)), "-".join(map(str,label_B))
            flux_name = f'{self.name}_source{_outA}_sink{_outB}'
    # =============================================================================
    #         if self.make_tpt_movie:
    #             file_name = f'{flux_name}_movie'
    #             tpt_samples = msm.sample_by_distributions(msm.metastable_distributions, MSM.movie_samples)
    #             print('\tGenerating TPT trajectory ')
    #             self.save_samples(file_name, tpt_samples)
    # =============================================================================  
            base_index = [[self.project_type], [self.mol], [self.it], [f'{n_state}@{lag_}']]
            base_names = ['project_type', 'mol', 'iterable', 'model']

            comm_index = base_index + [label_names]
            comm_names = base_names + ['states']
    
            #Calculate commmittors
            index_row_comm= pd.MultiIndex.from_product(comm_index, names= comm_names)
            f_committor=flux.forward_committor
            b_committor=flux.backward_committor            
            committor_df=pd.DataFrame(zip(f_committor, b_committor), index=index_row_comm, columns=['Forward', 'Backward'])
    
            #Calculate fluxes
            index_row= pd.MultiIndex.from_product(base_index, names=base_names)
            net_flux = flux.net_flux
            net_flux_s = in_units_of(net_flux, f'second/{self.timestep.unit.get_name()}', 'second/second')
            rate_s = in_units_of(flux.rate, f'second/{self.timestep.unit.get_name()}', 'second/second')
            major_flux = flux.major_flux()
            gross_flux = flux.gross_flux
            total_flux = flux.total_flux
            if (net_flux.all() != major_flux.all() != gross_flux.all() != total_flux.all()):
                print('\tWarning! fluxes are not the same')
            unit = r'$s^{-1}$'
            report_flux = "{:.1e} {}".format(net_flux_s.sum(), unit)
            report_rate = "{:.1e} {}".format(rate_s, unit)
    
            
            #In units of gets the conversion of timestep to second
            flux_df=pd.DataFrame(columns=['net flux', 'rate'], index=index_row)
            flux_df.loc[:, 'net flux'] = net_flux_s.sum(1).sum()
            flux_df.loc[:, 'rate'] = rate_s
            
    
            
            #Calculate the pathways	
            paths, path_fluxes = flux.pathways(fraction=MSM.pathway_fraction)
            path_labels=[[] for _ in paths]
            for idx, path in enumerate(paths):
                for p in path:
                    path_labels[idx-1].append(label_names[p])
            path_labels=[MSM.pathway_connect_symbol.join(k) for k in path_labels] #
            
            path_index = base_index + [path_labels]
            path_name = base_names + ['pathway']
            index_col_path = pd.MultiIndex.from_product(path_index, names=path_name)
            #path_fluxes_s = in_units_of(path_fluxes, f'second/{self.timestep.unit.get_name()}', 'second/second')
            pathway_df=pd.DataFrame((path_fluxes/np.sum(path_fluxes))*100, index=index_col_path, columns=['% Total'])
    
    # =============================================================================
    #             if scheme == 'combinatorial':    
    #                 label_names=Functions.sampledStateLabels(regions, sampled_states=states, labels=labels)
    # =============================================================================               
            #return 
            return flux_df, committor_df, pathway_df
        else:
            print(f'Model {self.name} does not sample required states\n\t{self.states}\n\tsource: {A}\n\tsink: {B}')
            return [],[],[]

    #@get_model
    #@execute_or_load
    def MFPT(self, macrostates=None, hmsm_lag=False):
        
        
        
        
        def mfpt_states():
            
            
            #states = range(1, n_state+1)
            index_row=pd.MultiIndex.from_product([[s for s in state_labels]], names=['source'])
            #index_col=pd.MultiIndex.from_product([['mean', 'stdev'], [s for s in state_labels], ], names=['values', 'sink'])
            index_col=pd.Index(state_labels, name='sink')
            if not os.path.exists(self.file_name) or self.overwrite:
                print('calculating MFPT :', self.name)
                mfpt=pd.DataFrame(index=index_row, columns=index_col)
                #print('\tCalculating MFPT matrix...')
    
                mfpt_array = np.empty((len(state_labels), len(state_labels)))
                
                #mfpt_array = np.vectorize(lambda x : msm.sample_mean('mfpt'))
                
                for i, s in enumerate(state_labels):
                    for j, s2 in enumerate(state_labels):
                        mfpt.loc[s, s2]= msm.sample_mean("mfpt", i,j)
                        mfpt_array[i, j] = msm.sample_mean("mfpt", i,j)
                        #mfpt.loc[s, ('stdev', s2)]= msm.sample_std("mfpt", i-1,j-1)
                        #print(f'\t\ttransition: {i} {s} -> {j}  {s2}', end = '\r')
                
                
                #mfpt.replace(np.nan, 0, inplace=True)
                _ind_max = np.unravel_index(np.nanargmax(mfpt_array, axis=None), mfpt_array.shape)
                ind_min = np.unravel_index(np.nanargmin(np.nonzero(mfpt_array), axis=None), mfpt_array.shape)
    
                ind_max = tuple(reversed(_ind_max))
    
                label_min = [state_labels[i] for i in ind_max]
                label_max = [state_labels[i] for i in ind_min]
                

                mfpt.to_csv(self.file_name)
                #mfpt.dropna(how='all', inplace=True)
                #mfpt.dropna(how='all', axis=1, inplace=True)
                #reload again otherwise cannot be processed pcolormesh.
            else:
                #print('loading :', self.file_name)
                mfpt = pd.read_csv(self.file_name, index_col = 0) 
            
            means = mfpt #.iloc[: mfpt.columns.get_level_values(0) == 'mean']
            #means = MSM.mfpt_filter(mfpt, 0)


            
# =============================================================================
#             cmap_plot=plt.cm.get_cmap("Greys")
#             fig, axes = plt.subplots(1,1, figsize=(8,6), constrained_layout=True)
#             matrix = axes.pcolormesh(mfpt_array,
#                                 cmap=cmap_plot,
#                                 shading='auto', 
#                                 norm=ml_colors.LogNorm(vmin = self.lag, vmax = means.max().max()))  
#             rect_min = patches.Rectangle(ind_min, 1, 1, linewidth=1, edgecolor='green', facecolor='none')
#             rect_max = patches.Rectangle(ind_max, 1, 1, linewidth=1, edgecolor='red', facecolor='none')
#             axes.add_patch(rect_min)
#             axes.add_patch(rect_max)
#             # shading='auto',means[means > 0].min().min() vmin=self.lag,vmax=means.max().max()) #,
#             colorbar = plt.colorbar(matrix, orientation='vertical')
#             colorbar.set_label(label=f'MFPT ({self.unit})', size='large')
#             colorbar.ax.tick_params(labelsize=14)
#             axes.set_title(f'MFPT {self.name}')
#             axes.set_xticks(range(n_state))
#             axes.set_yticks(range(n_state))
#             axes.set_xticklabels(state_labels)
#             axes.set_yticklabels(state_labels)
#             cmap_plot.set_bad(color='white')
#             axes.set_xlabel('From state')
#             axes.set_ylabel('To state')
#             cmap_plot.set_under(color='black')
#             cmap_plot.set_bad(color='white')
#             cmap_plot.set_over(color='grey')
#             #plt.tight_layout()
#             plt.show()
#             fig.savefig(self.file_name}.png', dpi=600, bbox_inches="tight")
# =============================================================================

    
            return mfpt    
        
        
        def mfpt_cg_states():
            macrostates= range(1, macrostate+1)
            file_name_cg = f'{file_name}_{macrostate}macrostates'
            mfpt = np.zeros((macrostate, macrostate))
            if self.CG == 'HMSM':
                hmsm = self.HMSM(msm, [macrostate], hmsm_lag, plot=False)
                print('Coarse-graining with Hidden MSM')
                print(hmsm)
                for i in macrostates:
                    for j in macrostates:
                        print(f'Transition MS: {i} -> {j}', end='\r')
                        try:
                            mfpt[i-1, j-1] = hmsm.mfpt([i-1], [j-1])
                        except:
                            pass
            else:
                msm.pcca(macrostate)
                print('Coarse-graining with PCCA')
                for i in macrostates:
                    for j in macrostates:
                        print(f'Transition MS: {i} -> {j}', end='\r')
                        mfpt[i-1, j-1] = msm.mfpt(msm.metastable_sets[i-1], msm.metastable_sets[j-1])
                            
                    
            print('\n')
            cmap_plot=plt.cm.get_cmap("gist_rainbow")
            fig, axes = plt.subplots(1,1, figsize=(8,6), constrained_layout=True)
            try:
                matrix = axes.pcolormesh(mfpt, 
                                         cmap=cmap_plot, 
                                         shading='auto',
                                         norm=ml_colors.LogNorm(vmin = self.lag, vmax=mfpt.max().max()),
                                         vmin=self.lag,
                                         vmax=mfpt.max().max())
                                         #vmin=mfpt_all[mfpt_all > 0].min().min(),
                                         #vmax=mfpt_all.max().max(),
                                         #norm=ml_colors.LogNorm(vmin = mfpt_all[mfpt_all > 0].min().min(), vmax = mfpt_all.max().max())) 
                colorbar = plt.colorbar(matrix, orientation='vertical')
                colorbar.set_label(label=f'MFPT ({self.unit})', size='large')
                colorbar.ax.tick_params(labelsize=14)
                axes.set_title(f'MFPT {self.full_name} -> {macrostate} macrostates')
                axes.set_xticks(macrostates)
                axes.set_yticks(macrostates)
                cmap_plot.set_bad(color='white')
                axes.set_xlabel('From macrostate')
                axes.set_ylabel('To macrostate')
                cmap_plot.set_under(color='black')
                cmap_plot.set_bad(color='white')
                cmap_plot.set_over(color='grey')
                #plt.tight_layout()
                plt.show()
                fig.suptitle(f'{self.method}')
                fig.savefig(f'{self.results}/{file_name_cg}.png', dpi=600, bbox_inches="tight")
            
            except ValueError as v:
                print(v)
            #pyemma.plots.plot_network(inverse_mfpt, arrow_label_format=f'%f. {self.unit}', arrow_labels=mfpt, size=12) 
            
            
            return mfpt
        
        msm = self.bayesMSM
        states = msm.active_set
        state_labels = [self.states[s] for s in states]
        n_state = len(msm.active_set)

        mfpt = mfpt_states()
        if macrostates != None:
            for macrostate in macrostates:
                mfpt_cg_states()

        return mfpt


    @execute_or_load
    def iTS(self):
        its = pyemma.msm.its(self.disc_trajs, lags=MSM.ITS_lags, errors='bayes')
        
        
        self.its = its
        return its



    @execute_or_load
    def bayesMSM(self):
        
        msm = pyemma.msm.bayesian_markov_model(self.disc_trajs, lag=self.lag, dt_traj=str(self.timestep), conf=0.95)
        
        
        self.msm = msm
        return msm

    @execute_or_load
    def CKTest(self):
        try:
            msm = self.msm
        except AttributeError as v:
            print(f'Warning! {v} for {self.name}')
            msm = self.bayesMSM()
        
        cktest = msm.cktest(len(msm.active_set))
        ck_plot, axes=pyemma.plots.plot_cktest(cktest, 
                                                dt=self.dt_scalar, 
                                                layout='wide', 
                                                marker='.',  
                                                y01=True, 
                                                units=self.unit) 
        
        
        
        return cktest


    @execute_or_load
    def bayesHMSM(self):
        
        hmsm = pyemma.msm.bayesian_hidden_markov_model(self.disc_trajs, nstates=self.macrostates, dt_traj=str(self.timestep), conf=0.95, lag=self.lag)

        print(hmsm)
        
        print(hmsm.cktest())
        self.hmsm = hmsm
        return hmsm

    def defineSubsetLabels(self, input_data_mols):
            
            mol_labels = collections.defaultdict(dict)
            subset_labels = {'source' : [], 'sink' : [], 'intermediate' : []}
            
            for mol, input_data in input_data_mols.items():

                labels = {'source' : [], 'sink' : [], 'intermediate' : []}
                for idx, model in enumerate(input_data['bayesMSM'].values()):
                
                    (A, B, I) = self.get_source_sink(self.regions, model, mode='by_regions')
                    
                    labels['source'].append([a[0] for a in A.values()])
                    labels['sink'].append([b[0] for b in B.values()])
                    labels['intermediate'].append([i[0] for i in I.values()])
                
                for subset, _labels in labels.items():
                    labels_subset = []
                    sampled_labels = set(list(itertools.chain(*_labels)))
                    for state, label in self.states.items():
                        if label in sampled_labels:
                            labels_subset.append(label)
                    mol_labels[mol][subset] = labels_subset
                    for l in labels_subset:
                        subset_labels[subset].append(l)
            
            
            labels_all = {'source' : [], 'sink' : [], 'intermediate' : []}
            for subset, label_lists in subset_labels.items():
                labels_all[subset] = np.unique(np.asarray(label_lists))
                
            
            return mol_labels, labels_all



    def get_source_sink(self, between, msm, mode, macrostate=None):
        #TODO: separate macrostate
        
        def set_labels(X):
            label= []
            for idx, set_ in enumerate(sets, 1):
                for x in X:
                    if x in set_:
                        label.append(f'MS {idx}')

            return list(set(label))
        
        _A, _B = between[0], between[1]
        if len(_A) > 0 and len(_B) > 0:
            pass
        else:
            raise ValueError("Source or sink states not defined (between =([A], [B])")

        if mode == 'manual':
            print('Warning!!!!!!!  test properly')
            A = [a-1 for a in A]#since user starts at index 1.
            B = [b-1 for b in B]
        
            #TODO: make adap for hmsm
            
            if isinstance(macrostate, int): 
                 msm.pcca(macrostate)
                 sets = msm.metastable_sets
                 _A = np.concatenate([sets[a] for a in A]) 
                 _B = np.concatenate([sets[b] for b in B])
                 label_names = [f'MS {i}' for i in range(1, macrostate+1)]  
    
                 
                 not_I = list(_A)+ list(_B)
                 I = list(set(not_I) ^ set(msm.active_set))
                 label_A = set_labels(_A)
                 label_B = set_labels(_B)
                 label_I = set_labels(I)
                 print(f'\tPerforming coarse-grained TPT from A->I->B\n\tA = {label_A}\n\tB = {label_B}\n\tI = {label_I}')
                 label_names_reord = label_A+label_I+label_B
                 cmap = pl.cm.Set1(np.linspace(0,1,macrostate))
                 colors = []
                 for reord in label_names_reord:
                     idx = label_names.index(reord)
                     colors.append(cmap[idx])
                     
                 return (_A, label_A, _B, label_B, label_names_reord)
                

        else:

            A = {idx : (self.states[a], a) for idx, a in enumerate(msm.active_set) if a in _A}
            B= {idx : (self.states[b], b) for idx, b in enumerate(msm.active_set) if b in _B}

            _I = np.setdiff1d(range(len(msm.active_set)), list(A.keys()) + list(B.keys()))
            _I_index = [msm.active_set.tolist()[i] for i in _I]            
            _I_label = [self.states[i] for i in _I_index]            
            I = dict(zip(_I, zip(_I_label, _I_index)))
            #print(f'\tPerforming  TPT from A->I->B\n\tA = {A}\n\tB = {B}\n\tI = {I}')            
            
            return (A,B,I)



        
    
    @staticmethod
    def plot_correlations(df,
                          target='Score', 
                          methods=['pearson', 'spearman'], 
                          evaluate_scalars=['Lag', 'tICA lag', 'Dimensions', 'States', 'Processes'],
                          evaluate_strings=['name', 'feature', 're-weighting'],
                          by='name'):

        correlations=[[] for i in range(len(methods))]
        features_to_evaluate=evaluate_scalars+evaluate_strings
        rows, columns = tools_plots.plot_layout(methods)
        plot, axes = plt.subplots(rows,columns, sharey=True, sharex=True, figsize=(9,6))
        for idx, method in enumerate(methods):
            for feature in evaluate_scalars:
                correlations[idx].append(df[target].corr(df[feature], method=method))
            for str_ft in evaluate_strings:
                correlations[idx].append(df[target].corr(df[str_ft].astype('category').cat.codes, method=method))
         
        for idx, (correlation, ax) in enumerate(zip(correlations, axes)):
            ax.bar(features_to_evaluate, correlation, align='center')
            ax.tick_params(labelrotation=30, axis='x')
            ax.axhline(0, color='k')
            ax.set_title(methods[idx])
        plt.show()
 
    
    def model_comparison(self, has_tica=False, vamp_cross_val=True):
        
# =============================================================================
#         def get_mean(vamp_):
#             vamp_str = [i for i in vamp_.strip(' []').split(' ')]
#             vamps = []
#             for i in vamp_str:
#                 try:
#                     vamps.append(float(i))
#                 except:
#                     pass
#             return np.mean(vamps)
# =============================================================================
        


        files=glob.glob(f'{self.results}/ModelRegistry_*.csv')
        df_=pd.DataFrame() 

        #print(count)
        if has_tica:
            fix = ['Discretized feature', 'feature', 'name', 'model']
            cols = [0,1,2,3]
        else:
            fix = ['feature', 'name', 'model']
            cols = [0,1,2]
        if len(files) > 0:
            print(f'{len(files)} files collected.')
            for idx, f in enumerate(files, 1):
                     
                df=pd.read_csv(f, index_col=cols)
                df_=pd.concat([df_, df])
        else:
            print('No files found.')    
        
        fix_=df_.loc[:,~df_.columns.str.contains('VAMP2')].columns.to_list()
        fix+=fix_
        df_.reset_index(inplace=True)
        if vamp_cross_val:
            print('Subset with VAMP2 cv')
            df_= df_.melt(id_vars=fix, var_name="VAMP2cv", value_name="Score")  
        else:
            
            pass
            print('Subset wit single VAMP2')
            #To obtain vamp from bayesMSM
# =============================================================================
#             df_.drop(list(df_.filter(regex ='VAMP2 \d+')), axis = 1, inplace = True)
#             #df_= df_.melt(id_vars=fix, var_name="VAMP2", value_name="Score")
#             #df_ = df_.loc[:, fix]
# =============================================================================
        self.model_df = df_       

        return self.model_df
    
# =============================================================================
#     def pca_models(self):
#        from sklearn.preprocessing import StandardScaler
#        from sklearn.impute import SimpleImputer
#        from sklearn.decomposition import PCA
#         from sklearn.decomposition import PCA
#         import pandas as pd
#     
#         features_to_evaluate=['Lag', 'tICA lag', 'Score', 'Dimensions', 'States']
#         extracted_features=passed_models.loc[:, features_to_evaluate].values
#         scaled_features = StandardScaler().fit_transform(extracted_features)
#         cleaned_features = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(scaled_features)
#         
#         
#         pca=PCA(0.95)
#         principal_components=pca.fit_transform(cleaned_features)
#         principal_components.explained_variance_ratio_(5)
#         for idx, pc in enumerate(principal_components.T, 1):
#         passed_models=pd.concat([passed_models, pd.DataFrame({f'PC {idx}': pc})], axis=1)
# =============================================================================

    
    def log(func):
        """Decorator for system setup"""
        
        @wraps(func)
        def model(*args, **kwargs):
            print(f'Executing {func.__name__} \n', end='\r')
            return func(*args, **kwargs)
        return model

    @tools.log
    def create_registry(self, msm, vamp_iters=10):
        """
        

        Parameters
        ----------
        msm : TYPE
            DESCRIPTION.
        vamp_iters : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        passed : TYPE
            DESCRIPTION.

        """

        lag_ =str(self.lag*self.timestep).replace(" ", "")
        msm_name = f'{self.n_state}@{lag_}'
        stored_parameters = ['Test', 'Filters', 'Processes', 'States', 'Lag', 'Cluster method']
        df_name = f'{self.results}/ModelRegistry_{self.ft_base_name}.csv'
        if self.get_tica:
            cols=[0,1,2,3]
            index_row= pd.MultiIndex.from_product([[self.disc_ft_name], [self.feature], [self.ft_name], [msm_name]], 
                                                  names=['Discretized feature', 'feature', 'name', 'model'])
            location = (self.disc_ft_name, self.feature, self.ft_name, msm_name)
            stored_parameters.append('Dimensions')
            stored_parameters.append('tICA lag')
            if self.tica_weights:
                stored_parameters.append('re-weighting')
        else:
            cols = [0,1,2]
            index_row= pd.MultiIndex.from_product([[self.feature], [self.ft_name], [msm_name]], 
                                                  names=['feature', 'name', 'model'])
            location = (self.feature, self.ft_name, msm_name)
        
        if os.path.exists(df_name):
            df=pd.read_csv(df_name, index_col=cols)           
        else:
            df=pd.DataFrame()
        
        
        if msm != None:
            resolved_processes = self.spectral_analysis(msm, self.lag, plot=False)
            
            if self.get_vamp:
                _, disc_trajs, _ = self.load_discTrajs() #clusters=self.clustering(self.n_state)
                if self.w_cv:
                    vamp_scores = [f'VAMP2 {i}' for i in range(1, vamp_iters+1)]
                    for v_s in vamp_scores:
                        stored_parameters.append(v_s)
                    try:
                        print('\tCalculating a MSM surrogate and VAMP2 score with cross-validation...', end='\r')
                        score_model = pyemma.msm.estimate_markov_model(disc_trajs, lag=self.lag, dt_traj=str(self.timestep))
                        vamp = score_model.score_cv(disc_trajs, n=vamp_iters, score_method='VAMP2', score_k=min(10, self.n_state)) 
                    except Exception as v:
                        passed = False
                        print(f'\tMSM estimation failed: {v}')
                        vamp = [np.nan]*vamp_iters
                else:
                    print('\tCalculating single VAMP2 score...', end='\r')
                    stored_parameters.append('VAMP2')
                    try:
                        vamp = msm.score(disc_trajs, score_method='VAMP2', score_k=min(10, self.n_state))
                    except Exception as v:
                        passed = False
                        print(f'MSM estimation failed: {v}')
                        vamp = np.nan  
                        
            keep = self.filter_model(msm)
            
            filters_ = ' '.join(list(compress(self.filters, keep)))
            if all(keep):
                #print(f'\tModel {self.full_name} passed all filters \n', end='\r')
                passed = True
            else:
                passed = False
        else:
            print('Warning! no MSM model specified.')
            passed = 'Failed'
            
        df_ = pd.DataFrame(index=index_row, columns=stored_parameters)
        df_.loc[location,'Test'] = passed
        
        if passed != 'Failed':

            if self.get_vamp:
                if self.w_cv:
                    for idx, v in enumerate(vamp, 1):
                        df_.loc[location,f'VAMP2 {idx}'] = v
                else:
                    df_.loc[location, 'VAMP2'] = vamp
            df_.loc[location,'Filters'] = filters_
            df_.loc[location,'Processes'] = resolved_processes
            df_.loc[location,'States'] = self.n_state
            df_.loc[location,'Lag'] = self.lag
            if self.get_tica and self.discretized_data != None:
                df_.loc[location, 'Dimensions'] = self.discretized_data.dimension()
                df_.loc[location, 'tICA lag'] = self.d_lag
                if self.tica_weights:
                    df_.loc[location, 're-weighting'] = self.tica_weights

        df=pd.concat([df, df_])
        df.to_csv(df_name)
        
        #print(df)

        return passed

    @tools.log    
    def check_registry(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        df_name = f'{self.results}/ModelRegistry_{self.ft_base_name}.csv'

        lag_ =str(self.lag*self.timestep).replace(" ", "")
        msm_name = f'{self.n_state}@{lag_}'
        if self.get_tica:
            location = (self.disc_ft_name, self.feature, self.ft_name, msm_name)
        else:
            location = (self.feature, self.ft_name, msm_name)
        try:
            df=pd.read_csv(df_name, index_col=[0,1,2,3])
            try:
                passed = df.loc[location,'Test']
                if isinstance(passed, np.bool_) or passed == 'Failed':
                    return passed
                elif isinstance(passed, pd.core.series.Series):
                    print('Bad data: ', self.full_name)

                    print(passed)
                    passed = passed.iloc[input('Which one? ')]
                    print(passed)
                    return passed
                
                elif isinstance(passed, str):
                    df.loc[location, 'Test'] = bool(passed)
                    df.to_csv(df_name)
                    return bool(passed)
                else:
                    print(passed)

            except KeyError as v:
                #print('No entry found for: ', self.full_name, v)
                return None
        except FileNotFoundError:
            #print('DataFrame not found: ', df_name)
            return None    
 
    def filter_model(self, msm):
        """
        

        Parameters
        ----------
        msm : TYPE
            DESCRIPTION.

        Returns
        -------
        keep : TYPE
            DESCRIPTION.

        """
        resolved_processes = self.spectral_analysis(msm, self.lag, plot=False)
        keep = []
        #print('Filtering models based on filters:' , self.filters)
        for filter_ in self.filters:
            if filter_ == 'connectivity':
                fraction_state = np.around(msm.active_state_fraction, 3)
                if fraction_state == 1.000:
                    keep.append(True)
                else:
                    keep.append(False)
                    #print(f'\tWarning! Model {self.full_name} is disconnected: {fraction_state} states, {fraction_count} counts')
            elif filter_ =='time_resolved':
                
                if resolved_processes > 1:
                    keep.append(True)
                else:
                    keep.append(False)
                    #print(f'\tWarning! Model {self.full_name} has no processes resolved above lag time.')
            elif filter_ =='counts':
                fraction_count = np.around(msm.active_count_fraction, 3)
                if fraction_count == 1.000:
                    keep.append(True)
                else:
                    keep.append(False)
            elif filter_ == 'reversible':
                reversible = msm.reversible
                if reversible:
                    keep.append(True)
                else:
                    keep.append(False)
            elif filter_ =='first_eigenvector':
                eigvec = msm.eigenvectors_right()
                allclose = np.allclose(eigvec[:, 0], 1, atol=1e-15)
                if allclose:
                    keep.append(True)
                else:
                    keep.append(False)
                    #print('\tWarning! first eigenvector is not 1 within error tol. 1e-15)')
        return keep   

    
    @tools.log
    def load_models(self, load_discretized=True):
        """
        

        Parameters
        ----------
        load_discretized : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        models_loaded : TYPE
            DESCRIPTION.

        """

        
        def load():
            for lag in lags:
                for n_state in n_states:
                    self.set_specs(ft_name, feature, lag=lag, n_state=n_state, set_mode='MSM')
                    if self.full_name in models:
                        if self.method != 'generate':
                            model = self.bayesMSM_calculation(lag)
                        else:
                            model = None
                        if self.get_tica:
                            models_loaded[self.full_name] = (model, ft_name, feature, n_state, lag, d_lag)
                        else:
                            models_loaded[self.full_name] = (model, ft_name, feature, n_state, lag)
                    elif self.full_name in models_to_load:
                        model = self.bayesMSM_calculation(lag)
                        filter_passed=self.create_registry(model, vamp_iters=10)
                        try:
                            model = self.bayesMSM_calculation(lag)
                            filter_passed=self.create_registry(model, vamp_iters=10)
                            if filter_passed:
                                if self.get_tica:
                                    models_loaded[self.full_name] = (model, ft_name, feature, n_state, lag, d_lag)
                                else:
                                    models_loaded[self.full_name] = (model, ft_name, feature, n_state, lag)
                            else:
                                models_failed.append(self.full_name) 
                        except Exception as v:
                            print(f'Warning! model failed due to {v.__class__}\n{v}')
                            models_failed.append(self.full_name)
                    else:
                        models_failed.append(self.full_name)
                        pass #print('Something else: ', self.full_name)
        def check():
            for lag in lags:
                for n_state in n_states:
                    self.set_specs(ft_name, feature, lag=lag, n_state=n_state, set_mode='MSM')
                    keep = self.check_registry()
                    if keep == True:
                        #print(f'Model {self.full_name} passed filter(s) ', end='\r')
                        models.append(self.full_name)
                    elif keep == False:
                        #print(f'Model {self.full_name} failed filter(s) ', end='\r')
                        models_discard.append(self.full_name)
                    elif keep == None:
                        #print('No model found for :', self.full_name)
                        models_to_load.append(self.full_name)
                        load_discretized_feature.append(self.ft_base_name)
                    elif keep == 'Failed':
                        #print('Failed model for :', self.full_name)
                        models_failed.append(self.full_name)
                    else:
                        print('something wrong')
                        pass
            
        models = []
        models_to_load = []
        models_loaded = {}
        models_failed = []
        models_discard = []
        load_discretized_feature = []
        
        #check model status
        for input_params in self.inputs:
            (ft_name, feature, n_state_sets, lag_sets) = input_params[:4]
            n_states = self.check_inputs(n_state_sets)
            #TODO: n_states to be fetched from self.discretized_data, or some other auto mechanism.
            lags = self.check_inputs(lag_sets)
            
            if self.get_tica:
                for d_lag in self.disc_lag:
                    self.d_lag = d_lag
                    self.set_specs(ft_name, feature, set_mode='base')
                    check()
            else:
                self.set_specs(ft_name, feature, set_mode='base')
                check()

        print('\tModels calculated: ', len(models))
        print('\tModels to discard: ', len(models_discard))
        print('\tModels to calculate: ', len(models_to_load))  

        #load models and create new ones
        for input_params in self.inputs:
            (ft_name, feature, n_state_sets, lag_sets) = input_params[:4]
            n_states = self.check_inputs(n_state_sets)
            lags = self.check_inputs(lag_sets)
            

            if self.get_tica:
                for d_lag in self.disc_lag:
                    self.d_lag = d_lag
                    self.set_specs(ft_name, feature, set_mode='base')
                    if self.ft_base_name in set(load_discretized_feature):
                        self.load_discretized_data(feature=feature)
                    load()         
            else:
                self.load_discretized_data(feature=feature)
                load()
        if self.method == 'generate':        
            print('\tFailed models: ', len(models_failed))
            print('\tLoaded models: ', len(models_loaded))
            print('\tTotal number of models :', len(models_failed)+len(models_loaded))
        
        return models_loaded
    
    

    
    @tools.log
    def set_specs(self, ft_name, feature, lag=1, n_state=0, set_mode='base'):
        """
        

        Parameters
        ----------
        ft_name : TYPE
            DESCRIPTION.
        feature : TYPE
            DESCRIPTION.
        lag : TYPE, optional
            DESCRIPTION. The default is 1.
        n_state : TYPE, optional
            DESCRIPTION. The default is 0.
        set_mode : TYPE, optional
            DESCRIPTION. The default is 'base'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        lag_ =str(lag*self.timestep).replace(" ", "")
        
        self.ft_name = ft_name
        self.feature = feature
        if self.get_tica: 
            disc_lag_=str(self.d_lag*self.timestep).replace(" ", "")
            self.disc_ft_name = f'{feature}@{disc_lag_}'
            if self.tica_weights:
                append = f'@{disc_lag_}_w-{self.tica_weights}'
            else:
                append = f'@{disc_lag_}'
                self.tica_weights='empirical'
        else:
            append = ''

        self.ft_base_name = f'{self.ligand}_{self.iterable}_{ft_name}_{self.feature}{append}_s{self.stride}'
        if set_mode == 'MSM':
            if self.cluster_method == 'regspace':
                append2 = f'_{self.cluster_method}'
            else:
                append2 = ''
            if isinstance(self.regions, dict):
                self.selections = self.regions[ft_name]
            else:
                self.selections = self.ft_name
                #TODO: make more robust maybe
            self.lag = lag
            self.n_state = n_state
            self.full_name = f'{self.ft_base_name}_{n_state}@{lag_}{append2}'
            return self.full_name
        else:    
            return self.ft_base_name

    @tools.log
    def set_systems(self):
        systems_specs=[]
        for name, system in self.systems.items():
            if system.parameter == self.iterable:
                results_folder=system.results_folder
                topology=Trajectory.Trajectory.fileFilter(name, 
                                                            system.topology, 
                                                            self.w_equilibration, 
                                                            self.w_production, 
                                                            def_file=self.def_top,
                                                            warnings=self.warnings, 
                                                            filterW=self.filter_water)
                
                trajectory=Trajectory.Trajectory.fileFilter(name, 
                                                            system.trajectory, 
                                                            self.w_equilibration, 
                                                            self.w_production,
                                                            def_file=self.def_traj,
                                                            warnings=self.warnings, 
                                                            filterW=self.filter_water)
                systems_specs.append((trajectory, topology, results_folder, name))            

        trajectories_to_load = []
        tops_to_load = []
        for system in systems_specs: #trajectory, topology, results_folder, name
            #TODO: check if only trj_list[0] is always the enough.
            trj_list, top=system[0][0], system[1][0] #[0] because only a single top must be handed out and Trajectory returns a list.
            if len(trj_list):
                trajectories_to_load.append(trj_list)
                tops_to_load.append(top)

        
        self.topology=tops_to_load[0]
        self.trajectories=trajectories_to_load
        
        
        superpose_indices=md.load(self.topology).topology.select(MSM.superpose)
        subset_indices=md.load(self.topology).topology.select(MSM.subset)
        md_top=md.load(self.topology) #, atom_indices=ref_atom_indices )
        
        if self.pre_process:
            self.trajectories=Trajectory.Trajectory.pre_process_trajectory(self.trajectories, md_top, superpose_to=superpose_indices)
            md_top.atom_slice(subset_indices, inplace=True)
            self.topology = md_top

   
    @log
    def load_discretized_data(self, feature=None):
        
        
        print(self.ft_name)
        print(self.regions)
        
        
        if self.get_tica:
            self.discretized_data=self.TICA_calculation(self.d_lag, feature)
        else:
            #TODO: make here entry for other discretization schemes
            #As in TICA calculation call inside to get inputs
            try: 
                inputs_ = self.selections
            except AttributeError:
                print('No selections found. Getting input information from MSM.regions')
                inputs_ = self.regions[self.ft_name]

            print(f'{self.project.results}/{feature}_{inputs_}_{self.iterable}_*.csv')

            file = glob.glob(f'{self.project.results}/{feature}_{inputs_}_{self.iterable}_*.csv')
            if len(file) == 1:
                self.discretized_data = file[0]
            else:
                print(file)
                self.discretized_data = file[int(input('More than one file found. Which one (index)?'))]
# =============================================================================
#             self.discretized_data = self.load_features(self.tops_to_load, 
#                                           self.trajectories_to_load, 
#                                           [feature], 
#                                           inputs=inputs_)[0] #because its yielding a list.
# =============================================================================

    def load_discTrajs(self):
        #Warning: some function calls might need np.concatenate(disc_trajs) others don't.
        #discretized_data = self.load_discretized_data(feature=self.feature)
        if self.get_tica:
            data_concat = np.concatenate(self.discretized_data.get_output())
            clusters=self.clustering(self.n_state) #tica, n_clusters
            disc_trajs = clusters.dtrajs #np.concatenate(clusters.dtrajs)
        else:
            
            data_concat = None
            clusters= None
            df = pd.read_csv(self.discretized_data, index_col = 0, header = Discretize.Discretize.headers[:-2])
            disc_trajs = df.values.T.astype(int).tolist()
        
        return data_concat, disc_trajs, clusters







    def analysis(self, 
                 inputs, 
                 tica_lag=None, 
                 method=None, 
                 dim=-1,
                 filters=['connectivity', 'counts', 'time_resolved', 'reversible', 'first_eigenvector'],
                 compare_pdbs = [],
                 compare_dists = [],
                 hmsm_lag = False,
                 eval_vamps=True,
                 vamp_cross_validation=True,
                 overwrite=False,
                 cluster_method='k-means',
                 tica_weights=False):
        
        self.overwrite=overwrite
        self.viewer = {}
        self.dim = dim
        self.filters = filters
        self.cluster_method= cluster_method
        self.get_vamp = eval_vamps
        self.w_cv = vamp_cross_validation
        self.method = method
        if tica_lag != None:
            self.get_tica=True
            self.disc_lag = self.check_inputs(tica_lag)
            self.tica_weights = tica_weights
        else:
            self.get_tica=False
        if hmsm_lag != False:
            self.CG = 'HMSM'
        else:
            self.CG = 'PCCA'
        self.make_tpt_movie=True
        #TODO: make default CG None in case not required.
        
        if isinstance(inputs, list):
            for v in inputs:
                if not 3 < len(v) <= 6:
                    raise ValueError('Input_params has to be tuple of kind: (region, feature, n_state(s), lag(s), *[n. macrostate(s), flux input(s)])')
            self.inputs = inputs
        else:
            raise TypeError('Input has to be a list of kind: region : [(input_params1), (input_params2)]')

        flux, committor, pathway = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        self.ligand = self.project.ligand[0]
        for iterable in self.project.parameter:
            print(iterable)
            self.iterable = iterable
            self.set_systems()
            models = self.load_models() #(model, ft_name, feature, n_state, lag, d_lag)

            out_data_models = []
            for name, model in models.items():
                if method == 'generate':
                    pass
                else:
                    out_data = []
                    if self.get_tica:
                        (msm, ft_name, feature, n_state, lag, d_lag) = model
                    else:
                        (msm, ft_name, feature, n_state, lag) = model   
                    self.set_specs(ft_name, feature, lag, n_state, set_mode='MSM')
                    if method == 'PCCA' or method == 'CKtest':
                        self.load_discretized_data(feature=feature)
                    
                    name_ = f'{ft_name}-{feature}'
                    for input_params in self.inputs:
                        if f'{input_params[0]}-{input_params[1]}' == name_:
                            try:
                                macrostates = self.check_inputs(input_params[4])
                            except IndexError:
                                macrostates = None
                                self.CG = None
                    if method == 'PCCA':
                        self.PCCA_calculation(msm, macrostates, auto=False, dims_plot=10)
                        
                    elif method == 'CKtest':
                        self.CKTest_calculation(msm, macrostates)
        
                    elif method == 'Spectral':
                        self.spectral_analysis(msm, lag, plot=True)
                    elif method == 'MFPT':
                        self.MFPT_calculation(msm, macrostates, hmsm_lag)
                    elif method == 'Visual':
                        for macrostate in macrostates:
                            state_samples = self.extract_metastates(msm, macrostate, hmsm_lag=hmsm_lag, set_mode='visual') 
                            self.viewer[f'{self.full_name}-{macrostate}'] = self.visualize_metastable(state_samples)
                    elif method == 'RMSD':
                        for macrostate in macrostates:
                            state_samples = self.extract_metastates(msm, macrostate, hmsm_lag=hmsm_lag)
                            self.state_rmsd_comparison(state_samples, pdb_files = compare_pdbs)
                    elif method == 'distances':
                        for macrostate in macrostates:
                            state_samples = self.extract_metastates(msm, macrostate, hmsm_lag=hmsm_lag)
                            out_data=self.state_distance_comparison(state_samples, distances=compare_dists)
                    elif method == 'get_samples':
                        for macrostate in macrostates:
                            self.extract_metastates(msm, macrostate, hmsm_lag=hmsm_lag, set_mode='generate')
                    elif method == 'flux':
                        try:
                            flux_inputs = input_params[5]
                        except IndexError as v:
                            print(v)
                        for macrostate in macrostates:
                            flux_df, committor_df, pathway_df = self.flux_calculation(msm, macrostate=macrostate, between=flux_inputs)
                            flux = pd.concat([flux, flux_df], axis=0)
                            committor = pd.concat([committor, committor_df], axis=0)
                            pathway = pd.concat([pathway, pathway_df], axis=0)
    
                    elif method == 'HMSM':
                        self.HMSM(msm, macrostates, hmsm_lag)
                    elif method == 'ITS':
                        self.ITS_calculation(c_stride=1)
                    else:
                        raise SyntaxError('Analysis method not defined')
            
                    out_data_models.append(out_data)
        
        
        if method == 'flux':
            print(flux)
            flux.to_csv(f'{self.results}/flux_all.csv')
            committor.to_csv(f'{self.results}/committor_all.csv')
            pathway.to_csv(f'{self.results}/pathway_all.csv')
            print(committor)
            print(pathway)
        if method == 'Visual':
            return self.viewer
        if len(out_data_models):
                    
            return out_data_models

    def calculate(self,
                  inputs=None,
                  method=None,
                  evaluate=[None], 
                  opt_feature={}, 
                  features=['torsions', 'positions', 'distances'],
                  TICA_lag=None,
                  cluster_lags=None,
                  VAMP_lags=None,
                  dim=-1,
                  overwrite=False,
                  tica_weights=False):
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
        self.features=features
        self.overwrite=overwrite
        self.inputs = inputs
        self.get_tica=False
        if TICA_lag != None:
            self.disc_lag = TICA_lag
            self.tica_weights = tica_weights
            self.get_tica=True
        
        self.set_systems()
        
        #Use optional feature not defined at __init__
        try:
            ft_names = self.regions
        except:
            if not isinstance(ft_names, dict):
                raise TypeError('Provide a dictionary of type {region : inputs}')
            ft_names = opt_feature  
 
        #Loop through features
        for ft_name, ft_inputs_ in ft_names.items():
            print(ft_name, ft_inputs_)

# =============================================================================
#             self.ft_name=ft_name 
            self.selections = ft_inputs_ #The resid selections from input in __init__
# =============================================================================

            #VAMPs
            if method == 'VAMP':
                self.set_specs(ft_name, None, set_mode='base')
                for ev in evaluate:
                    if ev == 'features':
                        self.VAMP2_features(VAMP_lags, dim)
                    elif ev == 'dimensions': 
                        self.VAMP2_dimensions(VAMP_lags)   
                    else:
                        print('No evaluation defined.')
            else:
                
                ft_inputs = inputs[ft_name] #reads from input in __init__ and maps to current input
                self.ft_inputs=ft_inputs #whatever inputs are
                for feature in self.features:
                    self.set_specs(ft_name, feature, set_mode='base')

                    print('Feature: ', feature)
                    
                    if self.get_tica:
                        self.discretized_data = self.TICA_calculation(TICA_lag, feature, dim=dim)
                    else:
                        self.discretized_data = []
                        self.ft_base_name = f'{ft_name}_{feature}_s{self.stride}'

                    if method == 'TICA':
                            pass
                        
                    elif method == 'Clustering':
                        self.cluster_calculation(lags=cluster_lags, TICA_lag=TICA_lag, method='kmeans')


                    elif method == 'bayesMSM':
                        if isinstance(self.ft_inputs, tuple) and len(self.ft_inputs) == 2:
                            (n_states, msm_lag) = inputs[ft_name]
                            self.set_specs(ft_name, feature, lag=1, n_state=0, set_mode='MSM')
                            self.bayesMSM_calculation(msm_lag)
                        else:
                            raise TypeError('Input values have to be a tuple of integers (number of states, lag)')
                    else:
                        raise ValueError('No method defined')








            




    def save_samples(self, file_name, samples, save_as='dcd'):
        
        pyemma.coordinates.save_traj(self.trajectories, samples, outfile=f'{file_name}.dcd', top=self.topology)
        try:
            
            if save_as == 'dcd':
                md.load_frame(f'{file_name}.dcd', index=0, top=self.topology).save_pdb(f'{file_name}.pdb')
            elif save_as == 'pdb':
                traj = md.load(f'{file_name}.dcd', top=self.topology)
                for idx, i in enumerate(traj, 1):
                    print(f'\tSaving {file_name}_{idx}.{save_as}', end='\r')
                    i.save_pdb(f'{file_name}_{idx}.{save_as}')
    # =============================================================================
    #                 for i in range(MSM.generate_samples+1):
    #                     print(f'{file_name}_{i}.{save_as}')
    #                     md.load_frame(name, index=i, top=self.topology).save_pdb(f'{file_name}_{i}.{save_as}')    
    # =============================================================================
                
        except Exception as v:
            print(f'Warning! Could not save trajectory of {file_name}\n{v.__class__}: {v}')


    def extract_metastates(self, msm, macrostate=None, hmsm_lag=False, set_mode='highest'):

        def extract(dist, highest_member):
            n_frames = int(np.rint(dist*MSM.visual_samples))
            if n_frames == 0:
                n_frames +=1
            n_state_frames = int(np.rint(dist*MSM.ref_samples))
            if n_state_frames == 0:
                n_state_frames += 1

            if set_mode == 'visual':

                file_name=f'{self.stored}/{msm_file_name}_{self.CG}{idx+1}of{macrostate}_v'
                sample = msm.sample_by_state(n_frames, subset=[highest_member], replace = False)
                if not os.path.exists(f'{file_name}.pdb') or self.overwrite:
                    print(f'\tGenerating {n_frames} samples for macrostate {idx+1} using frames assigned to state {highest_member+1}')
                    self.save_samples(file_name, sample, save_as='dcd')
            elif set_mode == 'highest':
             
                file_name=f'{self.stored}/{msm_file_name}_{self.CG}{idx+1}of{macrostate}_{n_state_frames}frames'
                sample = msm.sample_by_state(n_state_frames, subset=[highest_member], replace = False)
                if not os.path.exists(f'{file_name}.dcd') or self.overwrite:
                    print(f'\tGenerating {n_state_frames} samples for macrostate {idx+1} using frames assigned to state {highest_member+1}')
                    self.save_samples(file_name, sample, save_as='dcd')
            state_samples = (f'{file_name}.dcd', f'{file_name}.pdb')
            
            return state_samples        


        msm_file_name=f'bayesMSM_{self.full_name}'
        
        state_samples= {} 
        #TODO: get total number of frames, to get proper n_samples

        h_members = []
        if self.CG == 'HMSM':
            print(hmsm_lag)
            hmsm=self.HMSM(msm, [macrostate], hmsm_lag, plot=False)
            for idx, (s, idist) in enumerate(zip(hmsm.metastable_sets, hmsm.metastable_distributions)):
                dist = msm.pi[s].sum()
                
                highest_member =idist.argmax()
                h_members.append(highest_member)
                state_samples[idx] = extract(dist, highest_member)

        else:
            msm.pcca(macrostate)

            if set_mode == 'highest' or set_mode == 'visual':
                for idx, idist in enumerate(msm.metastable_distributions, 0):
                    dist = msm.pi[msm.metastable_sets[idx]].sum()
                    highest_member = idist.argmax()
                    #TODO: resolve conflict of low assign states take frames from same micro. only problematic for lowe assig cases anyways for now.
                    if highest_member in h_members:
                        idist = np.delete(idist, highest_member)
                        highest_member = idist.argmax()
                    h_members.append(highest_member)
                    state_samples[idx] = extract(dist, highest_member)
            elif set_mode == 'generate':
                base_name = f'{msm_file_name}_{self.CG}{macrostate}'
                file_name = f'{self.project.def_input_struct}/{base_name}/sample'
                tools.Functions.fileHandler([f'{self.project.def_input_struct}/{base_name}'])
                
                frames= np.int(MSM.generate_samples/macrostate)
                samples = msm.sample_by_distributions(msm.metastable_distributions, frames)
                print(f'\tGenerating {frames}*{macrostate}macrostates = {MSM.generate_samples} samples from metastable distributions')
                self.save_samples(file_name, samples, save_as='pdb')
        
                
        
        #TODO: make single state extraction.
        
        return state_samples


    def visualize_metastable(self, samples, opt_index=1449, opt_index_end=1704):
        """
        

        Parameters
        ----------
        samples : TYPE
            DESCRIPTION.
        opt_index : TYPE, optional
            DESCRIPTION. The default is 1449.
        opt_index_end : TYPE, optional
            DESCRIPTION. The default is 1704.

        Returns
        -------
        widget : TYPE
            DESCRIPTION.

        """

        from matplotlib.colors import to_hex
    
        cmap = pl.cm.get_cmap('Set1', len(samples))
    
        widget = nglview.NGLWidget()
        widget.clear_representations()
        #TODO: make opt_index robust. query input topology for first residue value

        ref = md.load_pdb(self.project.input_topology)
        n_residues = ref.n_residues
        resid_range_ = [int(word) for word in self.selections.split() if word.isdigit()]
        if len(resid_range_) == 0:
            print('Could not assign inputs to resid ranges, reverting to "all".')
            resid_range_ = [1, n_residues]
        
        resid_range = '-'.join([str(resid+opt_index) for resid in resid_range_])
        prev_range_ = [opt_index, resid_range_[-1]+opt_index]
        prev_range = '-'.join([str(resid) for resid in prev_range_])
        post_range_ = [resid_range_[-1]+opt_index, opt_index_end]
        post_range = '-'.join([str(resid) for resid in post_range_])
        
        ref_struct = widget.add_trajectory(ref, default=False)
        ref_struct.add_cartoon(resid_range, color='black')
        ref_struct.add_cartoon(prev_range, color='black', opacity=0.3)
        ref_struct.add_spacefill('_ZN', color='black') #'246-248'
        ref_struct.add_licorice('SAM', color='black')

        x = np.linspace(0, 1, num=len(samples))
        for idx, s in samples.items():
            (traj_file, top) = s

            c = to_hex(cmap(x[idx-1]))
            print('\tGenerating images of (macro)state', idx, end='\r')
            traj = md.load(traj_file, top=top)
            traj.superpose(traj[0])
            

            component = widget.add_trajectory(traj, default=False)
            component.add_cartoon(resid_range, color=c)
            component.add_cartoon(prev_range, color=c, opacity=0.2)
            component.add_cartoon(post_range, color=c, opacity=0.2)
            component.add_spacefill('_ZN', color=c) #'246-248' #ZNB
            component.add_licorice('SAM', color=c) #SAM

        print('\n')
        widget.center()
        widget.camera = 'perspective'
        widget.stage.set_parameters(**{
                                 "clipNear": 0, "clipFar": 100, "clipDist": 1,
                                 "backgroundColor": "white",})   

        return widget


    
    def spectral_analysis(self, msm, lag, plot=False):
    #from pyEMMA notebook
        
        def its_separation_err(ts, ts_err):
            """
            Error propagation from ITS standard deviation to timescale separation.
            """
            return ts[:-1] / ts[1:] * np.sqrt(
                (ts_err[:-1] / ts[:-1])**2 + (ts_err[1:] / ts[1:])**2)


        dt_scalar=int(str(self.timestep*self.stride).split(' ')[0])
        
        timescales_mean = msm.sample_mean('timescales')
        timescales_std = msm.sample_std('timescales')



        #Filter number os states above 0.95 range of ITS
        state_cutoff=-1
        factor_cutoff =  (1 - MSM.msm_var_cutoff) * (max(timescales_mean)-min(timescales_mean))
        for idx, t in enumerate(timescales_mean):
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
                print(f'\t{len(imp_ts_factor)} processes resolved above {MSM.msm_var_cutoff*100}% ITS ({self.timestep*lag})')
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
                fig.suptitle(f'ITS decompositon of {self.full_name}', weight='bold', fontsize=12)
                fig.tight_layout()
    
                fig.savefig(f'{self.results}/SpectralITS_{self.full_name}.png', dpi=600, bbox_inches="tight")
                plt.show()
            
            print(f'\t{len(imp_ts_factor)} processes with {MSM.msm_var_cutoff*100}% ITS resolved above lag time ({self.timestep*lag})', end='\r')
            return len(imp_ts_factor)

        else:
            return 0

    
    @log
    def state_distance_comparison(self, file_names, distances=[]):
        

        df = pd.DataFrame()
        rows, columns=tools_plots.plot_layout(distances)
        
        for distance in distances:
            
            for idx, file_name in file_names.items():
                (traj_, top) = file_name
                traj= md.load(traj_, top=top)
                df_topo, _=traj.topology.to_dataframe() #bonds
                #TODO: This is set for single atom selection. make robust for multiple, if needed
                try:
                    ref, sel = int(traj.topology.select(distance[0])), int(traj.topology.select(distance[1]))
                    ref_id = ('').join(map(str, df_topo.loc[ref,[ 'resName', 'resSeq', 'name']].to_list()))
                    sel_id = ('').join(map(str, df_topo.loc[sel,['resName', 'resSeq', 'name']].to_list()))
                    dist_id = f'{ref_id}---{sel_id}'
                    atom_pairs=np.asarray([[ref,sel]])
                except Exception as v:
                    raise(v)
                dist=md.compute_distances(traj, atom_pairs=atom_pairs)*10 #MDTRAJ is reporting nanometers
        
                columns_index=pd.MultiIndex.from_product([[dist_id], [f'MS{idx+1}']], names=['distance', 'macrostates'])
                rows_index=pd.Index(np.arange(1, len(dist)+1), name='sample')
                df_macro = pd.DataFrame(dist, index=rows_index, columns=columns_index)
                df=pd.concat([df, df_macro], axis=1)

        kde=[]
        distances_ = df.columns.get_level_values(0).unique()
        for d in distances_:
            kde.append(df.loc[:, d].plot.kde(title=d, cmap='Category10', value_label=MSM.report_units))
        layout = hv.Layout(kde).cols(columns)
        print(layout)
        return layout

        
        



    @log
    def state_rmsd_comparison(self, file_names, pdb_files=[]):
        
        df = pd.DataFrame()
        #TODO: make init robust to initial residue of structure
        try:
            selection = self.convert_resid_residue(self.selections, init=1449)
        except IndexError as v:
            print(v)
            selection = self.selections
        colors= pl.cm.Set1(np.linspace(0,1,len(file_names)))
        if len(pdb_files) == 0:
            print('No pdb structures provided. Reverting to input topology')
            pdb_files = self.project.input_topology
        rows, columns=tools_plots.plot_layout(pdb_files)
        fig, axes = plt.subplots(rows, columns, sharey=True, sharex=True, constrained_layout=True, figsize=(12,8))
        try:
            flat = axes.flat
        except:
            flat=[axes]
        for pdb, ax in zip(pdb_files, flat):
            for idx, file_name in file_names.items():
                (traj, top) = file_name
                traj= md.load(traj, top=top)
                frames=traj.n_frames
                weight= frames / MSM.ref_samples

                indexes_sink=[['RMSD'], [idx], [pdb]]
                names=['name', 'macrostates', 'reference']
                sink_traj = md.load(f'{self.project.def_input_struct}/pdb/{pdb}.pdb')
                
                atom_indices=traj.topology.select(selection)
                sink_atom_indices=sink_traj.topology.select(selection)
                common_atom_indices=list(set(atom_indices).intersection(sink_atom_indices))

                
                ax.set_title(pdb)

                try:                    
                    rmsd_sink=in_units_of(md.rmsd(traj,
                                     sink_traj,
                                     atom_indices =atom_indices,
                                     ref_atom_indices = sink_atom_indices,
                                     frame=0,
                                     precentered=False,
                                     parallel=True), 'nanometers', MSM.report_units)
                except ValueError as v:
                    print(v)
                    print('Using common atom indices.\nWarning! Not reliable if structures have different n. of atoms. ')
                    try:
                        rmsd_sink=in_units_of(md.rmsd(traj,
                        sink_traj,
                        atom_indices = common_atom_indices,
                        frame=0,
                        precentered=False,
                        parallel=True), 'nanometers', MSM.report_units)
                    except Exception as v_:
                        print(v_)
                        pass
                except:
                    break

                ax.hist(rmsd_sink, 
                        label=f'MS {idx}',
                        density=True,
                        weights=[weight]*len(rmsd_sink), 
                        color=colors[idx-1], 
                        alpha=0.6,
                        histtype='barstacked',
                        stacked=True)
                rows=pd.Index(np.arange(1, len(rmsd_sink)+1), name='sample')
                df_macro = pd.DataFrame(rmsd_sink, index=rows, columns=pd.MultiIndex.from_product(indexes_sink, names=names))
                df=pd.concat([df, df_macro], axis=1)
        #[-1].legend()
        #axes[1].legend()
        handles, labels = flat[0].get_legend_handles_labels()
        def_locs=(1.1, 0.6)
        fig.legend(handles, labels, bbox_to_anchor=def_locs)
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.suptitle(f'RMSD: {MSM.ref_samples} samples\nModel: {self.full_name}')
        fig.text(0.5, -0.04, f'RMSD ({MSM.report_units})', ha='center', va='center', fontsize=12)
        fig.text(-0.01, 0.5, 'Probability density', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.tight_layout()
        plt.show()
        
        return df


    def HMSM(self, msm, macrostates, lags, plot=True):
        
        
        dt_scalar=int(str(self.timestep*self.stride).split(' ')[0])
        for macrostate in macrostates:
# =============================================================================
#             if plot:
#                 pyemma.plots.plot_implied_timescales(
#                     pyemma.msm.timescales_hmsm(disc_trajs[0::self.stride], 
#                                                macrostate, 
#                                                lags=[1, 2, 3, 4, 5], 
#                                                errors='bayes'), 
#                     units=self.unit, 
#                     dt=dt_scalar)
# =============================================================================


            for lag in lags:
                file_name=f'{self.stored}/HMSM_{self.full_name}_{macrostate}_hlag{lag}.npy'
                if not os.path.exists(file_name):
                    print(f'\tGenerating Hidden MSM for {self.full_name}: {macrostate} macrostates with lag = {lag*self.timestep}')
                    self.load_discretized_data(disc_lag = self.d_lag)
                    data_concat, disc_trajs, clusters = self.load_discTrajs()
                    hmsm = pyemma.msm.bayesian_hidden_markov_model(disc_trajs[0::self.stride], 
                                                               macrostate, 
                                                               lag, 
                                                               dt_traj=str(self.timestep), 
                                                               conf=0.95)
                    hmsm.save(file_name, overwrite=True)
                else:
                    print(f'Loading HMSM for {self.full_name} with {macrostate} macrostates and lag {lag*self.timestep}')
                    hmsm = pyemma.load(file_name)
                if plot:
                    self.load_discretized_data(disc_lag = self.d_lag)
                    data_concat, disc_trajs, clusters = self.load_discTrajs()
                    self.plot_CG(msm, macrostate, data_concat, disc_trajs, clusters, hmsm=hmsm)
                    cktest=hmsm.cktest()
                    pyemma.plots.plot_cktest(cktest,
                                             dt=dt_scalar, 
                                             layout='wide', 
                                             marker='.',  
                                             y01=True, 
                                             units=self.unit)
                return hmsm

    @log
    def PCCA_calculation(self, msm, macrostates, auto=False, hmsm=False, dims_plot=10):
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
        
        def plot():
            statdist = msm.stationary_distribution
            eigvec = msm.eigenvectors_right()
            eig_projection = eigvec[np.concatenate(disc_trajs), 1]
            if hmsm:       
                highest_membership = hmsm.metastable_distributions.argmax(1)
                coarse_state_centers = coarse_state_centers = clusters.clustercenters[hmsm.observable_set[highest_membership]]
                assignments=hmsm.metastable_assignments
                metastable_traj = hmsm.metastable_assignments[np.concatenate(disc_trajs)]
                file_name=f'HMSM_{self.full_name}_{macrostate}MS'
            else:
                highest_membership = msm.metastable_distributions.argmax(1)
                coarse_state_centers = clusters.clustercenters[msm.active_set[highest_membership]]
                metastable_traj = msm.metastable_assignments[np.concatenate(disc_trajs)]
                assignments=msm.metastable_assignments
                file_name = f'PCCA_{self.full_name}_{macrostate}MS'
        
            f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
            g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))
            
            
            fig=plt.figure(figsize=(12,12))
            gs = gridspec.GridSpec(nrows=4, ncols=3)
            colors= pl.cm.Set1(np.linspace(0,1,macrostate))
            fig.suptitle(f'MSM {self.full_name}\n {self.method} : {self.n_state} states to {macrostate} MS', 
                         ha='center',
                         weight='bold')
            
            plot_statdist_states = fig.add_subplot(gs[0, :2]) 
            plot_statdist_states.set_xlabel('State index')
            plot_statdist_states.set_ylabel('Stationary distribution')
            #plot_statdist_states.set_ylim(0,1)
            plot_statdist_states.yaxis.grid(color='grey', linestyle='dashed')
            plot_statdist_states.set_yscale('log')
            for x, stat in zip(range(1, len(statdist)+1), statdist):
                plot_statdist_states.bar(x, stat, color=colors[assignments[x-1]])
            
            plot_statdist_ms = fig.add_subplot(gs[:2,2]) #, sharey=plot_statdist_states)
            plot_statdist_ms.set_xlabel('Macrostate')
            plot_statdist_ms.set_ylabel('MS stationary distribution')
            plot_statdist_ms.yaxis.grid(color='grey', linestyle='dashed')
            plot_statdist_ms.set_xticks(range(1,macrostate+1))
            #plot_statdist_ms.legend() 
            #plot_statdist_ms.set_ylim(0,1)
            plot_statdist_ms.set_yscale('log')
            plot_assign_ms = fig.add_subplot(gs[1,:2], sharex=plot_statdist_states, sharey=plot_statdist_states) 
            plot_assign_ms.set_xlabel('State index')
            plot_assign_ms.set_ylabel('MS assignment')
            plot_assign_ms.yaxis.grid(color='grey', linestyle='dashed')
            
            plot_ms_trajs = fig.add_subplot(gs[2,:])
            if hmsm:
                for idx, s in enumerate(hmsm.metastable_sets):
                    statdist_m = msm.pi[s].sum()
                    plot_statdist_ms.bar(idx+1, statdist_m, color=colors[idx]) #, label=f'MS {idx+1}')
                for idx, met_assign in enumerate(hmsm.metastable_distributions):
                    plot_assign_ms.bar(range(1, len(statdist)+1), met_assign, color=colors[idx], label=f'MS {idx+1}')
        
        
        # =============================================================================
        #             ms_state_trajs_concat = [[] for i in range(macrostate)]
        #             for n, ms_state_trajs in enumerate(hmsm.hidden_state_probabilities):
        #                 for idx, ms_state_traj in enumerate(ms_state_trajs.T):
        #                     ms_state_trajs_concat[idx].append(ms_state_traj)
        #                 
        #             for idx, ms_traj in enumerate(ms_state_trajs_concat):
        #                 flatten = [item for sublist in ms_traj for item in sublist]
        #                 ax2 = plot_ms_trajs.twinx()
        #                 ax2.plot(range(len(flatten)), flatten, color=colors[idx], label=f'P(MS {idx+1})')
        # =============================================================================
        
            else:
                for idx, (s, met_assign) in enumerate(zip(msm.metastable_sets, msm.metastable_distributions)):
                    statdist_m = msm.pi[s].sum()
                    plot_statdist_ms.bar(idx+1, statdist_m, color=colors[idx]) #, label=f'MS {idx+1}')
                    plot_assign_ms.bar(range(1, len(statdist)+1), met_assign, color=colors[idx], label=f'MS {idx+1}')
            
            plot_assign_ms.legend()
            plot_assign_ms.set_yscale('log')
        
            #plot_ms_trajs.plot(range(1,len(metastable_traj)+1), metastable_traj+1, lw=0.5, color='black')
            plot_ms_trajs.step(range(1,len(metastable_traj)+1), metastable_traj+1, color='black')
        
            plot_ms_trajs.set_yticks(range(1, macrostate+1))
            plot_ms_trajs.set_ylabel('Macrostate')
            plot_ms_trajs.set_xlabel('Discretized frames')
            plot_ms_trajs.xaxis.set_major_formatter(mtick.FuncFormatter(g))
            plot_ms_trajs.set_xlim(0, len(metastable_traj+1))
            
        
        # =============================================================================
        #         plot_tica = fig.add_subplot(gs[3, 0])
        #         pyemma.plots.plot_free_energy(*self.tica_concat[:, :2].T, ax=plot_tica) #, legacy=False)
        #         plot_tica.set_xlabel('IC 1')
        #         plot_tica.set_ylabel('IC 2')
        #         plot_tica.set_title('TICA')
        # =============================================================================
            
            plot_msm_tica = fig.add_subplot(gs[3, 0])
            pyemma.plots.plot_free_energy(*data_concat[:, :2].T, weights=np.concatenate(msm.trajectory_weights()), ax=plot_msm_tica)
            plot_msm_tica.set_xlabel('IC 1')
            plot_msm_tica.set_ylabel('IC 2')
            plot_msm_tica.set_title('MSM')
            
            plot_eigen = fig.add_subplot(gs[3, 1])
        
            pyemma.plots.plot_contour(*data_concat[:, :2].T, 
                                      eig_projection, 
                                      ax=plot_eigen, 
                                      cbar_label='2nd right eigenvector', 
                                      mask=True, 
                                      cmap='PiYG') 
            #another way to plot them all without colors
            #plot_eigen.scatter(*coarse_state_centers[:, :2].T, s=15, c='C1')
            plot_eigen.set_xlabel('IC 1')
            plot_eigen.set_ylabel('IC 2')
            plot_eigen.set_title('Slowest Process')
        
            for center, color in zip(coarse_state_centers, colors):
                plot_eigen.scatter(*clusters.clustercenters[:, :2].T, edgecolors='black', c='orange')
        
            plot_ms_tica = fig.add_subplot(gs[3, 2])
            _, _, misc = pyemma.plots.plot_state_map(*data_concat[:, :2].T, 
                                                     metastable_traj, 
                                                     ax=plot_ms_tica, 
                                                     cmap='Set1',
                                                     alpha=0.9)
            plot_ms_tica.set_xlabel('IC 1')
            plot_ms_tica.set_ylabel('IC 2')
            plot_ms_tica.set_title('MS Projection')
            
            for idx, (center, color) in enumerate(zip(coarse_state_centers, colors)):
                plot_ms_tica.scatter(x=center[0], y=center[1], edgecolors='black', color=color, alpha=0.6)
            misc['cbar'].set_ticklabels([i for i in range(1,macrostate+1)])
            misc['cbar'].set_label('Macrostate')
            fig.tight_layout()
            plt.show()
            try:
                fig.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
            except OverflowError:
                print('Warning! Figure is getting too complex. Reducing dpi to 300.')
                fig.savefig(f'{self.results}/{file_name}.png', dpi=300, bbox_inches="tight")
        
        
        pyemma.config.mute = True
        if macrostates == None or auto:
            print('Retrieving number of macrostates from spectral analysis')
            macrostates = [self.spectral_analysis(self.full_name, msm, self.lag, plot=False)]
        
        data_concat, disc_trajs, clusters = self.load_discTrajs()
        #loop through sets of macrostates
        for macrostate in macrostates:
            if macrostate != 0:
                print(f'\tPCCA with {macrostate} MS')
                msm.pcca(macrostate)
                plot()
            else:
                print('\tPCCA skipped')
    
    def TICA_calculation(self,
                         lags,
                         feature,
                         dim=-1,
                         opt_dim=5):
        """
        

        Parameters
        ----------
        lag : TYPE
            DESCRIPTION.
        feature : TYPE
            DESCRIPTION.
        dim : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        tica : TYPE
            DESCRIPTION.

        """

        def plot():
            f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
            g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))
            
            if tica.dimension() > 10 and dim == -1:
                print(f'\tWarning: TICA for {MSM.var_cutoff*100}% variance cutoff yields {tica.dimension()} dimensions.')
                print(f'Reducing to {opt_dim} dimensions for visualization only.')
                dims_plot=opt_dim
            elif tica.dimension() > dim and dim > 0:
                dims_plot=tica.dimension()
            else:
                dims_plot=opt_dim
            colors= pl.cm.Accent(np.linspace(0,1,dims_plot))
            tica_concat = np.concatenate(tica.get_output())
            
            fig=plt.figure(figsize=(8,6))
            gs = gridspec.GridSpec(nrows=1, ncols=2)
                
            #plot histogram
            ax0 = fig.add_subplot(gs[0, 0])
            pyemma.plots.plot_feature_histograms(tica_concat[:, :dims_plot], ax=ax0)
            ax0.set_ylabel('IC') #, ylog=True)
            ax0.set_title('Histogram')
                
            #plot projection along main components
            ax1 = fig.add_subplot(gs[0, 1])
            pyemma.plots.plot_free_energy(*tica_concat[:, :2].T, ax=ax1, legacy=False) #, logscale=True)
            ax1.set_xlabel('IC 1')
            ax1.set_ylabel('IC 2')
            ax1.set_title('IC density')
    
            fig.suptitle(fr'TICA: {self.ft_name} {feature} @ $\tau$ ={self.timestep*lag}, {tica.dimension()} dimensions (w. {self.tica_weights})', ha='center', weight='bold')
            fig.tight_layout()   
            fig.savefig(f'{self.results}/TICA_{self.ft_base_name}.png', dpi=600, bbox_inches="tight")
            plt.show()
    
            #plot discretized trajectories
            ticas=tica.get_output()
            
            rows, columns=tools_plots.plot_layout(len(ticas)+1)
            fig_trajs,axes_=plt.subplots(columns, rows, sharex=True, sharey=True, constrained_layout=True, figsize=(12,12))
            #fig_trajs.subplots_adjust(hspace=0, wspace=0)
    
            try:
                flat=axes_.flat
            except AttributeError:
                flat=[axes_]     
    
            #loop for the trajectories
            for (idx_t, trj_tica), ax in zip(enumerate(ticas), flat): #len trajs?
                
                
                x = self.timestep * np.arange(trj_tica.shape[0])
        
                for idx, tic in enumerate(trj_tica.T):
                    
                    if idx < dims_plot:
                        ax.plot(x, tic, label=f'IC {idx}', color=colors[idx])
                        ax.text(.5,.85, idx_t+1, horizontalalignment='center', transform=ax.transAxes)
                        ax.xaxis.set_major_formatter(mtick.FuncFormatter(g))
            fig_trajs.suptitle(fr'TICA: {self.ft_name} {feature} @ $\tau$ ={self.timestep*lag}', weight='bold')
            handles, labels = ax.get_legend_handles_labels()
            def_locs=(1.1, 0.6)
            fig_trajs.legend(handles, labels, bbox_to_anchor=def_locs)
            fig_trajs.tight_layout()
            fig_trajs.subplots_adjust(hspace=0, wspace=0)
            fig_trajs.text(0.5, 0, f'Trajectory time ({self.unit})', ha='center', va='center', fontsize=12)
            fig_trajs.text(-0.03, 0.6, 'IC value', ha='center', va='center', rotation='vertical', fontsize=12)
            fig_trajs.savefig(f'{self.results}/TICA_{self.ft_base_name}_discretized.png', dpi=600, bbox_inches="tight")
            plt.show()


        lags = self.check_inputs(lags)

        for lag in lags:

            file_name = f'{self.stored}/TICA_{self.ft_base_name}.npy'
                
            if not os.path.exists(file_name) or self.overwrite:
                print('Generating TICA of ', self.ft_base_name)
                try: 
                    inputs_ = self.selections
                except AttributeError:
                    print('No selections found. Getting input information from MSM.regions')
                    inputs_ = self.regions[self.ft_name]

                data= self.load_features([feature], 
                                         inputs=inputs_)[0] #because its yielding a list.
                
                #for koopman, some process on tica causes nan and infs to appear which are not in the data
                try:
                    tica = pyemma.coordinates.tica(data, 
                                                       lag=lag, 
                                                       var_cutoff=MSM.var_cutoff,
                                                       weights=self.tica_weights,
                                                       skip=self.skip, 
                                                       stride=self.stride)
                    tica.save(file_name, save_streaming_chain=True)
                    plot()
                except Exception as v:
                    print(v)
                    print(f'{v.__class__} Could not generate/load TICA data')
                    tica = None
                    
            else:
                print(f'\tFound TICA of {self.ft_base_name}')
                tica = pyemma.load(file_name)
            if tica != None:
                #print(tica.get_params())
                n_total_frames = tica.n_frames_total()
                n_trajs = tica.number_of_trajectories()
                min_traj, max_traj = tica.trajectory_lengths().min(), tica.trajectory_lengths().max()
                print(f'\tTotal number of frames: {n_total_frames}\n\tNumber of trajectories: {n_trajs}\n\tFrames/trajectory: [{min_traj}, {max_traj}]')     
        
        if len(lags) == 1: # update only if a single TICA was requested
            return tica
    
    
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
                                      inputs=self.selections,
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
                                      inputs=self.selections)[0] #[0] because its yielding a list.
            
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
        
    @log
    def load_features(self, 
                      features,
                      inputs=None,
                      get_dims=False):
        """
        

        Parameters
        ----------
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
        for f in features:
            feat=pyemma.coordinates.featurizer(self.topology)
            if type(inputs) is tuple:
                
                input_1 = feat.select(inputs[0])
                input_2 = feat.select(inputs[1])
                

                if f == 'contacts':
                    feat.add_contacts(indices=input_1, indices2=input_2)
                else:
                    for s1 in input_1:
                        for s2 in input_2:
                            #pairs_sel=np.concatenate((sel1, sel2))
                            res_pairs=feat.pairs([s1,s2], excluded_neighbors=0)

                            if f == 'distances':
                                feat.add_distances(indices=res_pairs)
                            #feat.add_distances(indices=input_1, indices2=input_2)
    
                            elif f == 'min_dist':
                                feat.add_residue_mindist(residue_pairs=res_pairs, 
                                                 scheme='closest-heavy', 
                                                 ignore_nonprotein=False, 
                                                 threshold=0.3)
                            
            else:
                try:
                    if f == 'torsions':
                        feat.add_backbone_torsions(selstr=inputs, cossin=True)       
                    elif f == 'positions':
                        feat.add_selection(feat.select(inputs))
                    elif f == 'chi':
                        for idx in inputs:
                            feat.add_chi1_torsions(selstr=f'resid {idx}', cossin=True)
                except Exception as v:
                    print(v)
                    raise v.__class__('Could not load featurize data ', f)
                    

            print(f'\tFrom: {feat.describe()[0]}\n\tTo: {feat.describe()[-1]}')

            try:
                dimensions=feat.dimension()
                feature_file = f'{self.stored}/featurized-{f}_{self.ft_name}_stride{self.stride}_dim{dimensions}.npy'
                if not os.path.exists(feature_file):
                    data = pyemma.coordinates.load(self.trajectories, features=feat, stride=self.stride)
                    print(f'Parsing {f} data with stride {self.stride} ({feat.dimension()} dimensions)')
                    ft_list=open(feature_file, 'wb')
                    pickle.dump(data, ft_list)
                else:
                    print(f"Loading featurized {f} file: {feature_file}")
                    ft_list=open(feature_file, 'rb')
                    data=pickle.load(ft_list)  
                #Check if has NaN or infs
                #data=np.ma.masked_array(np.asarray(data, dtype=float), ~np.isfinite(np.asarray(data, dtype=float))).filled(0)                               
                if not get_dims:
                    features_list.append(data)           
                else:
                    features_list.append([data, feat.dimension()])
                

            except Exception as v:
                print(v)
                raise v.__class__('Could not load featurize data ', f)

    
        #TODO: return number of dimensions for registry operations

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


    def clustering(self, n_clusters, metric='euclidean', cluster_method=None):
        
        if cluster_method == None:
            cluster_method=self.cluster_method

        pyemma.config.show_progress_bars = False
        if cluster_method == 'regspace':
            cluster_name=f'{self.stored}/Cluster_{self.ft_base_name}_{n_clusters}clusters_{cluster_method}.npy'
        else:
            cluster_name=f'{self.stored}/Cluster_{self.ft_base_name}_{n_clusters}clusters.npy'
        if not os.path.exists(cluster_name) or self.overwrite:
            print(f'\tClustering for {n_clusters} cluster centres using {cluster_method}.')
            if cluster_method == 'k-means':
                clusters=pyemma.coordinates.cluster_kmeans(self.discretized_data, 
                                                       max_iter=50, 
                                                       k=n_clusters, 
                                                       stride=MSM.c_stride, 
                                                       chunksize=self.chunksize)
            elif cluster_method == 'regspace':
                clusters=pyemma.coordinates.cluster_regspace(self.discretized_data, 
                                                             dmin=2.0, 
                                                             max_centers=n_clusters, 
                                                             stride=self.stride, 
                                                             metric=metric,
                                                             chunksize=self.chunksize, 
                                                             skip=self.skip)
            clusters.save(cluster_name, save_streaming_chain=True)
        else:
            print(f'\tCluster found for {n_clusters} cluster centers', end='\r')
            clusters=pyemma.load(cluster_name)
        
        return clusters

    #from pyEMMA notebook
    def cluster_calculation(self, 
                            lags=[],
                            TICA_lag=None,
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
            name = f'{self.feature}@{lag*self.timestep}'
            print(name)
            
            data = self.discretized_data
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
                fig_.suptitle(f'{self.ft_base_name} : {k} centres')
                fig_.text(0.5, -0.02, 'IC 1', ha='center', va='center')
                fig_.text(-0.02, 0.5, 'IC 2', ha='center', va='center', rotation='vertical')
                fig_.tight_layout()                        

                fig_.savefig(f'{self.results}/ClusterTICA_{self.ft_base_name}_{method}_Elag{self.timestep*lag}_{k}centres.png', 
                             dpi=600, 
                             bbox_inches="tight")
            
            lower, upper=pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
            ax.plot(n_centres, np.mean(scores, axis=1), '-o', label=self.feature)
            ax.fill_between(n_centres, lower, upper, alpha=0.3)
            ax.semilogx()
            ax.set_xlabel('Number of cluster centres')
            ax.set_ylabel('VAMP2 score')

            optk=dict(zip(n_centres, np.mean(scores, axis=1)))
            #print(f'\tVAMP2 scores for feature {feature}', optk)
            optk_features[f'{self.feature}-{lag}']=optk
            print('\n')
        ax.legend()
        ax.set_title(r'$\tau$ = '+f'{self.timestep*lag}')
        
        fig.suptitle(f'Clustering (TICA) for {self.ft_base_name}\nMethod: {method}', weight='bold', ha='center')
        fig.tight_layout()
        fig.savefig(f'{self.results}/ClusterTICA_{self.ft_base_name}_Elag{self.timestep*lag}.png', dpi=600, bbox_inches="tight")

                
        plt.show()        
        
        self.clusters=optk_features
        
        return optk_features


    @staticmethod
    def check_inputs(input_):
        
        if isinstance(input_, int):
            input_ =[input_]
        elif isinstance(input_, tuple) and len(input_) == 2:
            input_ =range(input_[0], input_[1]+1)
        elif isinstance(input_, list):
            pass
        else:
            raise TypeError(input_+'should be either a single integer, a tuple (min, max) or a list of integers')
        
        return input_

    @staticmethod
    def mfpt_filter(mfpt, error=0):
        """Function to filter out MFPT values whose standard deviations are above *error* value.
            Default value of *error* is 0"""
            
        #mfpt=mfpt_df.loc[(scheme, feature), (parameter)]
        mfpt.dropna(how='all', inplace=True)
        mfpt.dropna(how='all', axis=1, inplace=True)
            
        idx = pd.IndexSlice  
        means=mfpt.loc[:, idx[:, 'mean']]
        stdevs=mfpt.loc[:, idx[:, 'stdev']]
        
        ratios = pd.DataFrame()
        if error != 0:
            print('Before: ', means.isna().sum().sum())
            counts=0
            ####Filter the mean values to the error percentage of stdev
            for mean, stdev in zip(means, stdevs):
                ratio = mfpt[stdev] / mfpt[mean]
                #print(mfpt[mean], mfpt[stdev], ratio)
                for r, v in zip(ratio, means[mean]):
                    if r >= error:
                        index_means=means.index[means[mean] == v]
                        means.loc[index_means, mean] =np.nan
                        counts+=1
                            
            #means.dropna(how='all', inplace=True)
            #means.dropna(how='all', axis=1, inplace=True)
                

            
        #minimum_notzero, maximum= means[means.gt(0)].min().min(), means.max().max()
        
        return means


    @staticmethod
    def convert_resid_residue(selection, init=0):
        
        selection_ = selection.split()
        resid_range = [int(word) for word in selection.split() if word.isdigit()]
        selection_new = f'residue {resid_range[0]+init} to {resid_range[-1]+init} {" ".join(selection_[-2:])}'
        
        return selection_new


            
            
# =============================================================================
#   def dep_plot_flux():
#             fig, axes = plt.subplots(1,1, figsize=(10, 8))
#             pyemma.plots.plot_flux(flux,
#                                    flux_scale = in_units_of(1, f'second/{self.timestep.unit.get_name()}', 'second/second'),
#                                    pos=None, 
#                                    state_sizes = flux.stationary_distribution,
#                                    show_committor=True,
#                                    state_labels = label_names,
#                                    show_frame=False,
#                                    state_colors = colors,
#                                    arrow_curvature = 2,
#                                    state_scale = 0.5,
#                                    figpadding=0.1,
#                                    max_height=6,
#                                    max_width=8,
#                                    arrow_label_format='%.1e  / s',
#                                    ax=axes)
#             axes.set_title(f'{self.mol} {self.it} : {n_state}@{lag_}' +
#                            f'\n{" ".join(label_A)} -> {" ".join(label_B)}' + 
#                            f'\nRate: {report_rate} Net Flux: {report_flux}', 
#                       ha='center',
#                       fontsize=12)
#             axes.axvline(0, linestyle='dashed', color='black')
#             axes.axvline(1, linestyle='dashed', color='black')
#             axes.grid(True, axis='x', linestyle='dashed')
#             axes.set_xlim(-0.15, 1.15)
#             axes.set_xticks(np.arange(0,1.1,0.1))
#             fig.savefig(f'{flux_name}.png')
# =============================================================================


