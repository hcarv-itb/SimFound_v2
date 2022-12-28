# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:11:07 2020

@author: hca
"""
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import pyemma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as co
import matplotlib.pylab as pl
from matplotlib.ticker import FormatStrFormatter, FixedLocator, AutoLocator, PercentFormatter, MaxNLocator, ScalarFormatter, LogLocator, LinearLocator, AutoMinorLocator
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from functools import wraps
import Discretize
import tools
import re

import collections


      

            
            
class Plots:
    
    

                    
 
    
    def __init__(self,
                 supra_project : dict,
                    combinatorial : pd.DataFrame,
                    plot_specs :dict,
                    input_states : dict,
                    subset_states : dict,
                    labels_regions = ['A', 'T', 'E', 'S', 'B'],
                    subset_projects = None,
                    mol_water={} ,
                    color2='darkorange',
                    mol2={'BuOH' : 'ViAc', 'BeOH' : 'BeAc'},
                    metric_data={},
                    discretized_trajectories={},
                    correlation_data={},
                    metrics= False,
                    metric_method='Mutual Information (normalized)',
                    double_comb_fractions={},
                    double_comb_states={},
                    figure_format=(3,2),
                    figure_type = 'full_page',
                    ) -> dict:
        
        if subset_projects is not None:
            self.supra_project : dict = {k : v for k, v in supra_project.items() if k in subset_projects}
        else:
            self.supra_project = supra_project
 
        self.combinatorial : pd.DataFrame = combinatorial
        self.plot_specs : dict = plot_specs
        self.input_states : dict = input_states
        self.subset_states : dict = subset_states
        self.labels_regions : list = labels_regions
        self.mol2 : list = mol2
        self.color2 : str = color2 #pl.cm.Oranges(np.linspace(0.5,0.5,1))[0]
        self.mol_water : dict = mol_water
        self.metrics : bool = metrics
        self.metric_method : str = metric_method
        self.figure_format : dict = figure_format
        self.figure_type = figure_type
        
        if self.metrics:
            self.discretized_trajectories : dict = discretized_trajectories
            self.metric_data : dict = metric_data
            self.double_comb_fractions : dict = double_comb_fractions
            self.double_comb_states : dict = double_comb_states
            self.correlation_data : dict = correlation_data
            



    loc1, loc2 =0, 350
    loc1_w, loc2_w = 0, 250
    n_sizes = 10
    sizes = (1,100) #(1,100)
    sampling_fraction_bins = (-6,0)
    sampling_resolution = 6
    #Automate sampling resolution according to sampling fraction bins
    secax_offset = -0.3

    
    
    color_regions = ('aqua','lightgray','grey','thistle','magenta')
    mpl.rcParams.update(mpl.rcParamsDefault)

    dpi=600
    figure_types = {'big' : 11,
                    'full_page' : 7.48,
                   'one_and_half_page' : 5.51,
                   'one_colum' : 3.54 
                   }

    scale_factor = 1.5
    scalars_uniques = LogLocator(base=10.0,subs=(0.2, 0.5, 1))
    state_limits = [2,4,10,20,80]
    limit_dNAC = (2, 140)
    bulk_region_location = int((state_limits[-1] + limit_dNAC[1]) / 2)
    
    states_color_code = {'B' : [0,0,0,0,1], 
             'SB' : [0,0,0,1,1],
             'ESB' : [0,0,1,1,1],
             'TSB' : [0,1,0,1,1],
             'TESB' : [0,1,1,1,1],
             'ASB' : [1,0,0,1,1],
             'AESB' : [1,0,1,1,1],
             'ATSB' : [1,1,0,1,1],
             'ATES' : [1,1,1,1,0],
             'ATESB' : [1,1,1,1,1]
             }
    
    
    
    def set_figureDimensions(self):
        
        try:
            _fig_type = self.figure_type[self.project_type]
            unit = Plots.figure_types[_fig_type] #self.figure_type
            (width, height) = self.figure_format[self.project_type]
        except Exception:
            print('Setting to default', self.figure_format)
            (width, height) = self.figure_format
            unit = Plots.figure_types[self.figure_type]
            _fig_type = 'big'
            print('Setting to default', unit)
            
        if  width > height:
            orientation = 'horizontal'
        elif width == height:
            orientation = 'square'
        else:
            orientation = 'vertical'

   
        if orientation == 'horizontal':
            figsize = (unit, (unit*height/width))
           
        elif orientation == 'vertical':
            figsize = ((unit*width/height), unit)
        else:
            figsize = (unit, unit)
        
            
        print(f'Figure layout: { _fig_type} {orientation} {width}:{height} {figsize} ') 
        
        return figsize
    
    

    def setSamplingLabelSize(self, sampled_fraction):

        return self.sampling_fractions[np.digitize(sampled_fraction, bins=self.sampling_fraction_bins)-1]
        
        

    def setSamplingResolution(self):
        
        
        (min_bin, max_bin) = Plots.sampling_fraction_bins
        (min_size, max_size) = Plots.sizes
        resolution = Plots.sampling_resolution
        sampling_fraction_bins = [0]
        for i in np.logspace(min_bin, max_bin, num=resolution+1):    
            sampling_fraction_bins.append(i)
        
        sizes = [0]   
        for s in np.linspace(min_size, max_size, num=resolution+1):
            sizes.append(s)


        sampling_fractions = {idx : (k, v) for idx, (k, v) in enumerate(zip(sampling_fraction_bins, sizes))}
        
        self.sampling_fraction_bins = sampling_fraction_bins
        self.sampling_fractions = sampling_fractions

        #print(f'Sampling fraction resolution {resolution}: {sampling_fractions}')

    @staticmethod
    def set_ticks(ax):
        
        #ax.set_xticks([])
        ax.tick_params(axis='y', which='major')
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=4)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d")) #ScalarFormatter())
        #locmax = LogLocator(base=10.0,subs=(0.2, 0.4, 0.8, 1),numticks=12)
        locmax = FixedLocator(Plots.state_limits)
        ax.xaxis.set_major_locator(locmax)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d") ) 

    def plot_wrapper(self, 
                     dG_df : pd.DataFrame, 
                     comb_df : pd.DataFrame):
        
        idx = self.idx_it
        list_its_ = comb_df.columns.get_level_values('l3').unique().to_list()
        
        if self.project_type == 'inhibition':
            list_its_.remove('5mM')

        it_df = comb_df.loc[:, comb_df.columns.get_level_values('l3') == list_its_[idx]]
             
        self.counts_states[idx] =  np.asarray([np.count_nonzero(it_df == i) for i in self.states]) / it_df.size
        
        #if self.project_type != 'inhibition':
        self.counts_regions[idx] = self.getStateSamplingFraction(it_df)

        self.plot_dg(dG_df)
        
        if self.project_type != 'water':
            self.plot_regions(it_df)
        
        #self.plot_state_scheme(mode='regions')
        


    def setSubplots(self):

        idx_mol = self.idx_mol
        if self.project_type == 'normal':      
            self.ax_comb = self.subfig_states['combinatorial']
            self.ax_scheme_combinatorial = self.subfig_states['scheme']
            #self.ax_image = self.subfig_images[idx_mol]
            self.ax_regions =self.subfig_regions[idx_mol]
            
        
        elif self.project_type == 'inhibition':
            self.ax_comb = self.subfig_states[idx_mol+1]
            self.ax_scheme_combinatorial = self.subfig_states[0]
            self.ax_regions =self.subfig_regions[idx_mol]
        

            
        #self.ax_regions =self.subfig_regions[idx_mol]
        self.ax_dG = self.subfig_dG[idx_mol]
        
        
               

    def setFigure(self, mode='end'):
        
        project_type = self.project_type
        
        if not mode == 'end':
            
            self.setSamplingResolution()
                       

            plt.rcParams['font.size'] = '8'
                        
            its = len(self.projects)
            fig=plt.figure(figsize=self.set_figureDimensions(), constrained_layout=True) 
            
            if project_type == 'normal':
                
                
                fig_dimer = plt.figure(figsize=self.set_figureDimensions(), constrained_layout=True)
                self.subfig_subunits_metrics = fig_dimer.subfigures(1,2, width_ratios=[3,2])
                self.subfig_subunits = self.subfig_subunits_metrics[0].subfigures(its,1)
                self.subfig_metrics =  self.subfig_subunits_metrics[1].subfigures(its,1)
                self.subfig_subunits_metrics[0].suptitle('Subunit region sampling fraction')
                self.subfig_subunits_metrics[1].suptitle('Dimer state interactions')
                self.fig_dimer = fig_dimer
                
                subfigs = fig.subfigures(1,2, width_ratios=[2,3])
                
                #self.subfig_images = subfigs[0].subplots(its,1)
                self.subfig_dG = subfigs[0].subplots(its, 1, sharex=True, sharey=True) #, gridspec_kw={'hspace' : 0.2})
                subfig_states = subfigs[1].subfigures(2,1, height_ratios=[1,2])        
                self.subfig_regions = subfig_states[0].subplots(1,its, sharex=True, sharey=True) #, gridspec_kw={'wspace' : 0.0})
                self.subfig_states = subfig_states[1].subplot_mosaic([['.', 'labelsizes'],
                                                                     ['scheme', 'combinatorial']],
                                                                     gridspec_kw={'height_ratios': [1,5],
                                                                                 'width_ratios': [1,5]}) 
                
                
            elif project_type == 'inhibition':
                
                subfigs = fig.subfigures(3,1, height_ratios=[5,4,5])
                self.subfig_dG = subfigs[0].subplots(1,its, sharex=True, sharey=True)
                self.subfig_regions = subfigs[1].subplots(1,its, sharex=True, sharey=True)
                self.subfig_states = subfigs[2].subplots(1, its+1, gridspec_kw = {'width_ratios' : [1,4,4]})
            
            elif project_type == 'water':
                
                subfigs = fig.subfigures(1,2, width_ratios=[2,3])
                subfig_1 = subfigs[1].subfigures(2,1, height_ratios=[2,4])
                subfig_regions_states = subfig_1[0].subplots(1,2) #, sharex=True)
                
                self.subfig_dG = subfigs[0].subplots(3, 1, sharex=True)
                self.subfig_regions = subfig_regions_states[0]
                self.subfig_states = subfig_regions_states[1]
                self.subfig_scheme_metrics = subfig_1[1].subfigures(2,1)
            
            self.fig = fig

        else:
            
            if project_type == 'normal':
                self.fig_dimer.savefig(f'{self.project.results}/figures/Dimers_{project_type}.png', 
                                    dpi=600, 
                                    bbox_inches='tight')
                

            fig = self.fig
            
            #fig.tight_layout()
            fig.show()
            
            self.figures[project_type] = fig
            print(f'saving {self.project.results}/figures/Thermodynamics_{project_type}.png')
            fig.savefig(f'{self.project.results}/figures/Thermodynamics_{project_type}.png', 
                             dpi=600, 
                             bbox_inches='tight')

        
            
    
            return self.figures

        
    
    def plot_labelsizes(self):
        legend_markers = []
        ax =  self.subfig_states['labelsizes']
        
        
        
        #for idx, (label, size) in enumerate(np.unique(np.asarray(self.sizes),axis=0)):
        for idx, label_size in self.sampling_fractions.items():
            
            (label, size) = label_size
            if label != .0:
                legend_markers.append(label)
                
                ax.scatter(label, 
                           0, 
                            marker='o', 
                            s=size, 
                            edgecolor='grey',
                            facecolor='none'
                            )
        
        ax.set_xscale('log')
        #ax.set_xticks(range(1, len(legend_markers)+1))
        #ax.set_xticklabels(legend_markers)
        ax.tick_params(axis='y',
                       left=False)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.tick_top()
        ax.spines["bottom"].set_visible(False)
        #ax.set_xlabel('Sampling fraction') #, rotation='horizontal', ha='right')
        #ax.set_xlim(1e-6, 10)
   
        
        
    def plot_dg(self, input_df):


        it = self.it 
        

        if self.mol == 'H2O':
            _mol = 'water'
        else:
            _mol = self.mol
            
        dg, dg_m, dg_p = self.getDeltaG(input_df, it, self.idx_it)        

        ax = self.ax_dG
        loc1, loc2 = self.loc1, self.loc2
        ranges = dg.iloc[loc1:loc2].index.values
        
        if it.split('_')[-1] == '5mM':
            it = f'{it.split("_")[0]} + {" ".join(it.split("_")[1:])}'

        if self.project_type == 'water' and self.mol != 'H2O':
            ranges = dg.iloc[loc1:loc2].index.values
            dg_diff = dg.iloc[loc1:loc2].values - self.dG_base_df.iloc[loc1:loc2].values

            if self.idx_it == len(self.iterables) -1:
                mcolor = 'none'
            else:
                mcolor = self.color_mol[self.idx_it]

            ax.plot(ranges, 
                    -dg_diff, 
                    marker = self.plot_specs['normal'][self.mol][3], 
                    markeredgecolor = self.color_mol[self.idx_it],
                    markerfacecolor = mcolor,
                    linestyle='none',
                    label=it)

            ax.set_ylabel(r'water $\Delta$$\Delta$G ($\it{k}$$_B$T)')
            ax.set_ylim(-1, 3)
            
        else:
            self.dG_base_df = dg
            ln = ax.scatter(ranges, 
                         dg.iloc[loc1:loc2], 
                         marker='.',
                         color=self.color_mol[self.idx_it], 
                         label=it,
                         ) #, marker=marker, fillstyle=fillstyle)
            ax.fill_between(ranges, 
                            dg_m.iloc[loc1:loc2], 
                            dg_p.iloc[loc1:loc2], 
                            alpha=0.5, 
                            color=self.color_mol[self.idx_it]
                            )
            ax.set_ylabel(r'$\Delta$G ($\it{k}$$_B$T)')

            self.lns.append(ln)
        
        ax.legend(loc='upper right', title=_mol)
        ax.set_xscale('log') 
        ax.axhline(y=0, ls='--', color='darkgray')
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        #if self.mol == 'H2O':
        #    ax.set_ylim(ax.get_ylim()[0], 7)
        

        self.set_ticks(ax)
        
        if self.mol == 'ViAc':
            ax.set_xlabel(r'$\itd$$_{NAC,donor}$ ($\AA$)')
        elif self.project_type == 'water':
           ax.set_xlabel(r'$\itd$$_{NAC,hydrolysis}$ ($\AA$)') 
        else:
            ax.set_xlabel(r'$\itd$$_{NAC,acceptor}$ ($\AA$)')
            
                
            
            
 
    #TODO: merge with plot_dg
    def plot_dg_inhib(self, input_df):
        
        ax = self.ax_dG
        lns = self.lns

        dg2, dg2_m, dg2_p = self.getDeltaG(input_df, self.it, self.idx_it)
        loc1, loc2 = self.loc1, self.loc2
        ranges = dg2.iloc[loc1:loc2].index.values
        ln2 = ax.plot(ranges, 
                      dg2.iloc[loc1:loc2],
                      '.', 
                      fillstyle='none', 
                      color=self.color2, 
                      label=f'5mM {self.mol2[self.mol]}')
        ax.fill_between(ranges, 
                        dg2_m.iloc[loc1:loc2], 
                        dg2_p.iloc[loc1:loc2], 
                        alpha=0.2, 
                        color=self.color2)
        ax.set_ylim(-4, self.ax_dG.get_ylim()[1])

        lns.append(ln2[0])
        labs = [p.get_label() for p in lns]
        
        self.ax_dG.legend(lns, labs, loc='upper right')
        self.set_ticks(ax)
        
        ax.set_xlabel(r'$\itd$$_{NAC,acceptor}$ $\equiv$ $\itd$$_5$ ($\AA$)')
        ax.set_title(self.mol)

        
        


    def plot_regions(self, df):
                
        ax = self.ax_regions
        #count_total = df.size
        state_sampling_fraction = self.getStateSamplingFraction(df)
        color= self.color_mol[0] #[self.idx_it]
        
        
        if self.it == '5mM':
            color = self.color2
        else:
            color = self.color_mol[self.idx_it]

        ax.barh(range(len(state_sampling_fraction)), 
                state_sampling_fraction,  
                linestyle=self.linestyle, 
                edgecolor=color,
                facecolor='none',
                label=self.it,
                )
        ax.set_yticks(range(len(self.labels_regions)))
        
        ax.set_xscale('log')
        ax.set_xticks([1e-3, 1e-2, 1e-1])
        #ax.invert_yaxis()
        if self.idx_mol == 0:
            ax.set_ylabel('Regions')
            ax.set_yticklabels(self.labels_regions)
        if self.idx_mol == 1 and self.project_type == 'normal':
            ax.set_xlabel('Sampling fraction')
        
        if self.project_type == 'inhibition':
            ax.set_xlabel('Sampling fraction')
        elif self.project_type == 'normal':
            ax.set_xticks(np.logspace(-4,0, num=5))

 
    
    def plot_combinatorial(self, df, mode='histogram', opt_color=None):
         
        if mode == 'histogram':
            self.plot_combinatorialHistogram(df)
        elif mode == 'scalar':
            self.plot_combinatorialScalar(df)
        elif mode == 'polar':
            #TODO has to shift subplots to 'polar' projection
            pass
        else:
            print('Mode not defined: ', mode)
        
    
    def plot_combinatorialScalar(self, df, opt_color=None):
        
        if isinstance(opt_color, np.ndarray):
            color = opt_color
        else:
            color = self.color_mol
        ax=self.ax_comb
        states = self.states
        
        
        _scalars = tools.Functions.setScalar(df.columns.tolist())
        scalars = [float(re.split('(\d+)',s)[1]) for s in _scalars]

        for idx, it in enumerate(df.columns):
            #discretized_state_sampling_fraction = np.digitize(df[it].values, bins=self.sampling_fraction_bins)
            for idx_y, sampling in enumerate(df[it]):
                (label, size) = self.setSamplingLabelSize(sampling)

                ax.scatter(scalars[idx], 
                           idx_y, 
                           s=size, 
                           facecolor='none',
                           edgecolor=color[idx],
                           alpha=1,
                           label=label,
                           )
        

        ax.set_xscale('log')
        ax.set_xticks([])
        ax.set_xlabel('Concentration (mM)')
        ax.xaxis.set_major_locator(Plots.scalars_uniques)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d") ) 
        ax.set_yticks(list(states.keys()))
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='major', left=False)
            
            
                        

            
    def plot_combinatorialHistogram(self, df, opt_color=None):
        
        if isinstance(opt_color, np.ndarray):
            color = opt_color
        else:
            color = self.color_mol
        ax=self.ax_comb
        states = self.states
        locmax = LogLocator(base=10.0, numticks=4)

        #color = self.color_mol
        if self.project_type != 'normal':
            df.plot.barh(ax=ax,
                          color=color,
                          label=False),
                          #width=Plots.p_type[self.project_type]['bar_width'])
            ax.grid(linestyle='--', axis='x')
            ax.set_xscale('log') 
            #ax.set_xticks([1e-5, 1e-2, 1])
            ax.set_yticks(list(states.keys()))
            
            ax.tick_params(axis='x', which='major', width=2, length=4)
            ax.xaxis.set_major_locator(locmax)
            

        else:
            df.plot.bar(ax=ax, 
                        color=color, 
                        label=False, 
                        width=Plots.p_type_specs[self.project_type]['bar_width']) 
            ax.grid(linestyle='--', axis='y')
            ax.set_yscale('log') 
            ax.set_xticks(list(states.keys()))
            #ax.yaxis.set_minor_locator(LogLocator())
            #ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
            ax.set_xticklabels(list(states.values()), rotation=90)
        
            if self.idx_mol == 1:
                ax.set_ylabel('Sampling fraction')
            if self.idx_mol != len(self.projects)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel('States')
                
            ax.tick_params(axis='y', which='major', width=2, length=4)
            ax.yaxis.set_major_locator(locmax)
            #ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))  

        ax.legend().remove() 
        
    @tools.log
    def plot_combinatorial_polar(self, df, mode='single', opt_color=None):
        
        ax = self.ax_comb
        labels = list(self.states.values())
        n_states = len(df.index)
        n_its = len(df.columns)
        angles = np.linspace(0, 2 * np.pi, n_states, endpoint=False) #.tolist()
        width = 2 * np.pi / n_states  
        
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.set_yticks([1e-6, 1e-4, 1e-3, 1e-1, 1])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        #ax.set_rlabel_position(288)

        
        
        if isinstance(opt_color, np.ndarray):
            color = opt_color
        else:
            color = self.color_mol

        if mode == 'single':
            
            ax.set_rorigin(-1e-6)
            ax.set_rmax(1)
            
            for idx, col in enumerate(df.columns):
                statdist = df[col].values
                label = df[col].name
    
                #_statdist = np.log10(statdist) - min10 + bullseye#slope * statdist + lowerLimit
    
            
                ax.bar(
                    x=angles, 
                    height=statdist, 
                    width=width,
                    linewidth=2,
                    bottom=1e-6,
                    edgecolor=color[idx],
                    facecolor='none')
        
        elif mode == 'ternary':
            #ax.set_rorigin(1)
            #ax.set_rmax(2)
            
            
            ax.bar(x=angles,
                   height=df[self.it].values,
                   width=width,
                   linewidth=2,
                   #bottom=1e-6,
                   edgecolor=color,
                   facecolor='none'
                   )
            
            #ax.set_rorigin(-1)
            
            
    def plot_combinatorial_metric(self,
                                  df,
                                  df2,
                                  metric_data, 
                                  correlation_data,
                                  ax = None):
        
        
        
        if ax is None:
            ax = self.ax_comb

        ax.set_xticks([0,1])
        ax.set_xticklabels([self.mol, self.mol2[self.mol]])
        ax.tick_params(axis='y', direction='in')
        ax.set_yticklabels([])

        #print(metric_data)
        #print(correlation_data)
        
        df = pd.concat([df, df2], axis=1)
        color = [i for i in self.color_mol]
        color.append(self.color2)
        #print(metric_data)
        max_mi = np.max(metric_data)
        

        vmin, vmax = -0.1, 0.1 #np.nanmin(correlation_data), np.nanmax(correlation_data)

        
        scalars = [0,0,1]
        
        #Draw sampling fractions
        for idx, it in enumerate(df.columns):
            discretized_state_sampling_fraction = np.digitize(df[it].values, bins=self.sampling_fraction_bins)
            for idx_y, sampling in enumerate(df[it]):
                (label, size) = self.setSamplingLabelSize(sampling)
                ax.scatter(scalars[idx], 
                            idx_y, 
                            s=size, 
                            facecolor='none',
                            edgecolor=color[idx],
                            alpha=1,
                            #label=label,
                            )
        
        #draw edges
        self.draw_edgesMetrics(ax, metric_data, correlation_data)
        
        
        
    @staticmethod
    def draw_edgesMetrics(ax, 
                          metric_data, 
                          correlation_data, 
                          vlim = (-0.1, 0.1), 
                          max_mi = None):
        

        cmap = plt.cm.get_cmap("PiYG")
        cmap.set_bad('white')
        
        if max_mi is None:
            max_mi = np.max(metric_data)

        (vmin, vmax) = vlim 
        print(np.nanmin(correlation_data), np.nanmax(correlation_data))
        norm = mpl.colors.SymLogNorm(2, vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        for idx, (mi, corr) in enumerate(zip(metric_data, correlation_data)):
            for idx_e, (m, c) in enumerate(zip(mi, corr)):
                mi_norm = m / max_mi
                    
                color=cmap(norm(correlation_data[idx, idx_e]))
                disc_linewidth = np.digitize(mi_norm, bins=np.logspace(-2,0, num=3))

                ax.plot((0,1),
                    (idx, idx_e),
                    alpha= 1, #min_norm, 
                    linewidth=disc_linewidth,
                    color = color
                    )
        
# =============================================================================
#         plt.colorbar(sm, ticks=(vmin,0, vmax), 
#                                 format=mpl.ticker.ScalarFormatter(), 
#                                 shrink=1.0, fraction=0.1, pad=0.1,
#                                 label='Correlation Coefficient')
# =============================================================================
        
        
    
    
    def plot_subunitsMetrics(self, 
                            input_df, 
                            mode = 'sum',
                            ref_map = {'A-B' : (0,1), 
                                       'C-H' : (2,7), 
                                       'E-F' : (4,5), 
                                       'D-G' : (3,6)}):
        
        idx_mol = self.idx_mol
        

        sup_ax_sub = self.subfig_subunits[idx_mol].subplots(1,len(self.iterables),sharey=True, sharex=True) 
        sup_ax_metric = self.subfig_metrics[idx_mol].subplots(1, len(self.iterables), sharey=True, sharex=True)

        
        metric_data = self.metric_data[self.project_type][self.mol].values()
        correlation_data = self.correlation_data[self.project_type][self.mol].values()
        #This is assuming that iterables and keys are the same.
        
        cmap = plt.cm.get_cmap("PiYG")
        cmap.set_bad('white')
        vmin, vmax = -0.3,0.3
        norm = mpl.colors.SymLogNorm(2, vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        for idx, (it, metrics, correlations) in enumerate(zip(self.iterables, metric_data, correlation_data)): 
            
            _it_matrix = np.asarray([m for m in metrics.values()])
            _it_corr = np.asarray([m for m in correlations.values()])
            if mode == 'sum':
                it_matrix = _it_matrix.sum(axis=0)
            elif mode == 'mean':
                it_matrix = _it_matrix.mean(axis=0)
            
            it_corr = _it_corr.mean(axis=0)
            self.plot_subunitsStateSampling(input_df, sup_ax_sub, it, idx, ref_map)
            ax_metric = sup_ax_metric[idx]
            
            self.draw_edgesMetrics(ax_metric, it_matrix, it_corr, vlim=(vmin, vmax))
            ax_metric.set_yticks(range(len(self.states)))
            ax_metric.set_yticklabels(self.states.values())
            ax_metric.set_title(it)
            sup_ax_sub[idx].set_title(it)
            
            


            
            ax_metric.set_xticklabels([])
            #ax_metric.imshow(it_matrix)
            #ax_metric = self.plot_dimerMetrics(sup_ax, idx)
            
        if self.idx_mol == 0:    
            plt.colorbar(sm, ticks=(vmin,0, vmax), 
                        format=mpl.ticker.ScalarFormatter(), 
                        shrink=1.0, fraction=0.1, pad=0.1,
                        label='Correlation Coefficient')    
            
            
    def plot_subunitsStateSampling(self, input_df, sup_ax, it, idx, ref_map):
        
        ax = sup_ax[idx]
        #ax.set_title(it)
        
        df = input_df.iloc[:, input_df.columns.get_level_values('l3') == it]
        total_sf = np.array(self.getStateSamplingFraction(df))
        references = df.columns.get_level_values('l5').unique()
        

        sampling_fraction ={}
        
        for ref_pair_label, indexes in ref_map.items():
            pair_sf = []
            for i in indexes:

                #label = chr(ord('@')+idx_r+1)
                ref = references[i]
                ref_df = df.loc[:, df.columns.get_level_values('l5') == ref]
                _sf = self.getStateSamplingFraction(ref_df)
                pair_sf.append(_sf)
            sampling_fraction[ref_pair_label] = pair_sf
        
        total_sf = np.asarray(list(sampling_fraction.values()))
        
        #number_pairs = np.shape(total_sf)[0]
        number_regions = np.shape(total_sf)[2]      
        
        for pair in total_sf:
            sampled_regions = total_sf.sum(axis=0).sum(axis=0)
            _pair = pair / sampled_regions

            for idx_r, region in enumerate(_pair.T):
                x = [idx_r, idx_r+0.5]
                ax.plot(x, 
                        region,
                        '.-',
                        color=self.color_mol[idx]
                        ) #, alpha=(1/number_pairs*2))

        ax.set_xticks(np.arange(0.25, number_regions+0.25))
        ax.set_xticklabels(self.labels_regions)
        ax.set_ylim(0,1)

        
        if idx == 0:
            ax.set_ylabel(self.mol)
            ax.set_yticks(np.arange(0.125,1.125,0.125))
            ax.set_yticklabels([f'{i}/{len(references)}' for i in range(1,len(references)+1)])

            
    def plot_dimerMetrics(self, sup_ax, idx, mode = 'sum'):
        #TODOTODOTODOTODOTO migrate from notebook to here. keep only at level of iterables. 
        labels = self.states.values()
        ticks = self.states.keys()
        cmap = plt.get_cmap('Greys')
        
        
        metric_data = self.metric_data[self.project_type][self.mol].values()
        correlation_data = self.correlation_data[self.project_type][self.mol].values()
        iterables = self.metric_data[self.project_type][self.mol].keys()
        
        for iterables, metrics, correlations in zip(iterables, metric_data, correlation_data):
            
            
            ax = sup_ax[0, idx]
            

# =============================================================================
#             _vmax=0
#             _data_it = []
#             iterable_scalars = []
#             df = pd.DataFrame()
#             ax_all = sup_ax[0, idx] #plt.subplot(grid[0,:])
#             for idx, (it, subunits) in enumerate(v.items()):
#                 _data = []
#                 _data_states = []
#                 _labels = []
#                  #plt.subplot(grid[1, idx])
#                 ax.set_xticklabels(labels, rotation=90)
#                 ax.set_yticklabels(labels)
#                 ax.set_yticks(ticks)
#                 ax.set_xticks(ticks)
#                 
#                 ax.set_title(it)
#                 
#                 if idx != 0:
#                     ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
#                 
# 
#                 bulk_value=float(str(it).split('mM')[0])
#                 iterable_scalars.append(bulk_value)
#                 
#                 sub_grid = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = grid[2,idx], hspace=0, wspace=0) 
#                 _vmax_pair = 0
#                 
#                 for idx_pair, (pair_label, matrix) in enumerate(subunits.items()): 
#                     
#                     ax_pair = plt.subplot(sub_grid[idx_pair], sharex=ax, sharey=ax)
#                     plot = ax_pair.imshow(matrix, cmap=cmap, aspect='auto') #, vmin=0, vmax=max(_vmax_pair, v3.max()))
#                     ax_pair.set_title(pair_label)
#                     ax_pair.invert_yaxis()
#                     if idx_pair != 2 or idx != 0:
#                         ax_pair.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
#                     data_pair = matrix.sum(axis=0)
#                     data_pair_1 = matrix.sum(axis=1)
#         
#                     sums = data_pair + data_pair_1
#                     
#                     _data.append(matrix) #[data_pair, data_pair_1]
#                     _labels.append(pair_label)
#         
#                 data = np.asarray(_data).mean(axis=0)
#                 _data_it.append(data)
# 
#                 
#                 
#                 plot = ax.imshow(data, cmap=cmap) 
#                 ax.invert_yaxis()
#         
#                 mean_sums = np.vstack([data.sum(axis=0),data.sum(axis=1)]).mean(axis=0)
#                 df_it = pd.DataFrame(mean_sums / mean_sums.max(), columns=[it])
#                 df = pd.concat([df, df_it], axis=1)
#         
#             df.plot.bar(ax=ax_all)
#         print(df)
#         fig.tight_layout()            
#         plt.show()
# =============================================================================
            

    



    def plot_label_regions(self):    

        ax = self.subfig_dG[self.idx_mol]       
        steps = self.plot_specs['labels']
        
        for x, color in zip(Plots.state_limits[1:], Plots.color_regions):
            ax.axvline(x, color=color)
        if self.idx_mol == 0:
            secax = ax.secondary_xaxis('top') #, color='darkgrey')
            
            secax.set_xticks(steps)
            secax.set_xticklabels(self.labels_regions)
            secax.minorticks_off()
            if self.project_type == 'inhibition':
                secax.tick_params(axis='x', which='major', pad= Plots.secax_offset)

    def plot_state_scheme(self, mode='combinatorial', opt_ax = None):
        
        #TODO: This is outsourcing functionality to draw_scheme, so that it can be used by other modules.
        #make even less dependent on self
        
        if opt_ax == None:
            ax = self.ax_scheme_combinatorial
        else:
            ax = opt_ax
        
        if self.project_type != 'water':

            states = self.states
            ax.set_ylim(self.ax_comb.get_ylim())
        else:

            states = self.states_mol
            #ax.set_ylim(0,9)
            

        self.draw_scheme(ax, states, mode = 'combinatorial')
        
    @staticmethod
    def draw_scheme(ax, states, offset= 0.8, mode = 'combinatorial'):
        
        region_limits = Plots.state_limits
        
        
        if mode == 'combinatorial':
            ax.set_xlim(Plots.limit_dNAC)
            ax.set_xscale('log')
            #ax.set_xticks(steps)
            #ax.set_xticklabels(self.labels_regions)
            ax.set_xlabel(r'$\itd$$_{NAC}$ ($\AA$)')
            ax.get_xaxis().set_visible(False)
            ax.set_ylabel('States')
            ax.set_yticks(range(len(states)))
            ax.set_yticklabels(states.values())
        for k, spine in ax.spines.items():
            spine.set_visible(False)
        
        for idx_combination, label in enumerate(states.values()):
            
            combination = Plots.states_color_code[label]
            for idx, sampled in enumerate(combination):
                   
                if mode == 'combinatorial':
                    anchor = (region_limits[idx], idx_combination-(offset/2))#(idx_combination,idx)
                    try:
                        width = region_limits[idx+1] - region_limits[idx]
                    except IndexError:
                        width = Plots.limit_dNAC[1] - region_limits[-1]
                    height = offset
                elif mode == 'single':
                    anchor = (region_limits[idx],0)
                    height = 1
                    try:
                        width = region_limits[idx+1] - region_limits[idx]
                    except IndexError:
                        width = Plots.limit_dNAC[1] - region_limits[-1]
                    
                else:
                    anchor = (idx_combination, region_limits[idx])#(idx_combination,idx)
                    width = offset

                    try:
                        height = region_limits[idx+1] - region_limits[idx]
                    except IndexError:
                        height = Plots.limit_dNAC[1] - region_limits[-1]
                
                if sampled:
                    
                    color = Plots.color_regions[idx]
                else:
                    color = 'white'
                
                    
                rect = plt.Rectangle(anchor, 
                                     width, 
                                     height, 
                                     facecolor=color, 
                                     edgecolor='black', 
                                     )
                ax.add_patch(rect)  
        
        return ax





                   

    def plot_thermodynamics(self, dG):
        self.figures = {}
                
        self.supra_counts_regions = collections.defaultdict(dict)

        for project_type, projects in self.supra_project.items():
        
                
            print(project_type)
            input_states = self.input_states
            
            combinatorial = self.combinatorial
            plot_specs = self.plot_specs
            if project_type != 'water':
                self.loc1, self.loc2 = Plots.loc1, Plots.loc2
            else:
                self.loc1, self.loc2 = Plots.loc1_w, Plots.loc2_w 
            mol2 = self.mol2
            #color2 = self.color2
        
            
            self.projects : dict = projects
            self.project_type : str = project_type
            self.labels_regions = list(self.subset_states[project_type].keys())
            
            self.setFigure(mode='set')
            
            
            for idx_mol, (mol, project) in enumerate(projects.items()):
                self.project = project

                if project_type != 'water':
                    dG_df = dG[project_type][mol]
                else:
                    base_df = dG[project_type][mol]
                    dG_df = base_df.iloc[:, base_df.columns.get_level_values(0) == mol].droplevel(0, axis=1)

                if project_type == 'inhibition':
                    base_states = self.double_comb_states['inhibition']
                else:
                    base_states = input_states[project_type]

                if isinstance(input_states[project_type], tuple):
                    idx_state = 1
                    self.states = base_states[idx_state]
                    combinatorial_df = combinatorial[project_type][mol].replace(to_replace=base_states[0].keys(), value=self.states.keys())
                else:
                    idx_state = 0
                    self.states = base_states[idx_state]
                    combinatorial_df = combinatorial[project_type][mol]
 

                self.iterables = project.parameter
                self.color_mol = plot_specs[project_type][mol][1]
                self.idx_mol = idx_mol
                self.mol = mol
                self.linestyle = plot_specs[project_type]['linestyle']
                self.lns = [] 
                

                #Call to set subplots
                self.setSubplots()

                self.counts_states = np.empty([len(self.iterables), len(self.states)])
                self.counts_regions = np.empty([len(self.iterables), len(self.labels_regions)])

                #will populate counts_states and counts_regions
                for idx, it in enumerate(self.iterables): 
                    self.it : str = it
                    self.idx_it = idx
                    self.it_scalar = tools.Functions.setScalar(it)
                    self.plot_wrapper(dG_df, combinatorial_df) 

    
                states_df = pd.DataFrame(self.counts_states.T, index=self.states, columns=self.iterables)
                regions_df = pd.DataFrame(self.counts_regions.T, index=self.labels_regions, columns=self.iterables)
                
                
                if project_type == 'normal':
                    self.plot_labelsizes()
                    self.plot_label_regions()
                    self.base_combinatorial = self.plot_combinatorial(states_df, mode='scalar')
                    self.plot_state_scheme(mode='combinatorial')
                    self.plot_subunitsMetrics(combinatorial_df)
                    
                elif project_type == 'inhibition':
                    
                    self.linestyle = 'dashed'
                    self.idx_it = 0
                    self.it = '5mM'
                    self.states_mol = self.double_comb_states['normal'][idx_state]
                    inib_mol = mol2[mol]

                    inib_df = pd.read_csv(f'{project.results}/binding_profile/binding_profile_acylSer-His_{inib_mol}_mean_b0_e-1_s1.csv', index_col=0)    
                    df_comb_inhib = combinatorial_df.loc[:, combinatorial_df.columns.get_level_values('l3') == self.it]

                    self.plot_dg_inhib(inib_df)
                    self.plot_regions(df_comb_inhib)
                    self.plot_label_regions()
                    
                    _sampling_fraction = self.getStateSamplingFraction(df_comb_inhib,  states=self.states_mol, w_subset=False)
                    df_mol_inhib = pd.DataFrame(_sampling_fraction,index=self.states_mol,  columns=[self.it])

                    self.plot_combinatorial_metric(states_df,df_mol_inhib,self.metric_data[project_type][mol], self.correlation_data[project_type][mol])
                    self.plot_state_scheme(mode='combinatorial')  


                if project_type == 'water':
    
                    self.ax_comb = self.subfig_states
                    self.ax_regions = self.subfig_regions
                    self.plot_label_regions()
                    
                    
                    for idx_mol, (mol_w, iterables_mol_w2) in enumerate(self.mol_water.items(),1):

                            
                        base_df_mol = dG[project_type]['H2O']
                        dG_mol_df = base_df_mol.iloc[:, base_df_mol.columns.get_level_values(0) == mol_w].droplevel(0, axis=1)
                        
                        
                        _color_mol = plot_specs['normal'][mol_w][1] 
                        self.color_mol = np.vstack([_color_mol, pl.cm.Oranges(np.linspace(0.5,0.5,1))])
                        self.mol = mol_w
                        self.idx_mol = idx_mol 
                        self.iterables = iterables_mol_w2[0] + [f'100mM_{iterables_mol_w2[1]}_5mM']                  
                        self.ax_dG = self.subfig_dG[idx_mol]

                        self.counts_states = np.empty([len(self.iterables), len(self.states)])
                        self.counts_regions = np.empty([len(self.iterables), len(self.labels_regions)])
                            
                        if isinstance(input_states[project_type], tuple):

                            _states_mol = input_states['normal'][1]
                            double_combinatorial_states = self.double_comb_states['normal'][1]
                            comb_df = combinatorial[project_type][mol].replace(to_replace=input_states[project_type][0].keys(), value=self.states.keys())
                        else:
                            _states_mol = input_states['normal'][0]
                            double_combinatorial_states = self.double_comb_states['normal'][0]
                            comb_df = combinatorial[project_type][mol]
                        comb_df = comb_df.iloc[:, comb_df.columns.get_level_values('l2') == mol_w]
    
                        #Remove 'ATES' states from self.states_mol
                        self.states_mol = {k : v for k, v in _states_mol.items() if (v != 'ATES')}
    
                        if self.metrics:
                            self.plot_waterMetrics(double_combinatorial_states)

                        for idx_w, it_w in enumerate(self.iterables): 
                            self.it = it_w
                            self.idx_it = idx_w    
                            self.plot_wrapper(dG_mol_df, comb_df)

                        self.plot_waterStatesRegions(states_df)
                        self.plot_waterStatesRegions(regions_df, mode='regions')
                        

                self.supra_counts_regions[project_type][mol] = self.counts_regions            
          
            self.setFigure(mode='end')


        return self.figures
    
    
    
    def plot_waterMetrics(self, double_combinatorial_states):
    
    
        subfig_metrics = self.subfig_scheme_metrics[self.idx_mol-1].subplots(1,3, 
                                                                             gridspec_kw = {'width_ratios' : [1,3,3]}, 
                                                                             sharey=True)
        
        
        self.ax_scheme_combinatorial = subfig_metrics[0]
        ax_metric = subfig_metrics[1]
        ax_corr = subfig_metrics[2]
        self.plot_state_scheme(mode='combinatorial')
        
        self.ax_scheme_combinatorial.set_ylabel(f'{self.mol} & {self.mol2[self.mol]} States')
        
        metric_data = self.metric_data[self.project_type][self.mol]
        corr_data = self.correlation_data[self.project_type][self.mol]
        combinatorial_state_fractions = self.double_comb_fractions[self.mol]
        
        #print(combinatorial_state_fractions)
        
        mi_maxima = []
        for mi_iterable in metric_data.values():
            mi_maxima.append(np.max(mi_iterable[0,:]))
            
        m_max = np.max(mi_maxima)
        
        for idx, (mi_iterable, corr_iterable) in enumerate(zip(metric_data.values(), corr_data.values())): 

            mi = mi_iterable / m_max
            corr = corr_iterable 
            states = range(len(self.states_mol))

            ax_metric.barh(states, 
                           mi[0,:],
                           facecolor= 'none',
                           edgecolor=self.color_mol[idx])
            

            ax_corr.barh(states,
                         corr[0,:],                           
                         facecolor= 'none',
                         edgecolor=self.color_mol[idx]
                         )
            ax_corr.axvline(0, linestyle='dashed', color='grey')
            ax_corr.set_xlim(-0.15, 0.25)
            if self.idx_mol == 2: 
                ax_metric.set_xlabel('Mutual information\n(normalized)')
                ax_corr.set_xlabel('Correlation coefficient')
            else:
                self.subfig_scheme_metrics[self.idx_mol-1].suptitle(r'ATESB$_{water}$')
                
                
    def plot_waterStatesRegions(self, input_df, mode = 'states'):
        
        try:
            normal_regions = self.supra_counts_regions['normal'][self.mol]
            inhibition_regions = self.supra_counts_regions['inhibition'][self.mol]
        except:
            pass
        
        if mode == 'states':
            size = self.counts_states
            ax = self.ax_comb
            index = list(self.states.keys())
        else:
            size = self.counts_regions
            ax = self.ax_regions
            index = self.labels_regions
        
        #TODO: rewrite messy handling of states_df
        data = np.vstack([input_df.values.T, size]).T
        
        df = pd.DataFrame(data, 
                        index=index, 
                        columns=['0 mM'] + self.iterables)
        
        
        iterable_scalars = [] #[0]
        #WARNING! this is assuming mol2 is the last one on the list. make auto
        for it in self.iterables[:-1]:
            try:
                bulk_value=float(str(it).split('M')[0]) 
                unit='M'
            except:
                bulk_value=float(str(it).split('mM')[0])
                unit='mM'
            iterable_scalars.append(bulk_value)
        
        
        fillstyle_mol = self.plot_specs['normal'][self.mol][2]
        marker_mol = self.plot_specs['normal'][self.mol][3]
            
        
        
        #Using only second state for plotting metrics, since there is only two in MsAcT.
        #TODO: Query on which state to plot
        if mode == 'states':
            ax.scatter(iterable_scalars, 
                    df.iloc[1,1:-1], 
                    marker=marker_mol,
                    label=self.mol,
                    color=self.color_mol[:-1]) #self.plot_specs['inhibition'][self.mol][1][1])
            
            ax.plot(100.0, 
                    df.iloc[1,-1],  
                    marker=marker_mol,
                    markeredgecolor=self.color2,
                    markerfacecolor='none')

            ax.set_title(f'{self.states[1]} sampling fraction')
            #ax.set_ylabel('Water')
            ax.set_xlabel(f'[Acyl acceptor] ({unit})')
            ax.axhline(df.iloc[1,0], color = 'red')
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(Plots.scalars_uniques)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d") )
            #ax.set_xticks(Plots.scalars_uniques)
            #print(ax.get_xticks())
            
        else:
            
            w_regions = df.loc['A'].values
            n_regions = normal_regions[:,0]

            
            ax.scatter(n_regions,
                       w_regions[1:-1], 
                       marker=marker_mol,
                       #label=self.mol,
                       color=self.color_mol[:-1]) #self.plot_specs['inhibition'][self.mol][1][1])
            
            ax.plot(inhibition_regions[1,0],
                       w_regions[-1],
                       marker=marker_mol,
                       markeredgecolor=self.color2,
                       markerfacecolor='none')
            
            it = self.iterables[-1]
            label_inhib = f'{it.split("_")[0]} + {" ".join(it.split("_")[1:])}'
            #ax.annotate(label_inhib, (inhibition_regions[1,0],w_regions[-1]))

            #for idx, label in enumerate(self.iterables[:-1]):
            #    ax.annotate(label, (n_regions[idx], w_regions[1:-1][idx]))
            
            ax.set_xlabel('Acyl acceptor')
            ax.set_ylabel('Water')
            ax.set_title('Region A sampling fraction')

            ax.set_xscale('log')

           
    def getDeltaG(self, input_df, it, idx):
        

        if self.project_type == 'inhibition':
            #print(input_df)
            if it == '100mM_{self.mol2[self.mol]}_5mM' or it == self.mol2[self.mol]:
                it = '5mM'
                
                
        dg = input_df[f'$\Delta$G {it}']
        if not isinstance(dg, pd.Series):
            dg = dg.iloc[:, idx]
            dg_index = [i for i, x in enumerate(input_df.columns.get_loc(dg.name)) if x][idx]
        else:
            dg_index = input_df.columns.get_loc(dg.name)
        dg_m = input_df.iloc[:, dg_index+1]
        dg_p = input_df.iloc[:, dg_index+2]
        
        return dg, dg_m, dg_p



                           
                
                
    def getStateSamplingFraction(self, df, states=None, w_subset=True) -> list:
        
        if states == None:
            states = self.states
        
        count_total = df.size
        label_states = []
        state_sampling_fraction = []
        if w_subset:
            for idx_regions, (label_state, subset) in enumerate(self.subset_states[self.project_type].items()):
                state_sampling_fraction.append(np.asarray([np.count_nonzero(df == i) / count_total for i in states if i in subset]).sum())
                label_states.append(label_state)
        else:
            state_sampling_fraction = np.asarray([np.count_nonzero(df == i) / count_total for i in states])
        
        
        #self.labels_regions = label_states
        
        return state_sampling_fraction



    
    def deprecated_plot_StateSampling_vs_scalars(self, df, ax):        
        if self.project_type == 'water':
            iterable_scalars = [0]
            #WARNING! this is assuming mol2 is the last one on the list. make auto
            for it in self.iterables[:-1]:
                try:
                    bulk_value=float(str(it).split('M')[0]) 
                    unit='M'
                except:
                    bulk_value=float(str(it).split('mM')[0])
                    unit='mM'
                iterable_scalars.append(bulk_value)
            #print(iterable_scalars)    
            for state_col in df.columns:
                ax.plot(iterable_scalars, df[state_col].values[:-1])


def plot_layout(parameters):
    """Function to optimize plot layout based on number of subplots generated."""
    
    length=len(parameters)
    
    if length < 4:
        rows=1
        columns=length
    else:
        rows=2
        
        if not length % 2 == 0:
            columns=int(np.rint((length/2)+0.1))
    
        else:
            columns=int(length/2)
        
    if not length % 2 == 0 and length > 4:   
        
        fix_layout=True 
    else:
        fix_layout=False
    
    return rows, columns, fix_layout


def plot_feature_histogram(feature_dict):
    """Function to plot the histogram of raw features."""
    
    rows, columns, fix_layout=plot_layout(parameters)
    fig,axes=plt.subplots(rows, columns, sharex=True, constrained_layout=True)

    plt.suptitle(f'Feature: {name}')

    for plot, feature in zip(axes.flat, features.items()):
        for f in feature[1]:
            if not os.path.exists(f):
                feature[1].remove(f)
                print(f'\tCould not find data for {f}')
                
        data=np.asarray(pyemma.coordinates.load(feature[1]))
        plot.hist(data.flatten()) #, label=feature[0])
        plot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plot.set_xlabel(r'{} ($\AA$)'.format(name))
        plot.set_ylabel('Counts')
        #plot.set_title(feature[0])

        data=[]
        
        fix_layout

            
    plt.savefig(f'{results}/histogram_{name}.png', bbox_inches="tight", dpi=600)
    
    return plt.show()

def plot_MFPT(mfpt_df, scheme, feature, parameters, error, regions=None, labels=None):
    """Function to plot heatmap of MFPTS between all states."""
        
    # Taken from matplotlib documentation. Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())     
        
    images = []
    
    
    rows, columns, fix_layout=plot_layout(parameters)
    fig, axes = plt.subplots(rows, columns, constrained_layout=True, figsize=(9,6))
    fig.suptitle(f'Discretization: {scheme}\nFeature: {feature} (error tolerance {error:.1%})', fontsize=14)  
        
    cmap_plot=plt.cm.get_cmap("gist_rainbow")
    #cmap_plot.set_under(color='red')

    #cmap_plot.set_over(color='yellow')
    cmap_plot.set_bad(color='white')
        
        
    vmins, vmaxs=[], []
            
    for plot, parameter in zip(axes.flat, parameters):
            
        means=base.MSM.mfpt_filter(mfpt_df, scheme, feature, parameter, error) 
            
        try:
            if scheme == 'combinatorial':        
                contour_plot=plot.pcolormesh(means, edgecolors='k', linewidths=1, cmap=cmap_plot) 
                label_names=base.Functions.sampledStateLabels(regions, sampled_states=means.index.values, labels=labels)
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
    
    

def plot_flux(flux_df, ligand, results):
    """Function to plot fluxes from a dataframe of fluxes."""

    schemes=flux_df.index.unique(level='Scheme')
    features=flux_df.index.unique(level='feature')
    
    
    for scheme in schemes:
        for feature in features:
            
            properties=flux_df.columns
            
            fig, axes = plt.subplots(1,len(properties), sharex=True, constrained_layout=True)
            fig.suptitle(f'Discretization: {scheme}\n Feature: {feature}')
                       
            for ax, prop in zip(axes, properties):

                flux_df.loc[(scheme, feature), prop].plot(linestyle='-', marker='o', ax=ax, title=prop)
                ax.set_xlabel(f'[{ligand}] (M)')
                ax.set_ylabel(prop+r' ($s^{-1}$)')
            plt.savefig(f'{results}/netFlux_{scheme}-{feature}.png', dpi=600)
            plt.show()

    
        
def plot_pathways(pathway_df, ligand, results):
    """Function to plot pathways from a dataframe of pathways."""
    
    
    
    schemes=pathway_df.index.unique(level='Scheme')
    features=pathway_df.index.unique(level='feature')
    
    
    for scheme in schemes:
        for feature in features:
        
            print(scheme, feature)
            
            pathway_df.loc[(scheme, feature)].dropna(axis=1, how='all').plot(linestyle='-', marker='o')
            plt.title(f'Discretization: {scheme}\n Feature: {feature}')
            plt.xlabel(f'[{ligand}] (M)')
            plt.ylabel(r'Pathway flux ($s^{-1}$)')
            plt.savefig(f'{results}/pathways_{scheme}-{feature}.png', dpi=600)
            plt.show()
 

def plot_committor(committor_df, ligand, results, regions, labels):
    """Function to plot fluxes from a dataframe of fluxes."""
        
    schemes=committor_df.index.unique(level='Scheme')
    features=committor_df.index.unique(level='feature')
    parameters=committor_df.index.unique(level='parameter')
    
    
    committors=('Forward', 'Backward')
    
    for scheme in schemes:
        for feature in features:
            
            fig, axes = plt.subplots(1,len(committors), sharey=True, constrained_layout=True)
            for parameter in parameters:
                for ax, committor in zip(axes, committors):
                    try:
                        df_c=committor_df.loc[(scheme, feature, parameter), committor].dropna()

                        if scheme == 'combinatorial':

                            df_c.plot(linestyle='-', marker='o', ax=ax, label=parameter, title=committor)

                            sampled_states=committor_df.unstack(level=2).loc[(scheme, feature)][committor].dropna().index.unique(level='states').values
                            label_names=base.Functions.sampledStateLabels(regions, sampled_states=sampled_states, labels=labels)
                    
                            ax.set_xticks(sampled_states)
                            ax.set_xticklabels(label_names, rotation=70)

                        else:
                            df_c.plot(ax=ax, label=parameter, title=committor)  
                        ax.set_ylabel('Committor Probability')
                    except:
                        print('no plot at ', parameter)
                        
            plt.suptitle(f'Discretization: {scheme}\n Feature: {feature}')
            
            plt.legend(title=f'[{ligand}] (M)')
            plt.savefig(f'{results}/committors_{scheme}-{feature}.png', dpi=600)
            plt.show()

         
    