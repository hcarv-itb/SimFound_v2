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
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.pylab as pl
from matplotlib.ticker import FormatStrFormatter, AutoLocator, MaxNLocator, ScalarFormatter, LogLocator, AutoMinorLocator
import matplotlib.gridspec as gridspec

import Discretize

def plot_scheme(states_scheme):
       
    color_rect = ['white', 'dimgrey']
    for idx_combination, (label,combination) in enumerate(states_scheme.items()):
        fig, ax_scheme = plt.subplots(1,1, figsize=(5,1)) 
        print(label)

        #ax_scheme.s
        for idx,color_idx in enumerate(combination):
            color = color_rect[color_idx]
            if idx == 0 and color_idx == 1:
                color = 'darkorange'
            rect = plt.Rectangle((idx,0), 1, 1, facecolor=color, edgecolor='black', linewidth=5)
            #print(idx, color, rect)
            ax_scheme.add_patch(rect)
        #ax_scheme.set_title(label)
        ax_scheme.set_xlim([0,5])
        ax_scheme.axis('off')
        #ax_scheme.set_ylim([0,9])
        fig.show()
        fig.savefig(f'/media/dataHog/hca/msAcT-acylOctamer/results/figures/comb_states/{label}.png', dpi=300)
            #(idx, idx+1), 2, 1, linewidth=1, edgecolor='r', facecolor='none')
    #ax_scheme.set_xticks(ax_scheme.get_xticks()-0.5)
    #ax_scheme.set_xticklabels(['A', 'T', 'E', 'S', 'B'])
    
    #ax_scheme.set_ylim([0,9])
    #ax_scheme.axis('scaled')
        
        

            
            
class Plots:
    
    
    loc1, loc2 =0, 350
    plt.rc('font', size=10) 
    p_type = {'normal' : {'grid_layout' : (3,1), 'bar_width' : 0.8, 'outer_grid_heights' : [8,4,8], 'figsize' : (11,11)},
          'inhibition' : {'grid_layout' : (3,1), 'bar_width' : 0.4,  'outer_grid_heights' : [6,4,4], 'figsize' :(10,11)},
          'water' : {'grid_layout' : (3,3), 'bar_width' : 0.4,  'outer_grid_heights' : [6,2,6], 'figsize' :(10,11)}}
 
    
    def __init__(self,
                 supra_project : dict,
                    dG : pd.DataFrame,
                    combinatorial : pd.DataFrame,
                    plot_specs :dict,
                    input_states : dict,
                    subset_states : dict,
                    labels_regions = ['A', 'T', 'E', 'S', 'B'],
                    mol_water={} ,
                    color2='darkorange',
                    mol2={'BuOH' : 'ViAc', 'BeOH' : 'BeAc'},
                    discretized_matrixes={},
                    discretized_trajectories={},
                    metrics= False,
                    metric_method='Mutual Information (normalized)'):
        
        self.supra_project : dict = supra_project 
        self.dG : pd.DataFrame = dG 
        self.combinatorial : pd.DataFrame = combinatorial
        self.plot_specs : dict = plot_specs
        self.input_states : dict = input_states
        self.subset_states : dict = subset_states
        self.mol2 :dict = mol2
        self.color2 : str = color2
        self.mol_water : dict = mol_water
        self.metrics :bool = metrics
        self.metric_method : str = metric_method
        if self.metrics:
            self.discretized_trajectories : dict = discretized_trajectories
            self.discretized_matrixes : dict = discretized_matrixes
            

    @classmethod
    def MsAcTPublicationFigure(cls):
        print(cls)
        return cls.plotBindingProfileAndCombinatorial(cls)
        
        #return cls.plotBindingProfileAndCombinatorial()
        
    def plot_dg(self, input_df):


        it = self.it 
        
        dg, dg_m, dg_p = self.getDeltaG(input_df, it, self.idx_it)        

        ax = self.ax_dG
        loc1, loc2 = self.loc1, self.loc2
        ranges = dg.iloc[loc1:loc2].index.values
        if it.split('_')[-1] == '5mM':
            it = f'{it.split("_")[0]} + {" ".join(it.split("_")[1:])}'

        if self.project_type == 'water' and self.mol != 'H2O':
            #loc2 = loc2 - 60
            ranges = dg.iloc[loc1:loc2].index.values
            dg_diff = dg.iloc[loc1:loc2].values - self.dG_base_df.iloc[loc1:loc2].values

            ln = ax.plot(ranges, 
                         dg_diff, 
                         linestyle=self.linestyle, 
                         color= self.color_mol[self.idx_it], 
                         label=it)
            if self.idx_mol == 0:
                ax.set_ylabel(r'$\Delta$$\Delta$G ($\it{k}$$_B$T)')
            else:
                ax.set_yticklabels([])
            ax.set_ylim(-3, 1)
            ax.legend(loc='lower right')
        else:
            ln = ax.plot(ranges, 
                         dg.iloc[loc1:loc2], 
                         '.',   
                         color=self.color_mol[self.idx_it], 
                         label=it) #, marker=marker, fillstyle=fillstyle)
            ax.fill_between(ranges, dg_m.iloc[loc1:loc2], dg_p.iloc[loc1:loc2], alpha=0.5, color=self.color_mol[self.idx_it])
            ax.set_ylabel(r'$\Delta$G ($\it{k}$$_B$T)')
            ax.legend(loc='upper right')
        #ax.set_ylim(-1,14)
        ax.set_xticks([])
        
        ax.set_xscale('log') 
        ax.axhline(y=0, ls='--', color='darkgray')
        
        
        ax.set_title(self.mol)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        #ax.xaxis.set_major_formatter(FormatStrFormatter("%d")) #ScalarFormatter())
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(which='major', axis='y', linestyle='--')
        
        self.lns.append(ln)

        print(self.idx_it, self.it, self.mol)
        if self.project_type == 'water' and self.mol == 'H2O':
            self.dG_base_df = dg
            
            
 
    #TODO: merge with plot_dg
    def plot_dg_inhib(self, input_df):
        
        ax = self.ax_dG
        ax2 = self.ax2
        mol = self.mol
        mol2 = self.mol2
        lns = self.lns

        dg2, dg2_m, dg2_p = self.getDeltaG(input_df, self.it, self.idx_it)
        loc1, loc2 = self.loc1, self.loc2
        ranges = dg2.iloc[loc1:loc2].index.values
        ln2 = ax2.plot(ranges, dg2.iloc[loc1:loc2],'.', fillstyle='none', color=self.color2, label=f'5mM {mol2[mol]}')
        ax2.fill_between(ranges, dg2_m.iloc[loc1:loc2], dg2_p.iloc[loc1:loc2], alpha=0.2, color=self.color2)
        ax2.set_xlabel(r'$\itd$$_5$ ($\AA$)') #, color=self.color2)
        ax2.set_xlim(1.5, ax.get_xlim()[1])
        ax.set_xlim(1.5, ax.get_xlim()[1])
        ax2.set_xscale('log')
        #ax2.tick_params(axis='x', labelcolor=self.color2)
        ax2.set_ylim(-4, self.ax_dG.get_ylim()[1])
        lns.append(ln2)
        labs = [p[0].get_label() for p in lns]
        self.ax_dG.legend([ln[0] for ln in lns], labs,loc='upper right')
        #ax.set_title(mol)
        
        #ax2.set_ylim(-4, 6)
        

    def plot_regions(self, df):
        
        
        ax = self.ax_regions
        steps = self.steps
        count_total = df.size
        state_sampling_fraction = self.getStateSamplingFraction(df)
        

        steps_to_plot = [self.ax_dG.get_xlim()[0]]
        for i in steps:
            
            steps_to_plot.append(i)
        steps_to_plot.append(self.ax_dG.get_xlim()[1])
        
        sampling_fraction = [c for c in state_sampling_fraction]
        sampling_fraction.append(1.0)        
        
        ax.step(steps_to_plot, sampling_fraction, where='post', linestyle=self.linestyle, color=self.color_mol[self.idx_it], label=self.it)
        
        if self.project_type == 'normal':
            ax.set_yscale('log')
            ax.set_yticks([1e-4, 1e-2, 1])

        
        ax.set_xlim(self.ax_dG.get_xlim())
        ax.set_xscale('log')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        if self.mol == 'ViAc':
            ax.set_xlabel(r'$\itd$$_{NAC,donor}$ ($\AA$)')
        elif self.project_type == 'water':
           ax.set_xlabel(r'$\itd$$_{NAC,hydrolysis}$ ($\AA$)') 
           ax.set_ylim(0.8,1.05)

        else:
            ax.set_xlabel(r'$\itd$$_{NAC,acceptor}$ ($\AA$)')

        if (self.idx_mol == 0 and self.project_type == 'inhibition') or (self.idx_mol == 0 and self.mol == 'H2O'):   
            ax.set_ylabel('Sampling\nfraction')

                   
        if self.idx_it == len(self.iterables)-1:

            count_regions = self.count_regions[self.idx_it]
            label_ticks = []
            for i in self.steps:
                label_ticks.append(i) 
            label_ticks.append(self.ax_regions.axes.get_xlim()[1])
            label_centers = np.mean(np.array([self.steps, label_ticks[1:]]), axis=0) 
            
            if self.project_type == 'normal':
                label_location = 5e-4
            elif self.project_type == 'inhibition':
                label_location = 1e-1
            else:
                label_location = 0.85
                
            for idx_labels, x in enumerate(label_ticks): #
                if idx_labels >= 0:
                    ax.text(x, label_location, self.labels_regions[idx_labels], ha='right')
                if idx_labels < len(label_ticks)-1:
                    ax.vlines(x, 0, count_regions[idx_labels], linestyle='dotted', color='grey') 
 
    
    def plot_combinatorial(self, df, opt_color=None):
        
        if isinstance(opt_color, np.ndarray):
            color = opt_color
        else:
            color = self.color_mol
        ax=self.ax_comb
        states = self.states

        #color = self.color_mol
        if self.project_type == 'water':
            df.plot.barh(ax=ax,
                          color=color,
                          label=False),
                          #width=Plots.p_type[self.project_type]['bar_width'])
            #ax.set_xscale('log') 
            ax.set_yticks(list(states.keys()))
            
            ax.set_xlabel('Sampling fraction')
            print(self.idx_mol)
            
            #plt.setp(ax.get_xtick(), visible=True)
            if self.idx_mol == 2:
                
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_yticklabels(list(states.values())) #, rotation=0)
                ax.set_ylabel('States')
        else:
            df.plot.bar(ax=ax, 
                        color=color, 
                        label=False, 
                        width=Plots.p_type[self.project_type]['bar_width']) 

            ax.set_yscale('log') 
            ax.set_xticks(list(states.keys()))
            ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
            ax.set_xticklabels(list(states.values()), rotation=0)
        
    
            if self.project_type == 'normal':
                if self.idx_mol == 1:
                    ax.set_ylabel('Sampling fraction')
                    plt.setp(ax.get_xticklabels(), visible=False)
                elif self.idx_mol == len(self.projects)-1:
                    ax.set_xlabel('States')
                    ax.set_xticklabels(list(states.values()), rotation=0)
                    
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
            elif self.project_type == 'inhibition':
                ax.set_ylabel('Sampling fraction')
                ax.set_xlabel('States')

        ax.grid(linestyle='--', axis='y')
        
        ax.legend().remove() 
        
        
    
    def plot_wrapper(self, 
                     dG_df : pd.DataFrame, 
                     comb_df : pd.DataFrame):
        
        idx = self.idx_it
        list_its_ = comb_df.columns.get_level_values('l3').unique().to_list()
        
        if self.project_type == 'inhibition':
            list_its_.remove('5mM')

        comb_it_df = comb_df.loc[:, comb_df.columns.get_level_values('l3') == list_its_[idx]]
             
        self.counts_states[idx] =  np.asarray([np.count_nonzero(comb_it_df == i) for i in self.states]) / comb_it_df.size
        self.count_regions[idx] = self.getStateSamplingFraction(comb_it_df)
        
        self.plot_dg(dG_df)
        self.plot_regions(comb_it_df)
        

    
    
    
    def plot_metrics(self, correlation):
        
        
        #correlation = correlation_table[0]
        

        
        ax = self.ax_corr
        cmap= plt.get_cmap('YlGn') #'RdYlGn')
        cmap.set_bad('black')
        
        (state_labels, states) = list(self.states.values()), list(self.states.keys()), 
        #cmap.set_bad('grey')
        
        if self.project_type == 'inhibition':
            label1, label2 = self.mol2[self.mol], self.mol
            ticks1, ticks2 = states, states
            ticklabels1, ticklabels2 = state_labels, state_labels
            vmin, _vmax = 0, 0.006156262816740886

        else:
            
            label2, label1 = 'water', self.mol
            ticks2, ticks1 = states, list(self.states_mol)
            ticklabels2, ticklabels1 = state_labels, list(self.state_labels_mol)
            correlation = correlation.T
            vmin, _vmax = 0, 0.008432989270410602

        vmin, vmax = 0, 1
        print(np.nanmax(correlation), np.nanmin(correlation))
        corr_img = ax.imshow(correlation / _vmax, cmap=cmap, vmin=vmin, vmax=vmax)
        
        ax.set_yticks(ticks2)
        ax.set_xticklabels(ticklabels1, rotation=90)
        ax.set_yticklabels(ticklabels2)
        ax.set_xticks(ticks1)
        

        ax.invert_yaxis()
        if self.project_type == 'inhibition':
            ax.set_xlabel(label1)
            ax.set_ylabel(label2)
            ax.set_xticks(ticks1)

            if self.idx_mol == len(self.projects)-1 :
                cbar = plt.colorbar(corr_img, ax=ax)
                cbar.set_label(self.metric_method)
        
        elif self.project_type == 'water':
            
            if self.it.split('_')[-1] == '5mM':
                it = f'{self.it.split("_")[0]}\n+\n{" ".join(self.it.split("_")[1:])}'
            else:
                it = self.it
            
            ax.set_ylabel(it, rotation='horizontal', va='center', ha='right')
            
            if self.idx_it != len(self.iterables)-1:   
                plt.setp(ax.get_xticklabels(), visible=False)
        
            else:
                ax.set_xlabel(label1)
            
            if self.idx_mol == 1:
                plt.setp(ax.get_yticklabels(), visible=False)    
                
            return corr_img


    def setSubplots(self, grid_dG, grid_comb):
        
        idx_mol = self.idx_mol
        project_type = self.project_type
        
        if idx_mol == 0:
            if project_type == 'normal':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol])
                self.ax_comb = plt.subplot(grid_comb[idx_mol,0])
                self.ax_regions = plt.subplot(grid_dG[1, idx_mol])
            elif project_type == 'inhibition':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol])
                self.ax_comb = plt.subplot(grid_comb[0,idx_mol])
                self.ax_regions = plt.subplot(grid_dG[1, idx_mol])
            else:
                self.ax_dG = plt.subplot(grid_dG[0, 0])
                #self.ax_comb = plt.subplot(grid_comb[0,idx_mol])
                self.ax_regions = plt.subplot(grid_dG[1, 0])
        else:
            #ax = plt.subplot(grid_dG[0, idx_mol], sharey=ax)
            if project_type == 'normal':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol], sharey=self.ax_dG)
                self.ax_comb = plt.subplot(grid_comb[idx_mol,0], sharex=self.ax_comb, sharey=self.ax_comb)
            elif project_type == 'inhibition':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol], sharey=self.ax_dG)
                self.ax_comb = plt.subplot(grid_comb[0, idx_mol], sharey=self.ax_comb)
            elif project_type == 'water':
                self.ax_comb = plt.subplot(grid_comb[0, idx_mol])

            self.ax_regions = plt.subplot(grid_dG[1, idx_mol], sharey=self.ax_regions)



    def plotBindingProfileAndCombinatorial(self):
        figures = {}
        for project_type, projects in self.supra_project.items():
            print(project_type)
            input_states = self.input_states
            p_type = self.p_type
            dG = self.dG
            combinatorial = self.combinatorial
            plot_specs = self.plot_specs
            loc1, loc2 = self.loc1, self.loc2
            mol2 = self.mol2
            color2 = self.color2
             
            fig=plt.figure(figsize=p_type[project_type]['figsize'], constrained_layout=True) #
            (nrows, ncols) = p_type[project_type]['grid_layout']
            outer_grid = gridspec.GridSpec(nrows, ncols, hspace=0.3, height_ratios = p_type[project_type]['outer_grid_heights']) #, width_ratios = [3,1]) 
            
            if project_type == 'normal':
                grid_dG = gridspec.GridSpecFromSubplotSpec(2, len(projects), subplot_spec = outer_grid[0,:], hspace=0.3, wspace=0.1, height_ratios=[4,2]) 
                grid_comb = gridspec.GridSpecFromSubplotSpec(len(projects), 1, subplot_spec = outer_grid[2,:], hspace=0.2) #, wspace=0.1) #wspace=0.2)
                grid_figure = gridspec.GridSpecFromSubplotSpec(1, len(projects), subplot_spec = outer_grid[1,:], wspace=0.1)
            elif project_type == 'inhibition':
                grid_dG = gridspec.GridSpecFromSubplotSpec(2, len(projects), subplot_spec = outer_grid[0,:], hspace=0.3, wspace=0.1, height_ratios=[4,2]) 
                grid_comb = gridspec.GridSpecFromSubplotSpec(1, len(projects), subplot_spec = outer_grid[1,:], hspace=0.2, wspace=0.1)
                grid_figure = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer_grid[2,:], wspace=0.1)
            elif project_type == 'water':
                grid_dG = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec = outer_grid[0,:], height_ratios=[2,1], hspace=0.2, wspace=0.2)  
                grid_comb = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outer_grid[1,1:], wspace=0.2)
                grid_figure = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = outer_grid[2,1:], wspace=0.3, height_ratios=[10,2])
                #grid_corr = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer_grid[2,:], wspace=0.1)
                    
            
            
            supra_df = pd.DataFrame()
            
            self.projects : dict = projects
            self.project_type : str = project_type
            self.labels_regions = list(self.subset_states[project_type].keys())
            
            for idx_mol, (mol, project) in enumerate(projects.items()):
                print(mol)
                

                if project_type != 'water':
                    dG_df = dG[project_type][mol]
                else:
                    base_df = dG[project_type][mol]
                    dG_df = base_df.iloc[:, base_df.columns.get_level_values(0) == mol].droplevel(0, axis=1)

 
                if isinstance(input_states[project_type], tuple):
                    self.states = input_states[project_type][1]
                    combinatorial_df = combinatorial[project_type][mol].replace(to_replace=input_states[project_type][0].keys(), value=self.states.keys())
                else:
                    self.states = input_states[project_type][0].values()
                    combinatorial_df = combinatorial[project_type][mol]
                
                self.iterables = project.parameter
                self.steps = plot_specs[project_type][mol][0]
                self.color_mol = plot_specs[project_type][mol][1]
                self.linestyle = plot_specs[project_type]['linestyle'] 
                
                self.idx_mol = idx_mol
                self.mol = mol
                
                self.setSubplots(grid_dG, grid_comb)
                
                self.lns = []

                #ax_comb = plt.subplot(grid_comb[idx_mol])
                self.counts_states = np.empty([len(self.iterables), len(self.states)])
                self.count_regions = np.empty([len(self.iterables), len(self.steps)+1])
                
                
                for idx, it in enumerate(self.iterables): 
                    self.it : str = it
                    self.idx_it = idx
                    self.plot_wrapper(dG_df, combinatorial_df) 
                    
                    
                    
                mol_df = pd.DataFrame(self.counts_states.T, index=list(self.states.keys()), columns=self.iterables)
                #mol_df = pd.DataFrame(counts_states, columns=list(states.keys()), index=iterables)    
                supra_df = pd.concat([supra_df, mol_df], axis=1)
                if project_type != 'water':
                    self.plot_combinatorial(mol_df)
                
                #NOTE: loading of mol1 and mol1 + mol2 is done above. Here is only to handle mol2
                if project_type == 'inhibition':
                    print(mol)
                    if self.metrics:
                        self.ax_corr = plt.subplot(grid_figure[0,idx_mol])
                        self.plot_metrics(self.discretized_matrixes[project_type][mol])
                    
                    self.ax2 = self.ax_dG.twiny()
                    self.color_mol = [color2]
                    self.linestyle = 'dashed'
                    self.idx_it = 0
                    self.it = '5mM'
                    inib_mol = mol2[mol]

                    inib_df = pd.read_csv(f'{project.results}/binding_profile/binding_profile_acylSer-His_{inib_mol}_mean_b0_e-1_s1.csv', index_col=0)    
                    df_comb_inhib = combinatorial_df.loc[:, combinatorial_df.columns.get_level_values('l3') == self.it]
                    
                    
                    self.plot_dg_inhib(inib_df)
                    self.plot_regions(df_comb_inhib)
                    state_sampling_fraction = self.getStateSamplingFraction(df_comb_inhib, w_subset=False)
                    x_values = self.ax_comb.get_xticks().flatten() + 0.35
                        
                    #plot_combinatorial
                    self.ax_comb.bar(x_values, state_sampling_fraction, width=0.2, color=color2)
            
            #NOTE: This is only loading the other _water dfs. The df of water has been handled before
            if project_type == 'water':
                
                
                
                
                sub_w_grid_dG = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = grid_dG[:,1:], height_ratios=[2,1], hspace=0.2, wspace=0.1)
                
                for idx_mol_w, (mol_w, iterables_mol_w2) in enumerate(self.mol_water.items()):
                    print(mol_w, iterables_mol_w2)


                        #self.plot_correlations(self.discretized_trajectories[project_type][mol_w])  
                        
                    base_df_mol = dG[project_type]['H2O']
                    dG_mol_df = base_df_mol.iloc[:, base_df_mol.columns.get_level_values(0) == mol_w].droplevel(0, axis=1)
                    
                    
                    color_mol = plot_specs['normal'][mol_w][1] 
                    color_mol = np.vstack([color_mol, pl.cm.Oranges(np.linspace(0.5,0.5,1))])
                    color_mol_comb = np.vstack([pl.cm.Greys(np.linspace(1,1,1)), color_mol, pl.cm.Oranges(np.linspace(0.5,0.5,1))])
                    
                    self.color_mol = color_mol
                    self.mol = mol_w
                    self.idx_mol = idx_mol_w 
                    
                    self.ax_dG = plt.subplot(sub_w_grid_dG[0, idx_mol_w])
                    self.ax_regions = plt.subplot(sub_w_grid_dG[1, idx_mol_w], sharey=self.ax_regions)
                    self.iterables = iterables_mol_w2[0] + [f'100mM_{iterables_mol_w2[1]}_5mM']
                    self.counts_states = np.empty([len(self.iterables), len(self.states)])
                    self.count_regions = np.empty([len(self.iterables), len(self.steps)+1])
                    if idx_mol_w == 0:
                        self.ax_comb = plt.subplot(grid_comb[0, idx_mol_w])
                    else:
                        self.ax_comb = plt.subplot(grid_comb[0, idx_mol_w], sharey=self.ax_comb)
                    if isinstance(input_states[project_type], tuple):
                        self.states = input_states[project_type][1]
                        base_states_mol = input_states['normal'][1]

                        comb_df = combinatorial[project_type][mol].replace(to_replace=input_states[project_type][0].keys(), value=self.states.keys())
                    else:
                        self.states = input_states[project_type][0].values()
                        base_states_mol = input_states['normal'][0]
                        
                        comb_df = combinatorial[project_type][mol]
                    comb_df = comb_df.iloc[:, comb_df.columns.get_level_values('l2') == mol_w]
                    

                            
                    self.states_mol = base_states_mol.keys()
                    self.state_labels_mol = base_states_mol.values()
                    
                    if self.metrics:
                        sub_w_grid_metrics = gridspec.GridSpecFromSubplotSpec(len(self.iterables),1, subplot_spec = grid_figure[0, idx_mol_w])
                    for idx_w, it_w in enumerate(self.iterables): 
                        self.it = it_w
                        self.idx_it = idx_w
                        
                        #WARNING! This is assuming mol2 is the last iterable! make auto
                        if idx_w < len(self.iterables) -1 :
                            self.linestyle = plot_specs[project_type]['linestyle']
                        else:
                            self.linestyle = 'dashed'
                        #combinatorial_mol_df = comb_df.iloc[:, comb_df.columns.get_level_values('l2') == mol_w]

                        self.plot_wrapper(dG_mol_df, comb_df)
                        
                        
                        if self.metrics:    
                            self.ax_corr = plt.subplot(sub_w_grid_metrics[idx_w,0])
                            
                            corr_image = self.plot_metrics(self.discretized_matrixes[project_type][mol_w][it_w])
                            

                    
                        
                    w_count_states = np.vstack([mol_df.values.T, self.counts_states]) 
                    mol_water_df = pd.DataFrame(w_count_states.T, index=list(self.states.keys()), columns=['0 mM'] + self.iterables)
                    self.plot_combinatorial(mol_water_df, opt_color=color_mol_comb)
                    
                if self.metrics :
                    
                    _colorbar = plt.subplot(grid_figure[1,:])
                    _colorbar.axis('off')
                    cbar = plt.colorbar(corr_image, ax=_colorbar, orientation='horizontal')
                    cbar.set_label(self.metric_method)              
    # =============================================================================
    #             else:
    #                 ax_figure = fig.add_subplot(grid_figure[idx_mol])
    #                 if mol == 'BuOH':
    #                     image_mol = plt.imread(f'{project.results}/figures/msAcT_BuOH_100mM_activeSite_ATESB.png')
    #                 elif mol =='BeOH':
    #                     image_mol = plt.imread(f'{project.results}/figures/msAcT_BeOH_100mM_activeSite_ATESB.png')
    #                 else:
    #                     image_mol = plt.imread(f'{project.results}/figures/msAcT_ViAc_50mM_activeSite_ATESB.png')
    #                 
    #                 ax_figure.imshow(image_mol)
    #                 ax_figure.axis('off')
    # =============================================================================
                    
    
    
            #plot_combinatorial(supra_df)
            #fig.tight_layout()
            fig.show()
            figures[project_type] = fig
            fig.savefig(f'{project.results}/figures/BindingProfile_combinatorial_{project_type}.png', dpi=600, bbox_inches='tight')
        
        return figures
               
    def getDeltaG(self, input_df, it, idx):
        

        if self.project_type == 'inhibition':
            #print(input_df)
            print(it, self.mol2[self.mol])
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



                           
                
                
    def getStateSamplingFraction(self, df, w_subset=True) -> list:
        
        count_total = df.size
        
        state_sampling_fraction = []
        if w_subset:
            for idx_regions, (label_state, subset) in enumerate(self.subset_states[self.project_type].items()):
                state_sampling_fraction.append(np.asarray([np.count_nonzero(df == i) / count_total for i in self.states if i in subset]).sum())
        else:
            state_sampling_fraction = np.asarray([np.count_nonzero(df == i) / count_total for i in self.states])
        
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
            print(iterable_scalars)    
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
         
    