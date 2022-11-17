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
from matplotlib.ticker import FormatStrFormatter, AutoLocator, PercentFormatter, MaxNLocator, ScalarFormatter, LogLocator, LinearLocator, AutoMinorLocator
import matplotlib.gridspec as gridspec

import Discretize


      

            
            
class Plots:
    
    
    loc1, loc2 =0, 350
    plt.rc('font', size=10) 
    p_type = {'normal' : {'grid_layout' : (3,2), 'bar_width' : 0.8, 'outer_grid_heights' : [8,4,8], 'outer_grid_widths' : [4,6], 'figsize' : (11,11)},
          'inhibition' : {'grid_layout' : (2,1), 'bar_width' : 0.4,  'outer_grid_heights' : [6,5], 'outer_grid_widths' : [1], 'figsize' :(10,11)},
          'water' : {'grid_layout' : (2,3), 'bar_width' : 0.4,  'outer_grid_heights' : [6,6], 'outer_grid_widths' : [1,1,1], 'figsize' :(10,11)}}
 
    
    def __init__(self,
                 supra_project : dict,
                    dG : pd.DataFrame,
                    combinatorial : pd.DataFrame,
                    plot_specs :dict,
                    input_states : dict,
                    subset_states : dict,
                    #labels_regions = ['A', 'T', 'E', 'S', 'B'],
                    mol_water={} ,
                    color2='darkorange',
                    mol2={'BuOH' : 'ViAc', 'BeOH' : 'BeAc'},
                    metric_data={},
                    discretized_trajectories={},
                    correlation_data={},
                    metrics= False,
                    metric_method='Mutual Information (normalized)',
                    double_comb_fractions={},
                    double_comb_states={}):
        
        self.supra_project : dict = supra_project 
        self.dG : pd.DataFrame = dG 
        self.combinatorial : pd.DataFrame = combinatorial
        self.plot_specs : dict = plot_specs
        self.input_states : dict = input_states
        self.subset_states : dict = subset_states
        #self.labels_regions : labels_regions
        self.mol2 :dict = mol2
        self.color2 : str = color2
        self.mol_water : dict = mol_water
        self.metrics : bool = metrics
        self.metric_method : str = metric_method
        if self.metrics:
            self.discretized_trajectories : dict = discretized_trajectories
            self.metric_data : dict = metric_data
            self.double_comb_fractions : dict = double_comb_fractions
            self.double_comb_states : dict = double_comb_states
            self.correlation_data : dict = correlation_data
            

    @classmethod
    def MsAcTPublicationFigure(cls):
        print(cls)
        return cls.plotBindingProfileAndCombinatorial(cls)
        
        #return cls.plotBindingProfileAndCombinatorial()

    @staticmethod
    def set_ticks(ax):
        ax.tick_params(axis='y', which='major', width=2, length=4)
        ax.tick_params(axis='x', which='major', width=2, length=6)
        ax.tick_params(axis='x', which='minor', length=6)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d")) #ScalarFormatter())
        locmax = LogLocator(base=10.0,subs=(0.2, 0.3, 0.5, 0.7, 1),numticks=12)
        ax.xaxis.set_major_locator(locmax)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d") ) 

        
    def plot_dg(self, input_df):


        it = self.it 
        
        if it == '5mM':
            pass
        else:
        
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
            
            if self.mol == 'H2O':
                ax.set_title('water', size=12)
            else:
                ax.set_title(self.mol, size=12)


            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.grid(which='major', axis='y', linestyle='--')
            
            
            self.set_ticks(ax)
            
            if self.project_type != 'normal':
                ax.tick_params(labelbottom=False)
            
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
        ax2.set_ylim(-4, self.ax_dG.get_ylim()[1])
        lns.append(ln2)
        labs = [p[0].get_label() for p in lns]
        self.ax_dG.legend([ln[0] for ln in lns], labs,loc='upper right')

        self.set_ticks(ax)
        self.set_ticks(ax2)

        

    def plot_regions(self, df):
        
        
        ax = self.ax_regions
        steps = self.steps
        count_total = df.size
        state_sampling_fraction = self.getStateSamplingFraction(df)
        
        steps_to_plot = [self.ax_dG.get_xlim()[0]]  


        sampling_fraction = [c for c in state_sampling_fraction] 
        sampling_fraction.append(1.0)
        for i in steps:
            
            steps_to_plot.append(i)
        steps_to_plot.append(self.ax_dG.get_xlim()[1])
        
        ax.step(steps_to_plot, sampling_fraction, where='post', linestyle=self.linestyle, color=self.color_mol[self.idx_it], label=self.it)
        
        if self.project_type == 'normal':
            ax.set_yscale('log')
            locmax = LogLocator(base=10.0, numticks=6)
            ax.yaxis.set_major_locator(locmax)
            #ax.set_yticks([1e-4, 1e-2, 1])

        
        ax.set_xlim(self.ax_dG.get_xlim())
        ax.set_xscale('log')
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        self.set_ticks(ax)

        #ax.yaxis.set_minor_formatter(FormatStrFormatter("%d") ) 
        
        
        
        if self.mol == 'ViAc':
            ax.set_xlabel(r'$\itd$$_{NAC,donor}$ ($\AA$)')
        elif self.project_type == 'water':
           ax.set_xlabel(r'$\itd$$_{NAC,hydrolysis}$ ($\AA$)') 
           ax.set_ylim(0.8,1.05)

        else:
            ax.set_xlabel(r'$\itd$$_{NAC,acceptor}$ ($\AA$)')

        if (self.idx_mol == 0 and self.project_type != 'water') or (self.idx_mol == 0 and self.mol == 'H2O'):   
            ax.set_ylabel('Sampling\nfraction')

                   
        if self.idx_it == len(self.iterables)-1:

            count_regions = self.count_regions[self.idx_it]
            label_ticks = []
            for i in self.steps:
                label_ticks.append(i)
                
                
            if self.project_type != 'water':
                label_ticks.append(self.ax_regions.axes.get_xlim()[1])
            label_centers = np.mean(np.array([self.steps, label_ticks[1:]]), axis=0) 
            
            
            
            if self.project_type == 'normal':
                label_location = 5e-4
            elif self.project_type == 'inhibition':
                label_location = 0.5
            else:
                label_location = 0.85
                if self.idx_mol != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                
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
                        width=Plots.p_type[self.project_type]['bar_width']) 
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
        

    def plot_combinatorial_polar(self, df, opt_color=None):
        
        index = df.index.values
        num_vars = len(index)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        print(df)
        
        for idx, column in enumerate(df.columns):
            
            _df = df[column]
            values = _df.values.tolist()
            values += values[:1]
            ax = self.ax_comb
            
            ax.plot(angles, values, color=self.color_mol[idx], linewidth=1)
            
        #plt.show()
    
        
    def plot_subunitsStateSampling(self, input_df, 
                                   subunit_pair_labels, 
                                   ref_map = {'A-B' : (0,1), 
                                              'C-H' : (2,7), 
                                              'E-F' : (4,5), 
                                              'D-G' : (3,6)}):
        
        
        ax = self.ax_subunits
        #ax2 = self.ax_subunits2
        
        df = input_df.iloc[:, input_df.columns.get_level_values('l3') == self.it]
        total_sf = np.array(self.getStateSamplingFraction(df))
        references = df.columns.get_level_values('l5').unique()
        

        sampling_fraction ={}
        
        for ref_pair_label, indexes in ref_map.items():
            pair_sf = []
            for idx, idx_r in enumerate(indexes):

                label = chr(ord('@')+idx_r+1)
                ref = references[idx_r]
                ref_df = df.loc[:, df.columns.get_level_values('l5') == ref]
                _sf = self.getStateSamplingFraction(ref_df)
                pair_sf.append(_sf)
            sampling_fraction[ref_pair_label] = pair_sf
        
        total_sf = np.asarray(list(sampling_fraction.values()))
        
        number_pairs = np.shape(total_sf)[0]
        number_regions = np.shape(total_sf)[2]

            
        
        for idx_pair, pair in enumerate(total_sf):
            sampled_regions = total_sf.sum(axis=0).sum(axis=0)
            _sampled_regions = total_sf.sum(axis=1).sum(axis=0)
            _pair = ((pair / sampled_regions) * number_pairs*2) * 0.5
            _pair2 = (pair / _sampled_regions) 

            # * number_pairs #number dimers
            #max_pair = _pair.max()
            #_pair = pair
            
            #print(_pair, self.idx_it, self.idx_mol, self.mol)
            for idx_r, region in enumerate(_pair.T):
                
                x = [idx_r, idx_r+0.5]
                ax.plot(x, region, '.-' , color=self.color_mol[self.idx_it]) #, alpha=(1/number_pairs*2))

        ax.set_xticks(np.arange(0.25, number_regions+0.25))
        ax.set_xticklabels(self.labels_regions)
        ax.set_ylim(-0.25,4.5)
        ax.set_yticks(range(5))
        ax.set_yticklabels(['0:0', '2:1', '3:1', '4:1', '5:1'])
        
        if self.idx_mol == 1:
            ax.set_ylabel('Dimer sampling ratio')
        #ax.axhline(y=0, linestyle='dotted', color='red')
        
        

        #ax.yaxis.set_major_locator(LinearLocator(numticks=3))
        #ax.yaxis.set_major_formatter(PercentFormatter(25.0))

        #ax.set_ylim(-1, 3)
        

        if self.idx_mol != len(self.projects) -1:
            #plt.setp(ax.get_xticks(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            #plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_xlabel('Shells')

            #ax.spines["bottom"].set_visible(False)
            #ax.grid(linestyle='--', axis='y')
        
        if self.idx_it != len(self.iterables) -1 :
            ax.axhline(y=0.5, linestyle='solid', color='gray')
            ax.text(4, 0.8, '1:1', color='gray')
            
        

        
        

        
        
        
    
    

        

    
    
    
    def plot_metrics(self, input_, comb_states=None, color='red'):
        
        
        ax = self.ax_metric
        
        (metric, corr_coeff) = input_
        #cmap.set_bad('black')
        
        def set_symbols():
            for i in range(len(states)):
                for j in range(len(states_mol)):
                    if metric[i,j] == 0:
                        break
                    else:
                        value = corr_coeff[i, j]
                        if value < 0:
                            
                            _value = '-'
                            _color = 'red'
                        elif value > 0:
                            _value = '+'
                            _color = 'green'
                        else:
                            _value = ''
                            _color = 'white'
                        
                        ax.text(j, i, _value, ha="center", va="center", color=_color)
        
        #cmap.set_bad('grey')
        _legend = False
        cmap= plt.get_cmap('gray_r')
        #cmap.set_over('red')
        
        
        
        if self.project_type == 'normal':
            
            

            color = self.color_mol
            
            
            (state_labels, states) = list(self.states.values()), list(self.states.keys())
            plot = []

            df = pd.DataFrame()
            for iterable, matrix_pairs in metric.items():
                _data = []
                _labels = []
                for pair_label, matrix in matrix_pairs.items(): 
                    _data.append(matrix) #[data_pair, data_pair_1]
                    _labels.append(pair_label)

                data = np.asarray(_data).mean(axis=0)
                mean_sums = np.vstack([data.sum(axis=0),data.sum(axis=1)]).mean(axis=0)
                df_it = pd.DataFrame(mean_sums / mean_sums.max(), columns=[self.it])
                df = pd.concat([df, df_it], axis=1)
            
            plot = df.plot.bar(ax=ax, 
                        color=color, 
                        label=False, 
                        width=Plots.p_type[self.project_type]['bar_width']) 
            ax.grid(which='minor', linestyle='--', axis='y')
            ax.set_xticks(states)
            ax.legend().remove()
            ax.set_xticklabels(state_labels, rotation=90)
            ax.grid(linestyle='--', axis='y')
            if self.idx_mol == 1:
                ax.set_ylabel('Dimer mutual Information (normalized)')
            #ax.text(-0.05, 0.5, 'Mutual Information', orientation='vertical')
            if self.idx_mol != len(self.projects)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel('States')
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        
        elif self.project_type == 'inhibition':
            
            (state_labels, states) = list(self.states.values()), list(self.states.keys())
            (state_labels_mol, states_mol) = list(self.states_mol.values()), list(self.states_mol.keys())
            label1, label2 = f'{self.mol2[self.mol]} states', f'{self.mol} states'
            ticks1, ticks2 = states_mol, states
            ticklabels1, ticklabels2 = state_labels_mol, state_labels 
            
            vmin, vmax = 0, 1

            metric = metric / metric.max()

            ax.set_xlabel(label1)
            ax.set_ylabel(label2)
            ax.set_yticklabels(ticklabels2)
            ax.set_yticks(ticks2)
            if self.idx_mol == 1:
                plt.setp(ax.get_yticklabels(), visible=False)   
            
            plot = ax.imshow(metric, cmap=cmap, aspect='auto') #vmin=vmin, vmax=vmax, 
            set_symbols()
            
            ax.set_xticks(ticks1)
            ax.set_xticklabels(ticklabels1, rotation=90)

        elif self.project_type == 'water':
            
            (states_mol, comb_state_fractions) = comb_states
            ticks1 = list(states_mol.keys())
            ticklabels1 = list(states_mol.values())
            label1 = f'{self.mol} & {self.mol2[self.mol]} states'

            
            #selecting only one state since only two states are sampled here
            # _vmax = 0.008432989270410602
            _metric = metric[1,:]
            metric = _metric / np.max(_metric)
            corr_coeff = corr_coeff[1,:]

            
            ax_metric = ax.twinx()
            ax_metric.tick_params(axis='y', colors='red')
            ax_metric.spines['right'].set_color('red')
            ax.spines['top'].set_visible(False)
            ax_metric.spines['top'].set_visible(False)
            if self.idx_it != len(self.iterables)-1 or self.idx_mol == 1:
                plt.setp(ax.get_yticklabels(), visible=False)     
                plt.setp(ax.get_ylabel(), visible=False)
                plt.setp(ax.get_xlabel(), visible=False)

            else:
                ax.set_ylabel('Sampling \n fraction')

                

            if self.idx_it != len(self.iterables)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(label1)

            if self.idx_it != len(self.iterables)-1 or self.idx_mol == 0:                
                plt.setp(ax_metric.get_yticklabels(), visible=False)     
                plt.setp(ax_metric.get_ylabel(), visible=False)
                plt.setp(ax_metric.get_xlabel(), visible=False)
            elif self.idx_mol == 1:
                ax_metric.set_ylabel('MI (normalized)', color='red')
            

            
            plot = comb_state_fractions.plot.bar(stacked=True, ax=ax, legend=True, color=[color, 'white'], edgecolor=[color, color])
            if self.idx_mol == 0 and self.idx_it == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(labels, 
                          bbox_to_anchor=(0., 1.02, 1., .102), 
                          title='water states', 
                          loc='lower left',
                          ncol=2, 
                          mode="expand", 
                          borderaxespad=0.)
            else:
                ax.get_legend().remove()
            ax.set_yticks([0, 0.5,1])
            ax_metric.set_yticks([0, 0.5, 1])
            ax_metric.set_ylim(ax.get_ylim())
            markers = np.empty_like(metric).astype(str)

            
            ax.set_xticks(ticks1)
            ax.set_xticklabels(ticklabels1, rotation=90)
            for i in range(len(states_mol)):
                    
                value = corr_coeff[i]
                if value < 0:
                    
                    _value = r'$\ominus$'
                    _color = 'red'
                elif value > 0:
                    _value = r'$\oplus$'
                    _color = 'green'
                else:
                    _value = ''
                    _color = 'white'
                
                markers[i] = _value
                
                if metric[i] != 0:
                
                    ax.text(ax.get_xticks()[i], metric[i], _value, fontsize=12, ha="center", va="center", color=_color) 
            
            
            #ax.grid(which='major', axis='y', linestyle='--')
            #plot = ax_metric.scatter(comb_state_fractions.index.values, metric, marker=markers, color='red')


        
                
        return plot




    def plot_state_cartoon(self, orientation='horizontal'):
        

        color_code = {'B' : [0,0,0,0,1], 
                     'SB' : [0,0,0,1,1],
                     'ESB' : [0,0,1,1,1],
                     'TSB' : [0,1,0,1,1],
                     'TESB' : [0,1,1,1,1],
                     'ASB' : [1,0,0,1,1],
                     'AESB' : [1,0,1,1,1],
                     'ATSB' : [1,1,0,1,1],
                     'ATES' : [1,1,1,1,0],
                     'ATESB' : [1,1,1,1,1]}

       
        color_rect = ['white', 'lightgrey']
        
        
        ax = self.ax_scheme
        ax.set_xticks(range(len(color_code)))
        #ax.set_xticklabels(list(color_code.keys()))
        ax.set_ylim(0,5.5)
        ax.set_xlim(0,10)
        ax.get_xaxis().set_visible(False)

        for k, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_yticks(np.arange(0.5, len(self.labels_regions)+0.5))
        ax.set_yticklabels = self.labels_regions
        #ax.set_ylabel('Shells')
        for idx_combination, (label,combination) in enumerate(color_code.items()):
            
            for idx,sampled in enumerate(combination):
                   
                if orientation == 'horizontal':
                    anchor = (idx_combination,idx)
                else:
                    anchor = (idx, idx_combination)
                
                color = color_rect[sampled]
                if idx == 0 and sampled:
                    color = 'magenta'
                rect = plt.Rectangle(anchor, 0.8, 1, facecolor=color, edgecolor='black', linewidth=2)

                ax.add_patch(rect)        
        #ax_scheme.axis('scaled')
             


    def plot_wrapper(self, 
                     dG_df : pd.DataFrame, 
                     comb_df : pd.DataFrame):
        
        idx = self.idx_it
        list_its_ = comb_df.columns.get_level_values('l3').unique().to_list()
        
        if self.project_type == 'inhibition':
            list_its_.remove('5mM')

        it_df = comb_df.loc[:, comb_df.columns.get_level_values('l3') == list_its_[idx]]
             
        self.counts_states[idx] =  np.asarray([np.count_nonzero(it_df == i) for i in self.states]) / it_df.size
        self.count_regions[idx] = self.getStateSamplingFraction(it_df)

        self.plot_dg(dG_df)
        self.plot_regions(it_df)

    def setSubplots(self):
        
        
        grid_dG = self.grid_dG
        grid_comb = self.grid_comb
        
        idx_mol = self.idx_mol
        project_type = self.project_type
        
        if idx_mol == 0:
            if project_type == 'normal':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol])
                self.ax_comb = plt.subplot(grid_comb[idx_mol, 0], projection='polar')
                self.ax_regions = plt.subplot(grid_dG[1, idx_mol])

            elif project_type == 'inhibition':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol])
                self.ax_regions = plt.subplot(grid_dG[1, idx_mol])
            else:
                self.ax_dG = plt.subplot(grid_dG[0, 0])
                #self.ax_comb = plt.subplot(grid_comb[0,idx_mol])
                self.ax_regions = plt.subplot(grid_dG[1, 0])
        else:
            if project_type == 'normal':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol], sharex=self.ax_dG, sharey=self.ax_dG)
                self.ax_comb = plt.subplot(grid_comb[idx_mol, 0], sharex=self.ax_comb, sharey=self.ax_comb, projection='polar')
                

                #self.ax_comb = plt.subplot(grid_comb[0, idx_mol], sharex=self.ax_comb, sharey=self.ax_comb)
            elif project_type == 'inhibition':
                self.ax_dG = plt.subplot(grid_dG[0, idx_mol], sharey=self.ax_dG)
                #self.ax_comb = plt.subplot(grid_comb[0, idx_mol], sharey=self.ax_comb)
            elif project_type == 'water':
                self.ax_comb = plt.subplot(grid_comb[0, idx_mol])

            self.ax_regions = plt.subplot(grid_dG[1, idx_mol], sharey=self.ax_regions)
            
        if project_type == 'inhibition':
            if self.metrics:
                sub_grid = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = grid_comb[0, idx_mol], hspace=0, wspace=0, height_ratios=[4,10], width_ratios=[10,4])
                self.ax_metric = plt.subplot(sub_grid[1,0])
                if self.idx_mol == 0:
                    self.ax_comb = plt.subplot(sub_grid[1,1], sharey=self.ax_metric)
                    self.ax_comb2 = plt.subplot(sub_grid[0,0], sharex=self.ax_metric)
                else:
                    self.ax_comb = plt.subplot(sub_grid[1,1], sharey=self.ax_metric, sharex=self.ax_comb)
                    self.ax_comb2 = plt.subplot(sub_grid[0,0], sharex=self.ax_metric, sharey=self.ax_comb2)
            else:
                self.ax_comb = plt.subplot(grid_comb[0,idx_mol])
                self.ax_comb2 = plt.subplot(grid_comb[0,idx_mol])
        elif project_type == 'normal': 
            
            #self.ax_subunits2 = self.ax_subunits.twinx()
            if self.metrics:
                self.ax_metric = plt.subplot(grid_comb[idx_mol, 2])
        
        #elif project_type == 'water':
        #    if self.metrics:
        #        self.sub_w_grid_metrics = gridspec.GridSpecFromSubplotSpec(len(self.iterables),1, subplot_spec = grid_comb[:, self.idx_mol+1])
                

    def setGrids(self):
        
        p_type_specs = self.p_type
        project_type = self.project_type
        projects = self.projects
        
        self.fig=plt.figure(figsize=p_type_specs[project_type]['figsize'], constrained_layout=True) #
        (nrows, ncols) = p_type_specs[project_type]['grid_layout']
        
        outer_grid = gridspec.GridSpec(nrows, 
                                       ncols, 
                                       hspace=0.3, 
                                       height_ratios = p_type_specs[project_type]['outer_grid_heights'], 
                                       width_ratios=p_type_specs[project_type]['outer_grid_widths'])
        
        #TODO: use figure, subfigures  instead of gridspec to make axes sharing/handling MORE EASY
        if project_type == 'normal':
            grid_dG = gridspec.GridSpecFromSubplotSpec(2, len(projects), subplot_spec = outer_grid[0,:], hspace=0.1, wspace=0.1, height_ratios=[4,2]) 
            grid_comb = gridspec.GridSpecFromSubplotSpec(len(projects), 3, subplot_spec = outer_grid[2,:], width_ratios=[1,0.8,0.8], wspace=0.3) #, wspace=0.1) #wspace=0.2)
            grid_figure = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec = outer_grid[1,:], width_ratios=[1,0.8,0.8], wspace=0.3)
            self.grid_figure = grid_figure
            self.ax_scheme = plt.subplot(grid_figure[0,0])
        elif project_type == 'inhibition':
            grid_dG = gridspec.GridSpecFromSubplotSpec(2, len(projects), subplot_spec = outer_grid[0,:], hspace=0.1, wspace=0.1, height_ratios=[4,2]) 
            grid_comb = gridspec.GridSpecFromSubplotSpec(2, len(projects), subplot_spec = outer_grid[1,:], height_ratios=[10,2], wspace=0.2)
            #grid_figure = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer_grid[2,:], wspace=0.1)
        elif project_type == 'water':
            grid_dG = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec = outer_grid[0,:], height_ratios=[2,1], hspace=0.1, wspace=0.2)  
            grid_comb = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec = outer_grid[1,:], height_ratios=[6,5], wspace=0.2)
            
            sub_w_grid_dG = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = grid_dG[:,1:], height_ratios=[2,1], hspace=0.2, wspace=0.1)
            self.sub_w_grid_dG = sub_w_grid_dG
            #grid_figure = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = outer_grid[2,1:], wspace=0.3, height_ratios=[10,2])
            #grid_corr = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer_grid[2,:], wspace=0.1)      

        self.grid_dG = grid_dG
        self.grid_comb = grid_comb
        
        
        self.outer_grid = outer_grid
        
        
        

                        

    def plot_thermodynamics(self):
        figures = {}
        for project_type, projects in self.supra_project.items():
        

            print(project_type)
            input_states = self.input_states
            
            dG = self.dG
            combinatorial = self.combinatorial
            plot_specs = self.plot_specs
            if project_type != 'water':
                self.loc1, self.loc2 = Plots.loc1, Plots.loc2
            else:
                self.loc1, self.loc2 = 0, 250
            mol2 = self.mol2
            color2 = self.color2
            
            

                    
            
            
            supra_df = pd.DataFrame()
            
            self.projects : dict = projects
            self.project_type : str = project_type
            self.labels_regions = list(self.subset_states[project_type].keys())
            
            self.setGrids()
            
            
            if project_type == 'normal':
                
                self.plot_state_cartoon()
            
            for idx_mol, (mol, project) in enumerate(projects.items()):
                print(mol)
                

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
                self.steps = plot_specs[project_type][mol][0]
                self.color_mol = plot_specs[project_type][mol][1]
                self.linestyle = plot_specs[project_type]['linestyle'] 
                
                self.idx_mol = idx_mol
                self.mol = mol
                
                self.setSubplots()
                self.lns = []

                #ax_comb = plt.subplot(grid_comb[idx_mol])
                self.counts_states = np.empty([len(self.iterables), len(self.states)])
                self.count_regions = np.empty([len(self.iterables), len(self.steps)+1])
                
                
                if project_type == 'normal':
                    
                    subgrid_normal_subunits = gridspec.GridSpecFromSubplotSpec(len(self.iterables),1, subplot_spec = self.grid_comb[idx_mol, 2], hspace=0)
                    
                
                for idx, it in enumerate(self.iterables): 
                    self.it : str = it
                    self.idx_it = idx
                    
                    self.plot_wrapper(dG_df, combinatorial_df) 
                    
                            
        
                    if self.project_type == 'normal':
                        subunit_pair_labels = list(self.metric_data[project_type][mol][it].keys())
                        
                        #if idx_mol == 0:
                            #self.ax_subunits = plt.subplot(subgrid_normal_subunits[idx,0])
                        self.ax_subunits = plt.subplot(self.grid_comb[idx_mol, 1])
                        #else:
                            #self.ax_subunits = plt.subplot(subgrid_normal_subunits[idx,0], sharey=self.ax_subunits)
                        #    self.ax_subunits = plt.subplot(self.grid_comb[idx_mol, 2], sharey=self.ax_subunits)
                        self.plot_subunitsStateSampling(combinatorial_df, subunit_pair_labels)
                    
                    
                mol_df = pd.DataFrame(self.counts_states.T, index=self.states, columns=self.iterables)

                
                if project_type == 'normal':
                    self.plot_metrics((self.metric_data[project_type][mol], []))
                    self.base_combinatorial = self.plot_combinatorial_polar(mol_df)
                elif project_type == 'inhibition':
                    self.base_combinatorial = self.plot_combinatorial_polar(mol_df)
                    
                if project_type != 'water':
                    self.base_combinatorial = self.plot_combinatorial(mol_df)
                
                    

                    
                #NOTE: loading of mol1 and mol1 + mol2 is done above. Here is only to handle mol2
                if project_type == 'inhibition':

                    self.ax2 = self.ax_dG.twiny()
                    self.color_mol = [color2]
                    self.linestyle = 'dashed'
                    self.idx_it = 0
                    self.it = '5mM'
                    inib_mol = mol2[mol]

                    inib_df = pd.read_csv(f'{project.results}/binding_profile/binding_profile_acylSer-His_{inib_mol}_mean_b0_e-1_s1.csv', index_col=0)    
                    df_comb_inhib = combinatorial_df.loc[:, combinatorial_df.columns.get_level_values('l3') == self.it]

                    self.states_mol = self.double_comb_states['normal'][idx_state] 
                    print(self.states, self.states_mol)
                    
                    metric = self.plot_metrics((self.metric_data[project_type][mol], self.correlation_data[project_type][mol]))
                    if self.idx_mol == 1:
                    
                        _colorbar = plt.subplot(self.grid_comb[-1,:])
                        _colorbar.axis('off')
                        cbar = plt.colorbar(metric, ax=_colorbar, orientation='horizontal')
                        cbar.set_label(self.metric_method) 
                        
                        
                    self.plot_dg_inhib(inib_df)
                    self.plot_regions(df_comb_inhib)
                    
                    state_sampling_fraction = self.getStateSamplingFraction(df_comb_inhib, states=self.states_mol, w_subset=False)
                    #plot_combinatorial
                    self.ax_comb2.bar(self.states_mol.keys(), state_sampling_fraction, color=color2)
                    self.ax_comb2.set_yscale('log')
                    self.ax_comb2.tick_params(axis='y', which='major', width=2, length=4)
                    self.ax_comb2.tick_params(bottom=False, labelbottom=False)
                    self.ax_comb2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
                    self.ax_comb2.grid(linestyle='--', axis='y')
                    

            #NOTE: This is only loading the other _water dfs. The df of water has been handled before
            if project_type == 'water':

                

                self.ax_comb = plt.subplot(self.grid_comb[0, 0])
                
                
                for idx_mol_w, (mol_w, iterables_mol_w2) in enumerate(self.mol_water.items()):

                        #self.plot_correlations(self.discretized_trajectories[project_type][mol_w])  
                        
                    base_df_mol = dG[project_type]['H2O']
                    dG_mol_df = base_df_mol.iloc[:, base_df_mol.columns.get_level_values(0) == mol_w].droplevel(0, axis=1)
                    
                    
                    color_mol = plot_specs['normal'][mol_w][1] 
                    color_mol = np.vstack([color_mol, pl.cm.Oranges(np.linspace(0.5,0.5,1))])
                    
                    self.color_mol = color_mol
                    self.mol = mol_w
                    self.idx_mol = idx_mol_w 
                    
                    self.ax_dG = plt.subplot(self.sub_w_grid_dG[0, idx_mol_w])
                    self.ax_regions = plt.subplot(self.sub_w_grid_dG[1, idx_mol_w], sharey=self.ax_regions)
                    self.iterables = iterables_mol_w2[0] + [f'100mM_{iterables_mol_w2[1]}_5mM']
                    
                    self.counts_states = np.empty([len(self.iterables), len(self.states)])
                    self.count_regions = np.empty([len(self.iterables), len(self.steps)+1])
                        
                    if isinstance(input_states[project_type], tuple):
                        
                        #self.states = input_states[project_type][1]
                        self.states_mol = input_states['normal'][1]
                        double_combinatorial_states = self.double_comb_states['normal'][1]
                        comb_df = combinatorial[project_type][mol].replace(to_replace=input_states[project_type][0].keys(), value=self.states.keys())
                    else:
                        #self.states = input_states[project_type][0]
                        self.states_mol = input_states['normal'][0]
                        double_combinatorial_states = self.double_comb_states['normal'][0]
                        comb_df = combinatorial[project_type][mol]
                    comb_df = comb_df.iloc[:, comb_df.columns.get_level_values('l2') == mol_w]



                    if self.metrics:
                        sub_w_grid_metrics = gridspec.GridSpecFromSubplotSpec(len(self.iterables),1, subplot_spec = self.grid_comb[:, idx_mol_w+1])
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
                            
                            self.ax_metric = plt.subplot(sub_w_grid_metrics[idx_w,0])
                            combinatorial_state_fractions = self.double_comb_fractions[mol_w][it_w]
                            corr_image = self.plot_metrics((self.metric_data[project_type][mol_w][it_w], self.correlation_data[project_type][mol_w][it_w]),
                                                           comb_states=(double_combinatorial_states, combinatorial_state_fractions), 
                                                           color=self.color_mol[idx_w])
                            

                    
                        
                    w_count_states = np.vstack([mol_df.values.T, self.counts_states]) 
                    mol_water_df = pd.DataFrame(w_count_states.T, index=list(self.states.keys()), columns=['0 mM'] + self.iterables)
                    #self.plot_combinatorial(mol_water_df, opt_color=color_mol_comb)
                    
                    
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
                    
                    
                    fillstyle_mol = self.plot_specs['normal'][mol_w][2]
                    marker_mol = self.plot_specs['normal'][mol_w][3]
                    

                    self.ax_comb.plot(iterable_scalars, 
                                  mol_water_df.iloc[1,:-1], 
                                  marker=marker_mol,
                                  label=mol_w,
                                  color=self.plot_specs['inhibition'][mol_w][1][1])
                    self.ax_comb.plot(100.0, 
                                  mol_water_df.iloc[1,-1],  
                                  marker=marker_mol,
                                  color=self.color2)
                    self.ax_comb.legend()
                    self.ax_comb.set_ylabel(f'{self.states[1]} sampling fraction')
                    self.ax_comb.set_xlabel('concentration (mM)')      
                    
                    
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
        self.fig.tight_layout()
        self.fig.show()
        figures[project_type] = self.fig
        self.fig.savefig(f'{project.results}/figures/BindingProfile_combinatorial_{project_type}.png', dpi=600, bbox_inches='tight')
    
        return figures
               
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
        
        
        self.labels_regions = label_states
        
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
         
    