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
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib import colors as ml_colors
import numpy as np
import pandas as pd
import pickle
import nglview
import Trajectory
import tools_plots
import tools
import mdtraj as md
from mdtraj.utils import in_units_of
from simtk.unit import angstrom
from itertools import compress



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
        

    def plot_cg_TICA_contours(self, msm, macrostate, n_states, lag, file_name, name):
        """
        DEPRECATED. Plots distributions of states along 2 ICs instead of doing coarse masking. 
        Becomes cumbersome for systems with more than 6 macrostates.

        Parameters
        ----------
        msm : TYPE
            DESCRIPTION.
        macrostate : TYPE
            DESCRIPTION.
        n_states : TYPE
            DESCRIPTION.
        lag : TYPE
            DESCRIPTION.
        file_name : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        rows, columns=tools_plots.plot_layout(macrostate)
        pcca_plot, axes = plt.subplots(rows, columns, figsize=(8, 6), sharex=True, sharey=True)

        #loop through individual states
        for idx, ax in axes.flat:
#                 _=pyemma.plots.plot_contour(*tica_concat[:, :2].T, met_dist[dtrajs_concatenated], ax=ax, cmap='rainbow')
            _=pyemma.plots.plot_free_energy(*self.tica_concat[:, :2].T, msm.metastable_distributions[idx][self.dtrajs_concatenated],
                                       legacy=False, ax=ax, method='nearest') 
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
    
    
                    
# =============================================================================
#                 
#                 
#             mfpt_cg_table=pd.DataFrame([[msm.mfpt(msm.metastable_sets[i], msm.metastable_sets[j]) for j in range(macrostate)] for i in range(macrostate)], 
#                                        index=index_cg, columns=[c for c in np.arange(1, macrostate+1)])
#             #mfpt_cg_c=pd.concat([mfpt_cg_c, mfpt_cg_table], axis=0, sort=True)
#             print(mfpt_cg_table)
#             #sns.heatmap(mfpt_cg_table, linewidths=0.1, cmap='rainbow', cbar_kws={'label': r'rates (log $s^{-1}$)'})
#             #mfpt_cg_table.plot()
# =============================================================================
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

    norm = ml.colors.LogNorm(vmin=vmin, vmax=vmax)

    for im in images:
        im.set_norm(norm)
             
    print(f'limits: {vmin:e}, {vmax:e}')
         
    cbar=fig.colorbar(images[-1], ax=axes)
    cbar.set_label(label=r'MFPT (ps)', size='large')
    cbar.ax.tick_params(labelsize=12)
    for im in images:
        im.callbacksSM.connect('changed', update)
                        
    return images 