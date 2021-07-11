# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:11:07 2020

@author: hca
"""
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib import colors

#import base

try:
    import pyemma
except:
    
    print('some modules not loaded')

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
         
    