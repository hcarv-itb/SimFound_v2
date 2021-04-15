# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:34:59 2021

@author: hcarv
"""


import os
import nglview
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib notebook

from gridData import Grid

class Visualization:
    """
    Class to visualize trajectories
    """
    
    def __init__(self, results):
        self.results=results

        iso_original=30

    def density_data(self, dens, quantiles=np.arange(0,1.05, 0.05)):
    
        g = Grid(dens)
        g_max=np.max(g.grid.flat)
        g_min=np.min(g.grid.flat)
        g_mean=np.median(g.grid.flat)
        g_std=np.std(g.grid.flat)
    
    
    
        filtered=g.grid[g.grid > 0.001]

        g_q=np.quantile(filtered.flat, q=quantiles)

        print(f'Mean: {g_mean}\nSt. dev: {g_std}\nMax: {g_max}\nMin: {g_min}')
    
        return (quantiles, g_q, g.grid, g_mean, g_std, g_max, filtered)

    def density_plot_quantile(self, g_stats, concentration, iterable, states, iso, stat_dist=None):

        for state in states:
            if state[1] == iterable:
                state_name = state[0]
    
        x, y=g_stats[0], g_stats[1] #The quantile tuple
    
        plt.plot(x, y, marker='o') 
        #plt.axhline(y=iso)
        plt.axhline(y=stat_dist, ls='--', color='grey')
        plt.axhline(y=g_stats[3], color='black')
        plt.axhline(y=g_stats[3]+g_stats[4], color='red')
        plt.axhline(y=g_stats[3]+2*g_stats[4], color='red', ls='--')
    
        plt.title(f'{concentration} : {state_name}')
        plt.ylim(0.001, g_stats[5])
        plt.yscale('log', nonposy='clip')
        plt.show()
    
    
        return iso

    def density_plot_quantile_full(self, g_stats, concentration, iso):

    
        x, y=g_stats[0], g_stats[1] #The quantile tuple
    
        plt.plot(x, y, marker='o') 
        plt.axhline(y=iso)
        #plt.axhline(y=stat_dist, ls='--', color='grey')
        plt.axhline(y=g_stats[3], color='black')
        plt.axhline(y=g_stats[3]+g_stats[4], color='red')
        plt.axhline(y=g_stats[3]+2*g_stats[4], color='red', ls='--')
        
        plt.title(f'{concentration}')
        plt.ylim(0.001, g_stats[5])
        plt.yscale('log', nonposy='clip')
        plt.show()
    
    
        return iso

    def density_plot_hist(self, g_stats, concentration, iterable, state_name, iso, stat_dist=None):

    
        plt.hist(g_stats[6].flat, bins=500, density=True) #2
        
        plt.axvline(x=stat_dist, ls='--', color='grey')
        #plt.axvline(x=iso, color='green')
        plt.axvline(x=g_stats[3], color='black')
        plt.axvline(x=g_stats[3]+2*g_stats[4], color='red', ls='--')
        plt.axvline(x=g_stats[3]+g_stats[4], color='red')
        plt.title(f'{concentration} : {state_name}')
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
    
    
        return iso



    def plot_densityStats(self, figures, results, stride, states, stat_dist=None):



        iso_levels={}
        for c, it in figures.items():
    
            concentration=c
            iterables=it
    
            iso_iterables=[]
            for iterable in iterables:
            
                for state in states:
                    if state[1] == iterable:
                        state_name = state[0]
            
                stat_dist_it=stat_dist.loc[state_name, concentration]
                print(f'Stat Dist: {stat_dist_it}')

            
        
                dens=os.path.abspath(f'{results}/superposed_{concentration}-it{iterable}-s{stride}-Molar.dx')
                g_stats_=self.density_data(dens)

                print('Extracting the -2 quantile')
                iso_iterables.append(g_stats_[1][-2]) #-2 highest quantile
                #g_stats_[3]+g_stats_[4]) #The iso value as heuristic of stats

                plot=interact(self.density_plot_quantile, 
                              concentration=fixed(concentration),
                              iterable=fixed(iterable),
                              states=fixed(states),
                              state_name=fixed(state_name),
                              g_stats=fixed(g_stats_),
                              stat_dist=fixed(stat_dist_it),
                              iso = widgets.IntSlider(value=self.iso_original),
                              min=np.min(g_stats_[1]), 
                              max=1000, 
                              step=1)
        
                plot
        
            iso_levels[concentration]=iso_iterables

        return iso_levels


    def plot_densityStats_full(self, figures, results, stride, stat_dist=None):

        iso_levels={}
        for c in figures.keys():
    
            concentration=c
    
            iso=[]
        
            dens=os.path.abspath(f'{results}/superposed_{concentration}-s{stride}-Molar.dx')
            g_stats_=self.density_data(dens)
            print(g_stats_)

            print('Extracting the -2 quantile')
            iso.append(g_stats_[1][-2]) #-2 highest quantile
            #g_stats_[3]+g_stats_[4]) #The iso value as heuristic of stats

            plot=interact(density_plot_quantile_full, 
                          concentration=fixed(concentration),
                          g_stats=fixed(g_stats_),
                          iso = widgets.IntSlider(value=iso_original),
                          min=fixed(np.min(g_stats_[1])), 
                          max=fixed(1000), 
                          step=1)
        
            plot
        
            iso_levels[concentration]=iso

        return iso_levels




###############################################################################################################



    def viewer(self, concentration, iterable, stride, states, results, density=False, multiple=False, iso=1):

        struct=os.path.abspath(f'{results}/superposed_{concentration}-it{iterable}-s{stride}-clusters.pdb')
        struct_raw=os.path.abspath(f'{results}/superposed_{concentration}-it{iterable}-s{stride}.pdb')
        dens=os.path.abspath(f'{results}/superposed_{concentration}-it{iterable}-s{stride}-Molar.dx')

        if (os.path.exists(struct) or os.path.exists(struct_raw)):

            for state in states:
                if state[1] == iterable:
                    state_name = state[0]
                
            if density == True:

                view=view_presets_density(struct, dens, iso)
        
            elif multiple == True:
            
                view=view_presets_multiple(struct_raw)
        
            else:
                print('here')
        
            return view, state_name
    
        else:
            print(f'File not found for {concentration} and state {state_name} ({iterable}).')
        

    def viewer_full(self, concentration, stride, results, iso=1):

        struct=os.path.abspath(f'{results}/superposed_{concentration}-s{stride}-clusters.pdb')
        dens=os.path.abspath(f'{results}/superposed_{concentration}-s{stride}-Molar.dx')

        if (os.path.exists(struct) and os.path.exists(dens)):
        
            view=self.view_presets_density(struct, dens, iso)

            return view
    
        else:
            print(f'File not found for {concentration}.')
        
    def view_presets_density(self, struct, dens, iso):
        
        view = nglview.show_file(struct, default=False)
    
        view.add_component(dens)
    
        #Set color schemes
        colors={'normal':'whitesmoke', 
                'helix5': 'royalblue', 
                'helix10':'mediumspringgreen', 
                'density':'plum',
                'activeSite':'darkorange'}

        #Set residues ranges
        normal=['1-101', '109-139', '148-223', '226-266', '288-317']
        activeSite=['103-107', '223-226']
        
        #Representations
        for res in normal:
            view.add_cartoon(res, color=colors['normal'], opacity=0.3)
        for res in activeSite:
            view.add_cartoon(res, color=colors['activeSite'])
            
            view.add_cartoon('139-148', color=colors['helix5'])
            view.add_cartoon('266-288', color=colors['helix10'])
        
        view.add_licorice('224', color=colors['activeSite'])
        view.add_licorice('105', color=colors['activeSite'])
        
        ##surface density
        view.clear_representations(component=1)
        view.add_surface(component=1, 
                             color=colors['density'],  
                             isolevelType='level',
                             isolevel=iso)

        view.center()
        view.stage.set_parameters(**{
                                "clipNear": 0, "clipFar": 100, "clipDist": 10,
                                "fogNear": 20, "fogFar": 100,
                                "backgroundColor": "white",})   
        return view


    def view_presets_multiple(struct, traj):
        #print(f'Image for {concentration} and state {state_name} ({iterable}), iso {iso}.')
     
        if os.path.exists(struct):
            view = nglview.show_file(struct, default=False)
        else:
            print(f'Structure file not found: {struct}')   
        
        
        view.add_component(traj)
        #Set color schemes
        colors={'normal':'whitesmoke', 
                'helix5': 'royalblue', 
                'helix10':'mediumspringgreen', 
                'density':'plum',
                'activeSite':'darkorange'}

        #Set residues ranges
        normal=['1-101', '109-139', '148-223', '226-266', '288-317']
        activeSite=['103-107', '223-226']
        muts=['140', '285', '278']
        
        #Representations
        for res in normal:
            view.add_cartoon(res, color=colors['normal'], opacity=1)
        for res in activeSite:
            view.add_cartoon(res, color=colors['activeSite'])
            
        view.add_cartoon('139-148', color=colors['helix5'])
        view.add_cartoon('266-288', color=colors['helix10'])
        
        view.add_licorice('224') #, color=colors['activeSite'])
        view.add_licorice('105') #, color=colors['activeSite'])
    
        view.add_ball_and_stick('MeO') #, color=colors['activeSite'])
    
        for res in muts:
            view.add_licorice(res) #, color=colors['density'])
        
        view.center()
        view.stage.set_parameters(**{
                                "clipNear": 0, "clipFar": 100, "clipDist": 10,
                                "fogNear": 20, "fogFar": 100,
                                "backgroundColor": "white",})   
        return view



    def graphical(self, visualizations):
    
        c=input('Concentration: ')
        it=input('iterable: ')
    
        try:
            iso_=visualizations[f'{c}-{it}'][0]
            print('Isolevel: ', iso_)
            view=visualizations[f'{c}-{it}'][1]
            file=visualizations[f'{c}-{it}'][2]
            name=visualizations[f'{c}-{it}'][3]
            print('Name: ', name)
        
            view
        
            return view, file
        
        except KeyError:
            print('Concentration or iterable not defined.')

    def graphical_multiple(self, visualizations):
    
        c=input('Concentration: ')
        it=input('iterable: ')
    
        try:
            view=visualizations[f'{c}-{it}'][0]
            file=visualizations[f'{c}-{it}'][1]
            name=visualizations[f'{c}-{it}'][2]
            print('Name: ', name)
        
            view
        
            return view, file
        
        except KeyError:
            print('Concentration or iterable not defined.')

    def graphical_full(self, visualizations):
    
        c=input('Concentration: ')
    
        try:
            iso_=visualizations[f'{c}'][0]
            print('Isolevel: ', iso_)
            view=visualizations[f'{c}'][1]
        
            print(view)
            file=visualizations[f'{c}'][2]
        
            return view, file
        
        except KeyError:
            print('Concentration not defined.')
        
    def get_visualizations(self, figures, isos, states, stride, results):
    
        visualizations={}

        for concentration, it in figures.items():
    
            for iterable in it:

                iso_=isos[concentration][figures[concentration].index(iterable)]
        
                view, name=self.viewer(concentration, iterable, stride, states, results, iso=iso_)
                file=(f'superposed_{concentration}-it{iterable}-s{stride}-iso{np.round(iso_, decimals=2)}.png')
    
                visualizations[f'{concentration}-{iterable}']=[iso_, view, file, name]
    
        return visualizations



    def get_visualizations_multiple(self, figures, states, stride, results):
    
        visualizations={}

        for concentration, it in figures.items():
    
            for iterable in it:
        
                view, name=self.viewer(concentration, iterable, stride, states, results, density=False, multiple=True) #iso
                file=(f'superposed_{concentration}-it{iterable}-s{stride}.png')
    
                visualizations[f'{concentration}-{iterable}']=[view, file, name]
    
        return visualizations


    def get_visualizations_full(self, figures, isos, stride, results):
    
        visualizations={}

        for concentration in figures.keys():
    
            iso_=isos[concentration][0]
        
            view=self.viewer_full(concentration, stride, results, iso=iso_)
            file=(f'superposed_{concentration}-s{stride}-iso{np.round(iso_, decimals=2)}.png')
    
            visualizations[concentration]=[iso_, view, file]
        return visualizations
