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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

class Discretize:
    """Base class to discretize features. Takes as input a dictionary of features, 
    each containing a dictionary for each parameter and raw_data"""
    
    
    def_locs=(1.35, 0.6)
    
    def __init__(self, data, results, scalars=[], feature_name='undefined'):
        """
        

        Parameters
        ----------

        data : df
            A DataFrame of features.
        feature_name : str
            The name of the feature.The default is feature
        results : path
            The path were to store results from methods.

        Returns
        -------
        None.

        """

        self.data=data
        self.feature_name=feature_name
        self.results=results
        self.discretized={}
        self.scalars=scalars


        
        #print(f'Input feature: {self.feature_name} \nData: \n {self.data}')

    
    def combinatorial(self, 
                      shells,
                      level='ligand',
                      start=0,
                      stop=-1,
                      stride=1,
                      labels=None):
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


        if start == 0:
            start=self.data.index[0]

        if stop == -1:
            stop=self.data.index[-1]
            
        if stride == 1:
            
            stride=int((self.data.index[-1]-self.data.index[0])/(len(self.data.index.values)-1))
               
        levels=self.data.columns.levels

        #TODO: make levels more human readable. Automate level name in df, then query directly for the name search
        if level == 'ligand':
            level = 2


        shells_name='_'.join(str(shells))
        shells_name=str(shells)
        feature_df=pd.DataFrame()

        #TODO: consider multiproc on sub_df

        for iterable in levels[level]:
            
            print('Iterable: ', iterable)
            #This line is gold
            iterable_df=self.data.loc[start:stop:stride,self.data.columns.get_level_values(f'l{level+1}') == iterable]

            iterable_df_disc=pd.DataFrame()
            sub_iterable=iterable_df.columns.get_level_values(f'l{level+2}').unique()
            for i in sub_iterable:         
                sub_df = iterable_df.loc[:,iterable_df.columns.get_level_values(f'l{level+2}') == i]
                print('\tSub-level:', i)
                
                #Get only the matching iterable/subiterable combination. +2 for levels above (parameter, replicate)
                indexes, names=[], []
                for l in range(level+2):
                
                    names.append(sub_df.columns.levels[l].name)
                    value=sub_df.columns.levels[l].values

                    if l == level: #the iterable
                        value=value[value == iterable]
                    elif l == level+1: #the sublevel
                        value=value[value == i]
                    indexes.append(value)
                    
                column_index=pd.MultiIndex.from_product(indexes, names=names) 

                rows=pd.Index(iterable_df.index.values, name=self.data.index.name)

                state_series=tools.Functions.state_mapper(shells, 
                                                          sub_df, 
                                                          labels=labels)
                
                iterable_replicate_disc=pd.DataFrame(state_series, columns=column_index, index=iterable_df.index) #rows)
                iterable_df_disc=pd.concat([iterable_df_disc, iterable_replicate_disc], axis=1)
                
            feature_df=pd.concat([feature_df, iterable_df_disc], axis=1) 
            
        print(f'Combinatorial: \n {feature_df}')
             
        feature_df.to_csv(f'{self.results}/combinatorial_{self.feature_name}_{shells_name}.csv')
        self.discretized[f'combinatorial_{self.feature_name}_{shells_name}']=feature_df
        
        return feature_df


    def shell_profile(self,  
                    thickness=0.5,
                    limits=(0,150),
                    level=2,
                    start=0,
                    stop=-1,
                    stride=1,
                    n_cores=-1):
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

        #TODO: Go get the df of the system instead of passing full dataframe. 
        #Requires changing procedure.
        #NOTE: Needs to be consistent with Feature manipulation.
        
        feature_range=np.arange(limits[0], limits[1],thickness)
        feature_center_bin=feature_range+(thickness/2)                
        feature_df=pd.DataFrame()

        print(self.data.index)
        if stop == -1:
            stop=self.data.index[-1]
               
        levels=self.data.columns.levels
                
        print(f'Generating shell profile between {start} and {stop} (stride {stride}).')
            
        for iterable in levels[level]:
            
            print(f'Iterable: {iterable}')
            #This line is gold
            iterable_df=self.data.loc[start:stop:stride, self.data.columns.get_level_values(f'l{level+1}') == iterable]            
            iterable_df_disc=pd.DataFrame(index=feature_center_bin[:-1], columns=iterable_df.columns)
            values_=iterable_df_disc.columns
            values=[(iterable_df[value], value) for value in values_]
            
            fixed=(feature_range, feature_center_bin[:-1])

            shells=tools.Tasks.parallel_task(Discretize.shell_calculation, 
                                             values, 
                                             fixed, 
                                             n_cores=n_cores)


            for shell in shells:

                iterable_df_disc=pd.concat([iterable_df_disc, shell], axis=1)
            
            iterable_df_disc.dropna(axis=1, inplace=True)
            
            feature_df=pd.concat([feature_df, iterable_df_disc], axis=1)
       
        self.discretized[f'ShellProfile_{self.feature_name}']=feature_df
        
        print('Discretization updated: ', [discretized for discretized in self.discretized.keys()])
        feature_df.to_csv(f'{self.results}/shellProfile.csv')
        return feature_df
    
    
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
                       input_df=None,
                       start=0,
                       stop=-1,
                       stride=1,
                       n_cores=-1, 
                       bulk=(30, 41), 
                       level=2, 
                       resolution=0.5,
                       shells=[],
                       feature_name=None, 
                       describe='mean',
                       quantiles=[0.01,0.5,0.75,0.99]):
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

        print(type(input_df))
        if type(input_df) == 'NoneType':
             try:
                 
                 input_df = self.discretized[f'ShellProfile_{self.feature_name}']
                 print('Shell profile found')
             except:
                 print('\tWarning! Shell profile not found. Calculating one with default parameters.')
                 input_df = self.shell_profile(start=start, stop=stop, stride=stride, n_cores=n_cores) 

        print(input_df)

        #TODO: FIX THIS
        ordered_concentrations=['50mM', '150mM', '300mM', '600mM', '1M', '2.5M', '5.5M']
        ordered_concentrations=['50mM', '300mM', '1M', '5.5M']
            
        name_original=r'N$_{(i)}$reference (initial)'
        name_fit =r'N$_{(i)}$reference (fit)'
        input_df.name=f'{feature_name}_{describe}' 
        
        #Find the element of nac_hist that is closest to the min and max of bulk
        ranges=input_df.index.values
        bulk_range=np.arange(bulk[0], bulk[1], resolution)        

        iterables=input_df.columns.get_level_values(f'l{level+1}').unique()
        #TODO: FIX THIS 2
        #iterables=ordered_concentrations
        scalars = tools.Functions.setScalar(iterables, ordered=True, get_uniques=True)
        
        #TODO: get shell labels and limits from shell_profile. also in dG_plot.
        labels=['A', 'P', 'E', 'S', 'B']

        
        
        def bulk_fitting():
            
            rows, columns=tools_plots.plot_layout(iterables)
            fig_fits,axes_fit=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(6,6))
            
            try:
                subplots=axes_fit.flat
            except AttributeError:
                subplots=axes_fit
                subplots=[subplots]
            
            
            legends = [name_original, name_fit]
            
            for iterable, ax_fit in zip(iterables, subplots):
                
                #TODO: Store and send to dG to avoid double calc. Dict.
                #Important stuff going on here.
                
                iterable_df = tools.Functions.get_descriptors(input_df, 
                                                              level, 
                                                              iterable, 
                                                              describe=describe, 
                                                              quantiles=quantiles)
                N_enz, N_error_p, N_error_m  = self.get_sim_counts(iterable_df, iterable, quantiles, describe)            
                N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit = self.get_fittings(iterable, 
                                                                                                 N_enz, 
                                                                                                 ranges, 
                                                                                                 bulk_range, 
                                                                                                 resolution, 
                                                                                                 name_fit)
    
                ax_fit.plot(ranges, N_t, label=r'N$_{(i)}$reference (initial)', color='red', ls='--')
                ax_fit.plot(ranges, N_opt, label=r'N$_{(i)}$reference (fit)', color='black')
    

                if describe == 'mean':
                       
                    ax_fit.plot(ranges, N_enz, label='N$_{(i)}$enzyme', color='green')
                    ax_fit.fill_betweenx(N_enz_fit, bulk_range[0], bulk_range[-1], label='Bulk', color='grey', alpha=0.8)
                    ax_fit.set_ylim(1e-4, 100)               
                    ax_fit.fill_between(ranges, N_error_m, N_error_p, color='green', alpha=0.3)
                    locs=(0.9, 0.13)
                
                elif describe == 'quantile':
                        
                    ax_fit.plot(ranges, N_enz, label='N$_{(i)}$enzyme', color='orange')
                    ax_fit.set_ylim(1e-3, 200)
                    
                    for idx, m in enumerate(N_error_m, 1):
                        ax_fit.plot(ranges, N_error_m[m], alpha=1-(0.15*idx), label=m, color='green')
                        legends.append(m)
                    for idx, p in enumerate(N_error_p, 1):
                        ax_fit.plot(ranges, N_error_p[p], label=p, alpha=1-(0.15*idx), color='red') 
                        legends.append(p)
                    locs=(0.79, 0.12)
                
                
                #ax_fit.grid()        
                ax_fit.set_yscale('log')
                ax_fit.set_xscale('log')
                ax_fit.set_xlim(1,bulk_range[-1] + 10)
                ax_fit.set_title(f'{iterable} ({np.round(fitted_bulk, decimals=1)} {unit})', fontsize=12)
    
            if describe == 'mean':
                legends.append('N$_{(i)}$enzyme')
                legends.append('Bulk')
            if describe == 'quantile':
                legends.append(N_enz.name)
            
            if not axes_fit.flat[-1].lines: 
                axes_fit.flat[-1].set_visible(False)
            
            if len(axes_fit.flat) > 7 and (len(axes_fit.flat % 2)) != 0:
                axes_fit.flat[-1].axis("off")
                locs=(0.9, 0.5)
                
            handles, labels = ax_fit.get_legend_handles_labels()
            fig_fits.subplots_adjust(wspace=0) #wspace=0, hspace=0
            fig_fits.legend(handles, labels, bbox_to_anchor=Discretize.def_locs) #legends, loc=locs)
            fig_fits.text(0.5, -0.04, r'Shell $\iti$ ($\AA$)', ha='center', va='center', fontsize=12)
            fig_fits.text(-0.04, 0.5, r'$\itN$', ha='center', va='center', rotation='vertical', fontsize=12)
            fig_fits.suptitle(f'Feature: {feature_name}\n{describe}')
            fig_fits.tight_layout()
            fig_fits.show()
            fig_fits.savefig(f'{self.results}/{feature_name}_{describe}_fittings.png', dpi=600, bbox_inches="tight")


# =============================================================================
# fig.legend([l1, l2, l3, l4],     # The line objects
#            labels=line_labels,   # The labels for each line
#            loc="center right",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            title="Legend Title"  # Title for the legend
#            )
# =============================================================================
        
        
        def dG_plot():
            
            dG_fits=pd.DataFrame()
            dG_fits.name=r'$\Delta$G'
            

            
            
            ord_scalars=[]
            for o in ordered_concentrations:
                for scalar, iterables in scalars.items():
                    #print(scalar, iterables)
                    if iterables[0] == o:
                        ord_scalars.append((scalar, iterables))
            #print(ord_scalars)
            
            rows, columns=tools_plots.plot_layout(ord_scalars)
            fig_dG,axes_dG=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(6,6))
            
            
            for (scalar, iterables), ax_dG in zip(ord_scalars, axes_dG.flat): #scalars.items()
                
                legends_dG = []
                for iterable in iterables:
                    
                    #Important stuff going on here.
                    iterable_df = tools.Functions.get_descriptors(input_df, level, iterable, describe=describe, quantiles=quantiles)
                    N_enz, N_error_p, N_error_m  = self.get_sim_counts(iterable_df, iterable, quantiles, describe)
                    N_opt, N_enz_fit, N_t, bulk_value, fitted_bulk, factor, unit = self.get_fittings(iterable, 
                                                                                                     N_enz, 
                                                                                                     ranges, 
                                                                                                     bulk_range, 
                                                                                                     resolution, 
                                                                                                     name_fit)
                    #Calculate dG (kT)
                    (a, a_p, a_m, b) = [np.log(i) for i in [N_enz, N_error_p, N_error_m, N_opt]]                

                    dG=pd.DataFrame({f'$\Delta$G {iterable}': np.negative(a - b)}) 
                    dG_err_m=np.negative(a_m.subtract(b, axis='rows'))
                    dG_err_p=np.negative(a_p.subtract(b, axis='rows'))                            
                    
                    theoretic_df=pd.DataFrame({f'{name_original} {iterable}':N_t, 
                                               f'{name_fit} {iterable}':N_opt}, 
                                               index=N_enz.index.values)                
                    dG_fits=pd.concat([dG_fits, dG, dG_err_m, dG_err_p, N_enz, theoretic_df], axis=1)    
                    
                    if describe == 'mean':
                        
                        ax_dG.plot(ranges, dG, color='green')
                        ax_dG.fill_between(ranges, dG_err_m, dG_err_p, alpha=0.5, color='green')
                        ax_dG.set_ylim(-4, 4)
                        legends_dG.append(['shells'])
                        
# =============================================================================
#                         ax_dG.vlines([2.25,4.5,8,10,25], 
#                                      -4, 
#                                      4, 
#                                      linestyle='dashdot', 
#                                      colors=['darkorange', 'black', 'dimgray', 'silver', 'lightgray'], #
#                                      label=labels)
# =============================================================================
                        
                        
                        #legends_dG.append(labels)
                        locs=(0.85, 0.3)
                
                    elif describe == 'quantile':
                        
                        ax_dG.plot(ranges, dG, color='orange')
                        ax_dG.set_ylim(-9, 4)
                        
                        legends_dG.append(iterable)
                        legends_dG=[f'{iterable}-Q0.5']
                        
                        for idx, m in enumerate(dG_err_m, 1):
                            ax_dG.plot(ranges, dG_err_m[m], alpha=1-(0.15*idx)) #, color='green')
                            legends_dG.append(m)
                        for idx, p in enumerate(dG_err_p, 1):
                            ax_dG.plot(ranges, dG_err_p[p], alpha=1-(0.15*idx)) #, color='red') 
                            legends_dG.append(p)
                            
                        #locs=(0.79, 0.12)
                    legends_dG.append(iterable)
                
                #ax_dG.grid()    
                #ax_dG.legend() #legends_dG) #, loc=locs)
                ax_dG.axhline(y=0, ls='--', color='black')
                ax_dG.set_xlim(1,bulk_range[-1]+10)
                ax_dG.set_title(iterable, fontsize=10)
                ax_dG.set_xscale('log')
                
                if len(shells):
                    ax_dG.vlines(shells, color='grey', linewidth=1)
              
            #TODO: Change this for even number of iterables, otherwise data is wiped.    
            if len(axes_dG.flat) > 7 and (len(axes_dG.flat % 2)) != 0:
                axes_dG.flat[-1].axis("off")
                locs=(0.75, 0.4)
            
                
            handles, labels = ax_dG.get_legend_handles_labels()
            fig_dG.legend(handles, labels, bbox_to_anchor=Discretize.def_locs)
            #fig_dG.legend(labels, loc=locs) #legends_dG, loc=locs)
            #fig_dG.subplots_adjust(wspace=0, hspace=0)
            fig_dG.text(0.5, -0.04, r'$\itd$$_{NAC}$ ($\AA$)', ha='center', va='center', fontsize=12)
            fig_dG.text(-0.04, 0.5, r'$\Delta$G ($\it{k}$$_B$T)', ha='center', va='center', rotation='vertical', fontsize=12)
            fig_dG.suptitle(f'Feature: {feature_name}\n{describe}')
            
            fig_dG.savefig(f'{self.results}/binding_profile_{describe}.png', dpi=600, bbox_inches="tight")
            fig_dG.show()
            dG_fits.to_csv(f'{self.results}/dGProfile_-{describe}.csv')
            #print(dG_fits)
            
            return dG_fits, fig_dG
        
        
                        
        bulk_fitting()
        dG_fits, fig_dG=dG_plot()
        fig_dG.show()

        fig_dG.savefig(f'{self.results}/binding_profile_{describe}.png', dpi=600, bbox_inches="tight")
        
        self.discretized[f'dGProfile_{self.feature_name}_{describe}']=dG_fits
        
        print('Discretization updated: ', [discretized for discretized in self.discretized.keys()])
        
        return dG_fits
    
    @staticmethod        
    def get_fittings(iterable, N_enz, ranges, bulk_range, resolution, name_fit):
        
        from scipy.optimize import curve_fit
        
        bulk_min, bulk_max=ranges[np.abs(ranges - bulk_range[0]).argmin()], ranges[np.abs(ranges - bulk_range[-1]).argmin()]

        #get the value (in M) of the concentration from the string
        try:
            bulk_value=float(str(iterable).split('M')[0]) 
            factor=1e-4
            unit='M'
        except:
            bulk_value=float(str(iterable).split('mM')[0])
            factor=1e-7
            unit='mM'
        #print(f'The bulk value is: {bulk_value} \nThe factor is: {factor}')
        
        
        def N_ref(ranges, bulk_value):

            spherical=np.log(((4.0/3.0)*np.pi))
            sphere=np.diff(np.power(np.append(ranges, ranges[-1] + resolution), 3.0))
            c_normal=6.022*bulk_value*factor
            
            out=spherical*sphere*c_normal
                
            #print(out)

            return out*3 # The 3 factor Niels Hansen was talking about
        
        
        #Region of P_nac values to be used for fitting. 
        #Based on defined values for bulk
        idx=list(ranges)
        N_enz_fit=N_enz.iloc[idx.index(bulk_min):idx.index(bulk_max)].values
        
        #original theoretical distribution with predefined bulk
        N_t=N_ref(ranges, bulk_value)   

        #Optimize the value of bulk with curve_fit of nac vs N_ref. 
        #Initial guess is "bulk" value. Bounds can be set as  +/- 50% of bulk value.
        try:

            c_opt, c_cov = curve_fit(N_ref, 
                                 bulk_range[:-1], 
                                 N_enz_fit, 
                                 p0=bulk_value)
                                 #bounds=(bulk_value - bulk_value*0.5, bulk_value + bulk_value*0.5))
            stdev=np.sqrt(np.diag(c_cov))[0]
        except:
            c_opt = bulk_value
            stdev = 0
        
        fitted_bulk=c_opt[0]
        #print(f'The fitted bulk value is: {fitted_bulk} +/- {stdev}')  
        
        #Recalculate N_ref with adjusted bulk concentration.
        N_opt=pd.Series(N_ref(ranges, fitted_bulk), index=N_enz.index.values, name=f'{name_fit} {iterable}')
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
            
            N_a=N_enz
            
        elif describe == 'quantile':
            
            N_error_m=pd.DataFrame()
            N_error_p=pd.DataFrame()
                            
            for quantile in quantiles:
                
                #Define what is N(i)_enzyme
                if quantile == 0.5:
                    
                    N_enz=iterable_df[f'{iterable}-Q0.5']     
                    N_enz.replace(0, np.nan, inplace=True)
                
                #Define what is N(i)_enzyme error
                elif quantile < 0.5:                        
                    N_error_m=pd.concat([N_error_m, iterable_df[f'{iterable}-Q{quantile}']], axis=1)
                    #N_error_m.name=f'N$_{{(i)}}$enzyme {iterable} Q{quantile}'
                    #print(N_error_m)
                    
                elif quantile > 0.5:
                    N_error_p=pd.concat([N_error_p, iterable_df[f'{iterable}-Q{quantile}']], axis=1)
                    #N_error_p.name=f'N$_{{(i)}}$enzyme {iterable} Q{quantile}'
                    #print(N_error_p)
                    #N_error_p=df[f'{iterable}-Q{quantiles[-1]}']
        
            N_error_m.replace(0, np.nan, inplace=True)
            N_error_p.replace(0, np.nan, inplace=True)
            
            N_a=N_enz
            
            
        return N_a, N_error_p, N_error_m

    
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
    
    
    
    
    
    
    
    
    