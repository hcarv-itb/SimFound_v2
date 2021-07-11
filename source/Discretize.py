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


class Discretize:
    """Base class to discretize features. Takes as input a dictionary of features, 
    each containing a dictionary for each parameter and raw_data"""
    
    def __init__(self, data, feature_name='feature', results=os.getcwd()):
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

        
        #print(f'Input feature: {self.feature_name} \nData: \n {self.data}')

    
    def combinatorial(self, 
                      shells,
                      level=3,
                      start=0,
                      stop=10,
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

        #TODO: make levels more human readable. Automate level name in df, then query directly for the name search
        levels=self.data.columns.levels

        shells_name='_'.join(str(shells))
        shells_name=str(shells)
        feature_df=pd.DataFrame()

        #TODO: consider multiproc
        for iterable in levels[level]:
            
            print(f'Iterable: {iterable}')
            
            #This line is gold
            iterable_df=self.data.loc[start:stop:stride,self.data.columns.get_level_values(f'l{level+1}') == iterable]
            
            #Get discretization from state mapper
            iterable_df_disc=tools.Functions.state_mapper(shells, data=iterable_df, labels=labels)  
            iterable_df_disc.rename(iterable, inplace=True)
            
            feature_df=pd.concat([feature_df, iterable_df_disc], axis=1) 

        #TODODODODODODODOD
        print(f'Combinatorial: \n {feature_df}')
        

        feature_df.name=f'{self.feature_name}_combinatorial'              
        #feature_df.to_csv(f'{self.results}/combinatorial_{self.feature_name}_discretized_{shells_name}.csv')
        
        self.discretized[self.feature_name]=('combinatorial', feature_df)
        
        return feature_df


    def shell_profile(self,  
                    thickness=0.5,
                    limits=(0,150),
                    level=2,
                    start=0,
                    stop=10,
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
                
        levels=self.data.columns.levels
                
        print(f'Generating shell profile between {start} and {stop} frames (stride {stride}).')
            
        for iterable in levels[level]:
            
            print(f'Iterable: {iterable}')
            
            #This line is gold
            iterable_df=self.data.loc[start:stop:stride,self.data.columns.get_level_values(f'l{level+1}') == iterable]            
            iterable_df_disc=pd.DataFrame(index=feature_center_bin[:-1], columns=iterable_df.columns)
            
            #print(iterable_df)

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
            
        #print(f'Shell profile: \n {feature_df}')
        
        self.discretized[self.feature_name]=('shell_profile', feature_df)
        
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

    @staticmethod
    def dG_calculation(input_df, 
                       mol='MeOH', 
                       bulk=(30, 41), 
                       level=2, 
                       resolution=0.5, 
                       feature_name=None, 
                       describe=None,
                       quantiles=[0.01,0.5,0.75,0.99],
                       results=os.getcwd()):
        
        """
        Function to normalize NAC values according to theoretic distribution in a shpere with bulk concentration "c". 
        Takes as inputs "c", a NAC histogram with standard errors and number of frames where NAC was calculated "length".
        Calculations are made with the  N_theo() function. 
        The probability of NAC for each bin is P_nac and P_nac_err. The theoretical distribution of NAC at "c" is P_t.
        The value of bulk "c" is adjusted by fitting the P_nac to the bulk region "ranges_bulk" using curve_fit(). 
        P_t_optimized uses the "c_opt" bulk concentration.
        dG is calculated from P_t_optimized.
        NOTE: The "ranges_bulk" should be carefully chosen for each system.
        
        TODO: Fix mol requirement. Ask project.
    
        """
	
        from scipy.optimize import curve_fit


        #ordered_concentrations=['50mM', '150mM', '300mM', '600mM', '1M', '2.5M', '5.5M']
        ordered_concentrations=['300mM', '1M', '2.5M', '5.5M']
        #Calculates the theoretical NAC values in "bulk" in a spherical shell of size given by "ranges". 
        #The output is N - 1 compared to N values in ranges, so an additional value of "resolution" is added for equivalence.
        #N_ref= lambda ranges, bulk_value: np.log(((4.0/3.0)*np.pi)*np.diff(np.power(np.append(ranges, ranges[-1] + resolution), 3.0))*6.022*bulk_value*factor)
        
        def N_ref(ranges, bulk_value):
        
            spherical=np.log(((4.0/3.0)*np.pi))
            sphere=np.diff(np.power(np.append(ranges, ranges[-1] + resolution), 3.0))
            c_normal=6.022*bulk_value*factor
            
            out=spherical*sphere*c_normal
                
            #print(out)

            return out*3 # The 3 factor Niels Hansen was talking about

        name_original=r'N$_{(i)}$reference (initial)'
        name_fit=r'N$_{(i)}$reference (fit)'

        df, pairs, replicas, molecules, frames=tools.Functions.get_descriptors(input_df, 
                                                                   level, 
                                                                   describe=describe,
                                                                   quantiles=quantiles)
        
        df.name=f'{feature_name}_{describe}' 
        #print(df)
        df.to_csv(f'{results}/{feature_name}_{describe}.csv')
        
        #print(df)
        
        #Find the element of nac_hist that is closest to the min and max of bulk
        ranges=df.index.values
        bulk_range=np.arange(bulk[0], bulk[1], resolution)        
        bulk_min, bulk_max=ranges[np.abs(ranges - bulk_range[0]).argmin()], ranges[np.abs(ranges - bulk_range[-1]).argmin()]
        
        dG_fits=pd.DataFrame()
        dG_fits.name=r'$\Delta$G'
        
        descriptors_iterables=input_df.columns.get_level_values(f'l{level+1}').unique()
        
        rows, columns, fix_layout=tools_plots.plot_layout(descriptors_iterables)
        fig_fits,axes_fit=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        fig_dG,axes_dG=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))

        for iterable, ax_fit, ax_dG in zip(ordered_concentrations, axes_fit.flat, axes_dG.flat): #descriptors_iterables

            try:
                bulk_value=float(str(iterable).split('M')[0]) #get the value (in M) of the concentration from the label of c.
                factor=0.0001
                unit='M'
            except:
                bulk_value=float(str(iterable).split('mM')[0])
                factor=0.0000001
                unit='mM'
            
            #print(f'The bulk value is: {bulk_value}\nThe factor is: {factor}')

            if describe == 'mean':
            
                #Define what is N(i)_enzyme
                N_enz=df[iterable]
                N_enz.name=f'N$_{{(i)}}$enzyme {iterable}'
              
                #Define what is N(i)_enzyme error
                N_enz_err=df[f'{iterable}-St.Err.']
                N_enz_err.name=f'N$_{{(i)}}$enzyme {iterable} St.Err.'
                
                #Note: This is inverted so that dG_err calculation keeps the form.
                N_error_p, N_error_m=N_enz-N_enz_err, N_enz+N_enz_err 
                
                #dG elements
                a=np.log(N_enz.values)
                a_m=np.log(N_error_m)
                a_p=np.log(N_error_p)
                
            elif describe == 'quantile':
                
                N_error_m=pd.DataFrame()
                N_error_p=pd.DataFrame()
                                
                for quantile in quantiles:
                    
                    #Define what is N(i)_enzyme
                    if quantile == 0.5:
                        
                        N_enz=df[f'{iterable}-Q0.5']     
                        N_enz.replace(0, np.nan, inplace=True)
                    
                    #Define what is N(i)_enzyme error
                    elif quantile < 0.5:                        
                        N_error_m=pd.concat([N_error_m, df[f'{iterable}-Q{quantile}']], axis=1)
                        #N_error_m.name=f'N$_{{(i)}}$enzyme {iterable} Q{quantile}'
                        #print(N_error_m)
                        
                    elif quantile > 0.5:
                        N_error_p=pd.concat([N_error_p, df[f'{iterable}-Q{quantile}']], axis=1)
                        #N_error_p.name=f'N$_{{(i)}}$enzyme {iterable} Q{quantile}'
                        #print(N_error_p)
                        #N_error_p=df[f'{iterable}-Q{quantiles[-1]}']
            
                N_error_m.replace(0, np.nan, inplace=True)
                N_error_p.replace(0, np.nan, inplace=True)
                
                #dG elements
                a=np.log(N_enz)
                a_m=np.log(N_error_m)
                a_p=np.log(N_error_p)

                    
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
                                     p0=bulk_value) #, bounds=(bulk_value - bulk_value*0.5, bulk_value + bulk_value*0.5))
            except:
                c_opt = (bulk_value, 0)
            fitted_bulk=c_opt[0]
            #stdev=np.sqrt(np.diag(c_cov))[0]
            #print(f'The fitted bulk value is: {c_opt[0]} +/- {stdev}')  
            
            #Recalculate N_ref with adjusted bulk concentration.
            N_opt=pd.Series(N_ref(ranges, fitted_bulk), index=N_enz.index.values, name=f'{name_fit} {iterable}')
            N_opt.replace(0, np.nan, inplace=True)

            #Calculate dG (kT)
            
            b=np.log(N_opt)  
                               
            #print(np.shape(a), np.shape(b), np.shape(a_m), np.shape(a_p))
            
            dG=pd.DataFrame({f'$\Delta$G {iterable}': np.negative(a - b)}) 
            dG_err_m=np.negative(a_m.subtract(b, axis='rows'))
            dG_err_p=np.negative(a_p.subtract(b, axis='rows'))
                        
            theoretic_df=pd.DataFrame({f'{name_original} {iterable}':N_t, 
                                       f'{name_fit} {iterable}':N_opt}, 
                                       index=N_enz.index.values)
            
            dG_fits=pd.concat([dG_fits, dG, dG_err_m, dG_err_p, N_enz, theoretic_df], axis=1)

            legends=[name_original, name_fit]

            ax_fit.plot(ranges, N_t, color='red', ls='--')
            ax_fit.plot(ranges, N_opt, color='black') 
            
            if describe == 'mean':
                
                #Plot fit            
                ax_fit.plot(ranges, N_enz, color='green')
                ax_fit.fill_betweenx(N_enz_fit, bulk_range[0], bulk_range[-1], color='grey', alpha=0.8)
                ax_fit.set_ylim(1e-4, 100)
                ax_fit.fill_between(ranges, N_error_m, N_error_p, color='green', alpha=0.3)
            
                #Plot dG
                ax_dG.plot(ranges, dG, color='green')
                ax_dG.fill_between(ranges, dG_err_m, dG_err_p, color='green', alpha=0.5)
                ax_dG.set_ylim(-4, 4)
                
                legends.append('N$_{(i)}$enzyme')
                legends.append('Bulk')
                legends_dG=[r'$\Delta$G']
                
                locs=(0.79, 0.3)
        
            if describe == 'quantile':
                
                #Plot fit   
                ax_fit.plot(ranges, N_enz, color='orange')
                ax_fit.set_ylim(1e-3, 200)
                
                legends.append(N_enz.name)
 
                for idx, m in enumerate(N_error_m, 1):
                    ax_fit.plot(ranges, N_error_m[m], alpha=1-(0.15*idx), color='green')
                    legends.append(m)
                for idx, p in enumerate(N_error_p, 1):
                    ax_fit.plot(ranges, N_error_p[p], alpha=1-(0.15*idx), color='red') 
                    legends.append(p)
                    
                #Plot dG
                ax_dG.plot(ranges, dG, color='orange')
                ax_dG.set_ylim(-9, 4)
                
                legends_dG=[f'{iterable}-Q0.5']
                
                for idx, m in enumerate(dG_err_m, 1):
                    ax_dG.plot(ranges, dG_err_m[m], alpha=1-(0.15*idx), color='green')
                    legends_dG.append(m)
                for idx, p in enumerate(dG_err_p, 1):
                    ax_dG.plot(ranges, dG_err_p[p], alpha=1-(0.15*idx), color='red') 
                    legends_dG.append(p)
                    
                locs=(0.79, 0.12)

            ax_fit.set_yscale('log')
            ax_fit.set_xscale('log')
            ax_fit.set_xlim(1,bulk_range[-1] + 10)
            ax_fit.set_title(f'{bulk_value} {unit} ({np.round(fitted_bulk, decimals=1)} {unit})', fontsize=10)
            
            ax_dG.axhline(y=0, ls='--', color='black')
            ax_dG.set_xlim(1,bulk_range[-1]+10)
            ax_dG.set_title(iterable, fontsize=10)
            ax_dG.set_xscale('log')
            
        #TODO: Change this for even number of iterables, otherwise data is wiped.    
        axes_fit.flat[-1].axis("off")
        axes_dG.flat[-1].axis("off")  
        
        fig_fits.legend(legends, loc=locs) # 'lower right')
        fig_fits.text(0.5, -0.04, r'Shell $\iti$ ($\AA$)', ha='center', va='center', fontsize=14)
        fig_fits.text(-0.04, 0.5, r'$\itN$', ha='center', va='center', rotation='vertical', fontsize=14)
        
        fig_dG.legend(legends_dG, loc=locs)
        fig_dG.text(0.5, -0.04, r'$\itd$$_{NAC}$ ($\AA$)', ha='center', va='center', fontsize=14)
        fig_dG.text(-0.04, 0.5, r'$\Delta$G (k$_B$T)', ha='center', va='center', rotation='vertical', fontsize=14)
        
        fig_fits.show()
        fig_dG.show()
        
        fig_fits.suptitle(f'Feature: {feature_name}\n{describe}')
        fig_dG.suptitle(f'Feature: {feature_name}\n{describe}')
        
        
        fig_fits.savefig(f'{results}/{feature_name}_{describe}_fittings.png', dpi=600, bbox_inches="tight")
        fig_dG.savefig(f'{results}/{feature_name}_{describe}_binding_profile.png', dpi=600, bbox_inches="tight")
        
        dG_fits.to_csv(f'{feature_name}-{describe}.csv')
        
        return dG_fits
    
    
    @staticmethod
    def plot(df, level=2, subplots=False):
        

        levels=df.columns.levels #Exclude last, the values of states
        
        for iterable in levels[level]: 
            
            print(iterable)
            df_it=df.loc[:,df.columns.get_level_values(f'l{level+1}') == iterable] #level +1 due to index starting at l1
            
            #print(df_it)
            
            print(df_it.columns.unique(level=2).tolist())
            #plot_it=
            df_it.plot(kind='line', subplots=subplots, sharey=True, title=iterable, figsize=(7,5), legend=False, sort_columns=True)
            #current_x=plt.xticks()[0] #the array 0
            #new_x=(current_x*5)/1000
            #plt.xticks(ticks=current_x[1:-1].astype(int), labels=new_x[1:-1])
            #plt.xlabel('Simulation time (ns)')
            
            #plots[iterable]=plot_it
            #plt.savefig(f'{self.results}/discretized_{df.name}_{iterable}.png', bbox_inches="tight", dpi=600)
        
            plt.show()
    
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
    
    
    
    
    
    
    
    
    