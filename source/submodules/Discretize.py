# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:59:08 2021

@author: hcarv
"""

#Workflow modules
import tools
import tools_plots

#python modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Discretize:
    """Base class to discretize features. Takes as input a dictionary of features, 
    each containing a dictionary for each parameter and raw_data"""
    
    def __init__(self, systems, results, feature=None):
        """
        

        Parameters
        ----------
        systems : dict
            The dictionary of systems.
        features : array
            The array of feature dictionaries. The default is 'None'.
        results_folder : path
            The path were to store results from methods.

        Returns
        -------
        None.

        """
        
        self.systems=systems
        self.feature=feature #dictionary of features
        self.results=results
        self.discretized={}
    

    
    def combinatorial(self, shells, labels=None):
        """
        
        Function to generate combinatorial encoding of shells into states. Discretization is made based on combination of shells. 
        Not sensitive to which shell a given molecule is at each frame. Produces equal-sized strings for all parameters.
        Static methods defined in *tools.Functions*  are employed here.
        
        
        Parameters
        ----------
        shells: array
            The array of shell boundaries.
        labels: array
            The array of shell labels (+1 shells). Method can resolve for inconsistencies by reverting to default numeric label.
            
        
        Returns
        -------
        Dataframe of discretized features
        
        
        """

        
        def calculate(name, system, feature_name):
                
            system.discretized={}
            file=f'{system.results_folder}discretized_{feature_name}-combinatorial-{shells_name}.npy'
                    
            indexes=system.name.split('-') # + (feature,) #For df build        
            names=[f'l{i}' for i in range(1, len(indexes)+1)] 
            column_index=pd.MultiIndex.from_tuples([indexes], names=names)
       
            if not os.path.exists(file):
                print(f'Discretized {feature_name} array for {name} not found. Calculating...')
                
                feature_file=system.features[feature_name]
                        
                if feature_file == None:
                    print(f'Feature {feature_name} data not found.')
                else:
                    
                    raw_data=np.load(feature_file)
                          
                    #Reconstruct a DataFrame of stored feature into a Dataframe
                    #TODO: Check behaviour for sel > 1.
                    
                    frames, ref, sel=np.shape(raw_data)
                    raw_reshape=raw_data.reshape(frames, ref*sel)
                            
                    if ref == 1:
                        RAW=pd.DataFrame(raw_reshape)
                        RAW.columns=RAW.columns + 1
                    else:
                        RAW=pd.DataFrame()
                        RAW_split=np.split(raw_reshape, ref, axis=1)
                        for ref in RAW_split:
                            RAW_ref=pd.DataFrame(ref)
                            RAW_ref.columns=RAW_ref.columns + 1
                            RAW=pd.concat([RAW, RAW_ref], axis=0)  

                    state_df=tools.Functions.state_mapper(self.shells, array=RAW.values, labels=labels) #Get discretization from state mapper

                    comb_states=pd.DataFrame(state_df.values, index=0, columns=column_index[:-1])
                    comb_states.to_csv(f'{system.results_folder}discretized_{feature_name}-combinatorial-{shells_name}.csv')
                        
                    np.save(file, state_df.values)
                        
            else:
                #print(f'File found for {name}.')
                comb_states=pd.read_csv(f'{system.results_folder}discretized_{feature_name}-combinatorial-{shells_name}.csv',
                                        index_col=0, header=list(range(0,len(names))))
            comb_states.index.names=['Frame index']    
            system.discretized[feature_name]=file #Update system atributes
            #print(comb_states)
            return comb_states    
        

        #Set up

        self.shells=shells
        shells_name='_'.join(str(self.shells))
        
        #TODO: iterate through features.
        feature_name, feature_data=[i for i in self.feature]
        
        
        discretized_df_systems=pd.DataFrame()
        
        #Access individual systems. 
        #The id of project upon feature level should match the current id of project 
        for name, system in self.systems.items(): 
                    
            discretized_df_system=calculate(name, system, feature_name)
            discretized_df_systems=pd.concat([discretized_df_systems, discretized_df_system], axis=1) #call to function
        
        #print(discretized_df_systems)
        discretized_df_systems.name=f'{feature_name}_combinatorial'              
        discretized_df_systems.to_csv(f'{self.results}/combinatorial_{feature_name}_discretized.csv')
            
   
        self.discretized[feature_name]=('combinatorial', discretized_df_systems)
        
        return discretized_df_systems



    def nac_profile(self, 
                        shells, 
                        resolution=0.5,
                        nac_lims=(0,150), 
                        labels=None,
                        sels=None,
                        start=0,
                        stop=10,
                        stride=1,
                        dt=1):
        
        """
        Function to obtain orientation of molecules (as given by difference between distances) for each NAC bin. 
        Takes as inputs the nac and distance .npy objects calculated in NAC_calculation() and distances_calculation_all(). 
        Needs to convert the arrays into dataframes. 
        NOTE: The combined dataframe of NAC and DIST_O "merge" can be built with less number of frames, e.g. skipping each 2,  with the ".iloc[::2]" option.
        This may be required due to too much data being passed from the replicate_iterator().
        
        
        TODO: make call to ask for start stop, etc.
        """

        nac_range=np.arange(nac_lims[0], nac_lims[1], resolution)
    
        def calculate(name, system, feature_name):
                
            system.discretized={}
      
            indexes=system.name.split('-') # + (feature,) #For df build        
            names=[f'l{i}' for i in range(1, len(indexes)+1)] 
            column_index=pd.MultiIndex.from_tuples([indexes], names=names)
       
            feature_file=system.features[feature_name]
            print(feature_file)            
            
            if feature_name == 'pre-calculated':
                
               print(f'Feature name is: {feature_name}.\nWill load pre-calculated file.')
                
               raw_data=np.load(feature_file)
               #Reconstruct a DataFrame of stored feature into a Dataframe
               #TODO: Check behaviour for ref > 1.
                    
               frames, ref, sel=np.shape(raw_data)
               raw_reshape=raw_data.reshape(frames, ref*sel)
                            
               if ref == 1:
                   RAW=pd.DataFrame(raw_reshape)
                   RAW.columns=RAW.columns + 1
               else:
                   nac_df_system=pd.DataFrame()
                   split=np.split(raw_reshape, ref, axis=1)
                   for ref in split:
                       df_ref=pd.DataFrame(ref)
                       df_ref.columns=df_ref.columns + 1
                       nac_df_system=pd.concat([nac_df_system, df_ref], axis=0) 
                       

            elif feature_name == 'dNAC':
        
                nac_file=f'{results_folder}/dNAC_{len(sels)}-i{start*ps}-o{stop}-s{dt}.npy'

                if not os.path.exists(nac_file):
                    
                    dists=[] #list of distance arrays to be populated
                    
                    #iterate through the list of sels. 
                    #For each sel, define atomgroup1 (sel1) and atomgroup2 (sel2)
                    for idx, sel in enumerate(sels, 1): 
                        dist_file=f'{system_folder}/distance{idx}-i{start*ps}-o{stop}-s{dt}.npy'
            
                        if not os.path.exists(dist_file):
                
                            print(f'\t{dist_file} not found. Reading trajectory...')  
                            u=mda.Universe(topology, trajectory)
                            sel1, sel2 =u.select_atoms(sel[0]), u.select_atoms(sel[1]).positions
                    
                            print(f'\t\tsel1: {sel1}\n\t\tsel2: {sel2}\n\t\tCalculating...')        
                    
                            dists_=np.around(
                                [D.distance_array(sel1, sel2, box=u.dimensions) for ts in u.trajectory[start:stop:dt]],
                                decimals=3) 
                                
                            np.save(filename, dists_)
                        
                        else:
                            dists_=np.load(dist_file)
                            print(f'\t{dist_file} found. Shape: {np.shape(dists[idx])}')
                    
                        dists.append(dists_)
                        
                    #NAC calculation
                    nac_array=np.around(np.sqrt(np.power(np.asarray(dists), 2).sum(axis=0)/len(dists)), 3) #SQRT(SUM(di^2)/#i)  
                    
                    nac_df_system=pd.DataFrame(nac_array)
                    #nac_df=pd.DataFrame(np.round(nac_array.ravel(), decimals=1))
                    
                    print(nac_df)
     
            
            else:
               print('Feature name not defined')
               nac_df_system=None
               
               
            return nac_df_system


            #Generate the discretization dNAC into shells of thickness "resolution" across "range"
            nac_center_bin=nac_range+(resolution/2)
            
            NAC=pd.DataFrame(index=nac_center_bin[:-1], columns=column_index)

            for value in RAW: #optional: Do it across entire daframe to ignore individual molecule "value" information 
            
                hist, _=np.histogram(RAW[value], bins=nac_range)    
                NAC=pd.concat([NAC, pd.DataFrame(hist, index=nac_center_bin[:-1], columns=column_index)], axis=1)
            
            NAC.dropna(axis=1, inplace=True)
            
            return NAC, np.shape(RAW)[0] #0, frames, 1 ref, 2 sel. TODO: Check this well, its needed for ref > 1
        


        #Set up

        self.shells=shells
        shells_name='_'.join(str(self.shells))
        
        #TODO: iterate through features.
        feature_name, feature_data=[i for i in self.feature]
        
        
        nac_df=pd.DataFrame()
        
        #Access individual systems. 
        #The id of project upon feature level should match the current id of project 
        for name, system in self.systems.items(): 
                                
            nac_df_system, frames=calculate(name, system, feature_name)            
            nac_df=pd.concat([nac_df, nac_df_system], axis=1) #call to function
                
        
        return nac_df


    def dG_calculation(self, 
                       input_df, 
                       mol='methanol', 
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
    
        """
	
        from scipy.optimize import curve_fit

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

        df, pairs, replicas, molecules, frames=Discretize.get_descriptors(input_df, 
                                                                   level, 
                                                                   describe=describe,
                                                                   quantiles=quantiles)


        
        df.name=f'{feature_name}_{describe}' 
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

        for iterable, ax_fit, ax_dG in zip(descriptors_iterables, axes_fit.flat, axes_dG.flat):

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
            c_opt, c_cov = curve_fit(N_ref, 
                                     bulk_range[:-1], 
                                     N_enz_fit, 
                                     p0=bulk_value) #, bounds=(bulk_value - bulk_value*0.5, bulk_value + bulk_value*0.5))

            #stdev=np.sqrt(np.diag(c_cov))[0]
            #print(f'The fitted bulk value is: {c_opt[0]} +/- {stdev}')  
            
            #Recalculate N_ref with adjusted bulk concentration.
            N_opt=pd.Series(N_ref(ranges, c_opt[0]), index=N_enz.index.values, name=f'{name_fit} {iterable}')
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
            ax_fit.set_title(f'{bulk_value} {unit} ({np.round(c_opt[0], decimals=1)} {unit})', fontsize=10)
            
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
    def get_descriptors(input_df, level, describe='sum', quantiles=[0.1, 0.5, 0.99]):
        """
        Agreggate based on level (concentration) the quantile description (or other).
        
        """
        
        
        
        print(f'Descriptor of: {describe}')
        levels=input_df.columns.levels
        
        #TODO: remove to previous.
        ordered_concentrations=['50mM', '150mM', '300mM', '600mM', '1M', '2.5M', '5.5M']
        
        df=pd.DataFrame()
                    
        o_c=[]
        for c in ordered_concentrations:
            for q in quantiles:
                o_c.append(f'{c}-Q{q}')
            
        for iterable in ordered_concentrations: #levels[level]:
            
            df_it=input_df.loc[:,input_df.columns.get_level_values(f'l{level+1}') == iterable] #level +1 due to index starting at l1

            pairs=len(df_it.columns.get_level_values(f'l{level+2}'))
            replicas=len(df_it.columns.get_level_values(f'l{level+2}').unique()) #level +2 is is the replicates x number of molecules
            molecules=int(pairs/replicas)

            frames=int(df_it.sum(axis=0).unique()) #Optional, unpack the number of frames for each molecule.
            
            
            #print(f'Iterable: {iterable}\n\tPairs: {pairs}\n\treplicas: {replicas}\n\tmolecules: {molecules}\n\tframes: {frames}\n\tcounts: {counts}')
            

            if describe == 'single':
                descriptor=df_it.quantile(q=0.5)/frames*molecules
    
            elif describe == 'mean':
                descriptor=df_it.mean(axis=1)/frames*molecules
    
                descriptor.name=iterable
                descriptor_sem=df_it.sem(axis=1)/frames*molecules
                
                descriptor_sem.name=f'{iterable}-St.Err.'
                descriptor=pd.concat([descriptor, descriptor_sem], axis=1)
                #print(descriptor)
            
            elif describe == 'quantile':
                    
                descriptor=pd.DataFrame()    
                ordered_concentrations=o_c                    
                    
                for quantile in quantiles:        
                    descriptor_q=df_it.quantile(q=quantile, axis=1)/frames*molecules
                    descriptor_q.name=f'{iterable}-Q{quantile}'
                    descriptor=pd.concat([descriptor, descriptor_q], axis=1)
            
                #print(descriptor)
                #descriptor.plot(logy=True, logx=True)
            
            elif describe == 'sum':
                
                #descriptor=df_it.sum(axis=1)/(frames*len(replicas))
                descriptor=df_it.sum(axis=1)/(frames*replicas)
            
            descriptor.name=iterable            
            df=pd.concat([df, descriptor], axis=1)
        
        return df, pairs, replicas, molecules, frames
    
    
    
    
    @staticmethod
    def plot(df, level=2):
        

        levels=df.columns.levels #Exclude last, the values of states
        
        for iterable in levels[level]: 
            
            print(iterable)
            df_it=df.loc[:,df.columns.get_level_values(f'l{level+1}') == iterable] #level +1 due to index starting at l1
            
            #print(df_it)
            
            print(df_it.columns.unique(level=2).tolist())
            #plot_it=
            df_it.plot(kind='line', subplots=True, sharey=True, title=iterable, figsize=(7,5), legend=False, sort_columns=True)
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
    
    
    
    
    
    
    
    
    