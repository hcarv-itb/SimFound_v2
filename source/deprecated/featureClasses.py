#!/bin/python3

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

kwargs={'results_dir': 'os.getcwd()', 
        'temp_dir': 'os.getcwd()', 
        'dt': 1,
        'c':'300mM',
        'mol': 'MeOH', 
        'prot': 'calb-S47L', 
        'start': 20000, 
        'stop': -1, 
        'ps': 5, 
        'sels': ['1', '2'],
        'replicate':'../calb-MeOH/calb-S47L-MeOH-300mM/sim2/results/NAC2-MeOH-300mM-i100000-o-1-s1-rep2.npy',
        'i':2}


class InputParameters:
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
        
    def info(self):
        for k,v in self.__dict__.items():
            print(f'{k} is {v}')

class NumberMolecules(InputParameters):
    '''Class to obtain the number of molecules for each NAC shell of radius 'nmol_integral' in the interval 'nmol_range'.
    nmol_range is an array of length 2 with lower and upper boundaries and must be specified.
    nmol_integral is a float value. If not defined, it will default to 1'''
    
    def __init__(self, sim):
        self.sim= sim

    
    def builder(self, nmol_range, nmol_integral=1):
        '''Checks if inputs are correct. Generates an array of NAC bins'''
        
        if len(nmol_range) != 2:
            raise ValueError(f'nmol_range not properly defined: 2 values were expected')
        elif nmol_integral == 1:
            print('Warning: Using default value of 1 angstrom for nmol_integral. Change it if desired.')
        print(f'NumberMolecules:NAC range (in angstrom) from {nmol_range[0]} to {nmol_range[-1]} with bins of {nmol_integral}.')
        return np.arange(nmol_range[0], nmol_range[-1], nmol_integral)
    
    
    def calculator(self, nmol_range, nmol_integral):
        '''Reconstructs the NAC array and counts number of times NAC values are within each bin'''
        
        
        #rebuild NAC
        filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sim.sels), sim.mol, sim.c, sim.start*ps, sim.stop, sim.dt, sim.i)
        NAC_array=np.load(replicate + 'results/' + filename_NAC) 

        #Make calculations
        Nmol_range=self.builder()
        Nmol_range_index=Nmol_range + (self.nmol_integral/2) #Center value of bins
        
        index=pd.MultiIndex.from_product([[c], Nmol_range_index], names=['[{}]'.format(mol), 'NAC'])
        nmol_df=pd.DataFrame(columns=['nmol', 'total'], index=index)

        for lower, upper, value in zip(Nmol_range, Nmol_range[1:], Nmol_range_index):
            nmol_df.xs(c).at[value ,'nmol']= ((NAC_array > lower) & (NAC_array <= upper)).sum() #Count NAC > lower and < upper
    
        nmol_df['total']= NAC_array.shape[0]
        nmol_df.dropna(inplace=True)
     
        return nmol_df
 



    @staticmethod
    def nmolStats(Nmol_c, results_dir, prot, mol):
        '''Calculates the statistics of number of molecules for each radius and concentration
        Performs fitting of data to Langmuir model'''
        
        def  Langmuir(c, Nmax, Kd): #Langmuir formula for Nmax and Kd calculation. Will take as input Nmol vs c (conc)
            return Nmax*c / (Kd + c)
        
        nmol_list=Nmol_c.index.unique(level=1).tolist()
        index=pd.MultiIndex.from_product([nmol_list, ['average', 'sem']], names=['NAC', 'stats'])
        stats_df=pd.DataFrame(columns=index, index=Nmol_c.index.unique(level=0))
        
        for r, r_df in Nmol_c.groupby(level=1):
            mean=r_df['nmol'].mean(axis=1) / r_df['total'].mean(axis=1)
            stats_df.loc[:, (r,'average')]=mean.values 
            sem=r_df['nmol'].sem(axis=1) / r_df['total'].mean(axis=1)
            stats_df.loc[:, (r, 'sem')]=sem.values
        
        try: #Check if there is already a correction of bulk concentrations (NAC method)
            conc_table=pd.read_csv('{}nac_concentrations-{}-{}.csv'.format(results_dir, prot, mol), index_col=0)
        except: #If not, use the defaul -conc values
            print("No bulk concentration(s) from NAC normalization found. Using inputs -conc")
            conc_table=stats_df.index
        
        #Convert all concentration values to M (if mM)
        for c in conc_table.index:
            try:
                float(str(c).split('mM')[0])
                conc_table.loc[c,'c_opt']=conc_table.loc[c,'c_opt']/1000
                conc_table.loc[c,'c_std']=conc_table.loc[c,'c_std']/1000
            except:
                pass
        
        #Calculate Nmax and Kd for each nmol_range
        index_fit=pd.MultiIndex.from_product([['Nmax', 'Kd'], ['value', 'std']])	
        fit=pd.DataFrame(columns=index_fit, index=nmol_list)
        
        x_points=np.linspace(conc_table['c_opt'][0], conc_table['c_opt'][-1], 100)
        langmuir_df=pd.DataFrame(index=x_points, columns=nmol_list)
        
        for NAC in stats_df.columns.levels[0].values:
            #Optional: Use the standard deviation as sigma. For Calb-MeOH errors in Kd got worse. From aprox. 30% to >40%
            popt, pcov=curve_fit(Langmuir, conc_table['c_opt'].values[:len(stats_df)], stats_df[NAC, 'average'].values,
                                 p0=[stats_df.at[stats_df.index[-1], (NAC, 'average')], conc_table['c_opt'].values[1]]) 
            #, sigma=stats_df[nmol_init, 'sem'].values, absolute_sigma=True)
            
            #Store the Nmax and Kd values with corresponding standard deviations if Kd stdev is < 0.5
            if np.sqrt(np.diag(pcov))[1] / popt[1] < 0.5:
                fit.loc[NAC, 'Nmax']=popt[0], np.sqrt(np.diag(pcov))[0]
                fit.loc[NAC, 'Kd']=popt[1], np.sqrt(np.diag(pcov))[1]
                
                #Generate Langmuir curve with fitted values
                langmuir_df[NAC]=Langmuir(x_points, popt[0], popt[1])
        
        #Append concentrations values in M to the final list
        stats_df['[{}] (M)'.format(mol)]=conc_table['c_opt'].values[:len(stats_df)]
        stats_df['+/- [{}] (M)'.format(mol)]=conc_table['c_std'].values[:len(stats_df)]
        
        return stats_df, fit, langmuir_df