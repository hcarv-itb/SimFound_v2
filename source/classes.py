# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:30:16 2020

@author: hca
"""

import numpy as np
import pandas as pd
import pyemma
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors, ticker, cm 
import os


class System():
    """Base class to build the paths and files of a given system (protein-ligand-parameter).
    workdir: Location of the files
    parent_folder: Location of the folder containing all the parameter
    parameter: list of parameter values (extention name of the parent folder)
    parameter_scalar: in the case of concentrations, the scalar value in Molar is also provided
    replicas: Number of replicas contained within each parent_folder-parameter folder. 
    trajectory: file name of the trajectory file in each replica folder
    topology: file name of the topology file in each replica folder
    protein: Given name of protein molecule
    ligand: Given name of ligand molecule
    timestep: The physical time of frame (ps)"""
    
    
    def __init__(self, workdir="workdir", parent_folder="parent folder", results_folder="results_folder", 
                 parameter="parameter", parameter_scalar="parameter_scalar", 
                 replicas="number replicates", 
                 trajectory="trajectory file", 
                 topology="topology file",
                 protein="protein name",
                 ligand="ligand name",
                 timestep="ps value of frame"):
        self.workdir=workdir
        self.parent_folder=parent_folder
        self.results_folder=results_folder
        self.parameter=parameter
        self.parameter_scalar=parameter_scalar
        self.replicas=np.arange(1, int(replicas)+1)
        self.trajectory=trajectory
        self.topology=topology
        self.protein=protein
        self.ligand=ligand
        self.timestep=int(timestep)
        
    def fileLocations(self):
        systems={}
        for p in self.parameter:
            p_folder=f'{self.workdir}{self.parent_folder}-{p}/'
            trj=[]
            top=[]
            nac=[]
            folder=[]
            for i in self.replicas:
                folder_i=f"{p_folder}sim{i}"
                trj_i=f"{folder_i}/{self.trajectory}"
                top_i=f"{folder_i}/{self.topology}"
                nac_i=f"{folder_i}/results/NAC2-MeOH-{p}-i100000-o-1-s1-rep{i}.npy"
                trj.append(trj_i)
                top.append(top_i)
                nac.append(nac_i)
                folder.append(folder_i)
            systems[p]={'parent_folder':p_folder, 'folders':folder, 'trjs':trj, 'tops':top, 'nacs':nac}
        return systems
    
    def getProperties(self, *args):
       
        properties=[]
        for prop in args:
            if prop in self.__dict__.keys():
                properties.append(self.__dict__[prop])
        
            else:
                print(f'Property "{prop}" not defined in system.')
        return properties
        


class Features():
    """Base class to create a 'features' object. Different featurization schemes can be coded.
    Currently, the default is to use NAC as a feature. The second method is to use the Ser105-MeOH distance as feature."""
    
    def __init__(self, systems):
        self.systems=systems
        
    def nac(self):
        """Read the stored NAC file inherited from System. Returns a dictionary with NAC arrays for each concentration
        Uses the pyEMMA load module to load the NAC .npy array
        TODO: Consider migrating NAC file specification from class System to these function"""
        
        nac={}
        #systems=super().fileLocations()
        for k,v in self.systems.items():
            #values=v['nacs']
            #nac_c=pyemma.coordinates.load(values)
            nac[k]=v['nacs']
        return nac
    
    def dist(self, dist1, dist2, stride):
        """Generate a features with pyEMMA "featurizer" and "add_distance" functions. 
        Takes as input 'dist1' and 'dist2' and 'stride' value.
        Units are nanometer. Conversion to angstrom is made by multiplication"""
        dist={}
        for parameter,v in self.systems.items():
            trjs=v['trjs']
            tops=v['tops']
            folders=v['folders']
            dists=[]
            for top, trj, folder in zip(tops, trjs, folders):
                dist_path=f'{folder}/results/distance_feature_pyemma-{stride}.npy'
                dists.append(dist_path)
                
                print(f'Distance path is {dist_path}')
                if os.path.exists(dist_path):
                    print(f'Array found in {dist_path}')
                else:
                    print(f'Calculating...')
                    feature=pyemma.coordinates.featurizer(top)               
                    feature.add_distances(indices=feature.select(dist1), 
                                      indices2=feature.select(dist2))
                    distance=pyemma.coordinates.load(trj, features=feature, stride=stride)*10
                    shape=np.shape(distance)
                    distance=distance.reshape(shape[0], 1, shape[1]) #WARNING: Test this for oligomers (not 1)
                    np.save(dist_path, distance)
                    
            dist[parameter]=dists
        return dist
            
class Discretize():
    """Base class to discretize features. Takes as input a dictionary of features, 
    each containing a dictionary for each parameter and raw_data"""
    
    def __init__(self, name, features, results):
        self.name=name #feature name
        self.features=features #dictionary of feature
        self.results=results
    
    def minValue(self, state_shell):
        """Discretization is made directly on raw_data using the numpy digitize function.
        Minimum value per frame (axis=2) is found using the numpy min function. 
        Discretization controled by state_shell"""
        
        msm_min_f={}
        for parameter, data_dict in self.features.items():
            raw_data=[] #data_dict may contain one or more .npy objects
            for i in data_dict:
                i_arr=np.load(i)
                raw_data.append(i_arr)
            rep, frames, ref, sel=np.shape(raw_data)
            raw_reshape=np.asarray(raw_data).reshape(rep*frames, ref*sel)
            discretize=np.min(np.digitize(raw_reshape, state_shell), axis=1)

            disc_path=f'{self.results}/discretized-minimum-{self.name}-{parameter}.npy'
            np.save(disc_path, discretize)
            msm_min_f[parameter]={'discretized':[disc_path]}
        return msm_min_f
    
    def single(self, state_shell):
        """Discretization is made directly on raw_data using the numpy digitize function.
        For each ligand, the digitized value is obtained. Discretization controled by state_shell.
        WARNING: size of output array is N_frames*N_ligands*N_replicates. Might crash for high N_frames or N_ligands"""
        
        msm_single_f={}
        for parameter, data_dict in self.features.items():
            raw_data=[] #data_dict may contain one or more .npy objects
            for i in data_dict:
                i_arr=np.load(i)
                raw_data.append(i_arr)
            rep, frames, ref, sel=np.shape(raw_data)
            raw_reshape=np.asarray(raw_data).reshape(rep*frames*ref*sel)
            
            discretize=np.digitize(raw_reshape, state_shell)
            disc_path='{}/discretized-single-{}-{}.npy'.format(self.results, self.name, parameter)
            np.save(disc_path, discretize)
            msm_single_f[parameter]={'discretized':[disc_path]}
        return msm_single_f
    

    def combinatorial(self, regions, labels=None):
        """Class to generate combinatorial encoding of regions into states. Discretization is made based on combination of regions. 
        Not sensitive to which region a given molecule is at each frame. Produces equal-sized strings for all parameters
        Values of state boundaries are given by regions array.
        Static methods defined Functions class are employed here."""

        msm_combinatorial_f={}
        for parameter, data_dict in self.features.items():
            
            disc_path='{}/discretized-combinatorial-{}-{}.npy'.format(self.results, self.name, parameter)
            raw_data=[] #data_dict may contain one or more .npy objects
            for i in data_dict:
                i_arr=np.load(i)
                raw_data.append(i_arr)
            rep, frames, ref, sel=np.shape(raw_data)
            raw_reshape=np.asarray(raw_data).reshape(rep*frames, ref*sel)
            
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
            
            state_df=Functions.state_mapper(regions, array=RAW.values, labels=labels) #Call to Functions class
            np.save(disc_path, state_df.values)
            msm_combinatorial_f[parameter]={'discretized':[disc_path]}
        
        return msm_combinatorial_f


class MSM():
    """Base class to create Markov state models. Discretization scheme 'scheme' needs to provided.
    The dictionary of values needs to be provided. Accepts name of protein 'prot' and name of ligand 'mol' to generate file names.
    Physical time of frames needs to be provided 'ps'."""
    
    def __init__(self, scheme, feature, parameter, values, prot, mol, ps, results='results'):
        self.scheme=scheme
        self.feature=feature
        self.parameter=parameter
        self.values=values
        self.prot=prot
        self.mol=mol
        self.ps=int(ps)
        self.results=results
        
    def ITS(self, lags):
        """Function to create ITS plot using as input 'lags' and the 'discretized' values.
        The corresponding 'png' file is checked in the 'results' folder. If found, calculations will be skipped."""
        
        
        its_name=f'{self.results}/its_bayesMSM_{self.scheme}-{self.feature}-{self.prot}-{self.mol}-{self.parameter}-{self.ps}ps.png'
        self.values['its']=its_name
        print(its_name)
        
        if os.path.exists(its_name):
            print('ITS plot already calculated:')
            plt.axis("off")
            plt.imshow(mpimg.imread(its_name))
            plt.clf()
        else:
            data=np.array(np.load(self.values['discretized'][0]))
            its=None
            its_stride=1
            while its == None:
                try:
                    print(f'Trying stride {its_stride}')
                    data_i=np.ascontiguousarray(data[0::its_stride])
                    its=pyemma.msm.its(data_i, lags=lags, errors='bayes')
                except:
                    print(f'Could not generate ITS. Increasing stride.')
                    its_stride+=its_stride*2
                    
            its_plot=pyemma.plots.plot_implied_timescales(its, units='ps', dt=self.ps)
            its_plot.set_title(f"{self.scheme} - {self.feature} - {self.prot} - {self.mol} - {self.parameter}")
            plt.savefig(its_name, dpi=600)
            plt.show()
            
        return its_name

      
    def bayesMSM(self, lag, variant=False, statdist_model=None):
        """Function to calculate model based on provided lag time. Must be evaluated suitable lag time based on ITS.
        If no suitable lag is found, model calculation will be skipped.
        Accepts variants. Currently, only *norm* is defined."""
        
        lag_model=lag[self.scheme][self.feature][self.parameter]


        if lag_model == None:
            print('No suitable lag time found. Skipping model')
            bayesMSM=None       
        else:
            if variant == False:
                bayesMSM_name=f'{self.results}/bayesMSM_{self.scheme}-{self.feature}-{self.prot}-{self.mol}-{self.parameter}-{self.ps}ps-lag{lag_model}.npy'
            else:
                bayesMSM_name=f'{self.results}/bayesMSM_{variant}-{self.scheme}-{self.feature}-{self.prot}-{self.mol}-{self.parameter}-{self.ps}ps-lag{lag_model}.npy'
            
            if os.path.exists(bayesMSM_name):
                bayesMSM=pyemma.load(bayesMSM_name)
            else:
                data=np.array(np.load(self.values['discretized'][0]))
                bayesMSM=None
                msm_stride=1
                print(bayesMSM_name)
                while bayesMSM == None and msm_stride < 10000:
                    data_i=np.ascontiguousarray(data[0::msm_stride])
                    bayesMSM=pyemma.msm.bayesian_markov_model(data_i, lag=lag_model, dt_traj=f'{self.ps} ps', conf=0.95, statdist=statdist_model)
                    try:
                        print(f'Trying stride {msm_stride}')
                        data_i=np.ascontiguousarray(data[0::msm_stride])
                        
                        #if variant == False:
                        #    bayesMSM=pyemma.msm.bayesian_markov_model(data_i, lag=lag_model, dt_traj=f'{self.ps} ps', conf=0.95)
                            
                        #else:
                        bayesMSM=pyemma.msm.bayesian_markov_model(data_i, lag=lag_model, dt_traj=f'{self.ps} ps', conf=0.95, statdist=statdist_model)
                        bayesMSM.save(bayesMSM_name, overwrite=True) 
                    except:
                        print(f'Could not generate bayesMSM. Increasing stride.')
                        msm_stride=msm_stride*2
                         
            #print(bayesMSM)
            
        return bayesMSM
        
    @staticmethod
    def stationaryDistribution(name, model):
        """Calculates the stationary distribution for input model."""
        
        
        scheme, feature, parameter=name.split('-')
        
        
        if model != None:
            states=model.active_set
            statdist=model.stationary_distribution
            #nstates=len(states)
            counts=model.active_count_fraction
            if counts != 1.0:
                print('\tWarning: Fraction of counts is not 1.')
                print(f'\tActive set={states}')
        else:
            statdist=None
            states=None
    
        column_index=pd.MultiIndex.from_product([[scheme], [feature], [parameter]], names=['Scheme', 'feature', 'parameter'])
        pi_model=pd.DataFrame(statdist, index=states, columns=column_index)

        return pi_model

    
    
    def CKTest(self, name, v, lag, mt, mlags=5):
        """Function to calculate CK test. Takes as input name of the model, the variant (name and model), dictionary of lags
        the number of coarse grained states, and the defail*cg_states* and *mlags*."""
        
        lag_model=lag[self.scheme][self.feature][self.parameter]
        
        
        if lag_model == None:
            print('No suitable lag time found. Skipping model')
            cktest_fig=None
        
        else:
            cktest_fig=f'{self.results}/cktest_{name}-{v[0]}-{self.prot}-{self.mol}-{self.parameter}-{self.ps}ps-lag{lag_model}-cg{mt}.png'
        
            if not os.path.exists(cktest_fig):
                try:
                    cktest=v[1].cktest(mt, mlags=mlags)
                    pyemma.plots.plot_cktest(cktest, units='ps')
                    plt.savefig(cktest_fig, dpi=600)
                    plt.show()
                except:
                    print(f'CK test not calculated for variant {v[0]} and {mt} states')
            else:
                print("CK test already calculated:")
                plt.axis("off")
                plt.imshow(mpimg.imread(cktest_fig))
                plt.show()    
             
        return cktest_fig
        
    @staticmethod   
    def flux(name, model, parameter_scalar=None, regions=None, labels=None, A_source=None, B_sink=None):
        """Function to calculate flux of model. A and B need to provided."""
        
        scheme, feature, parameter=name.split('-')
                
        if parameter_scalar != None:
            parameter=parameter_scalar[parameter]
                       
        if model == None:
            net_flux = None 
            
        else:
            if scheme == 'combinatorial':
                states=list(model.active_set)
                state_names=Functions.sampledStateLabels(regions, labels=labels)
                state_labels=[]
                
                
                for v in states:
                    state_labels.append(state_names[int(v)])
                
                #print(f'{parameter} states: {state_labels}')
                
                #Remove states in A or B that are not sampled at c. NOTE: The set of states will become different!
                A_filter = list(filter(lambda k: k in state_labels, A_source))
                A=([state_labels.index(k) for k in A_filter])

                B_filter = list(filter(lambda k: k in state_labels, B_sink))
                B=([state_labels.index(k) for k in B_filter])              
                
                if len(A_filter) != len(A_source) or len(B_filter) != len(B_sink):
                    print("\tWarning: not all A (source) or B (sink) states were sampled at {}".format(parameter))
                print("\tset A indexes: {} \n\tset B indexes: {}".format(A_filter, B_filter))
                try:
                    flux=pyemma.msm.tpt(model, A, B)
                    net_flux=flux.net_flux*1e12 #conversion from ps to s
                except:
                    print("No A or B states sampled")
                    net_flux=None
		      
            else:
                states=list(model.active_set)

                A=[states.index(states[0])]
                B=[states.index(states[-1])]
        
                flux=pyemma.msm.tpt(model, A, B)
                net_flux=flux.net_flux*1e12 #conversion from ps to s
            
        index_row=pd.MultiIndex.from_product([[scheme], [feature], [parameter]], 
                                                 names=['Scheme', 'feature', 'parameter'])
        flux_model=pd.DataFrame({'net_flux': np.sum(net_flux)}, index=index_row)
  
        return flux_model


    @staticmethod
    def MFPT(name, model):
	###Calculate MFPTs and corresponding rates
	#Replaces the method shown in the pyEMMA documentation with nested for loops
    
    
        scheme, feature, parameter=name.split('-')
        
        if model == None:
            print('no model')
        
        #elif parameter in ['50mM', '300mM'] and scheme is not 'minimum':
        else:
            states=model.active_set
            
            index_row=pd.MultiIndex.from_product([[scheme], [feature], [s for s in states]], 
                                                 names=['Scheme', 'feature', 'state_source'])
            index_col=pd.MultiIndex.from_product([[parameter], [s for s in states], ['mean', 'stdev']], names=['parameter', 'state_sink', 'values'])
            
            mfpt=pd.DataFrame(index=index_row, columns=index_col)
            
            for i, state_source in enumerate(states):
                for j, state_sink in enumerate(states):
                    mfpt.loc[(scheme, feature, state_source), (parameter, state_sink, 'mean')]= model.sample_mean("mfpt", i,j)
                    mfpt.loc[(scheme, feature, state_source), (parameter, state_sink, 'stdev')]= model.sample_std("mfpt", i,j)

            #Nested loop is not ideal, but method for requires definition of state_source and state_sink to fill dataframe
            
            return mfpt
        
    @staticmethod
    def mfpt_filter(mfpt_df, scheme, feature, parameter, error):
        """Function to filter out MFPT values whose standard deviations are above *error* value.
        Default value of *error* is 20%"""
        
        mfpt=mfpt_df.loc[(scheme, feature), (parameter)]
        mfpt.dropna(how='all', inplace=True)
        mfpt.dropna(how='all', axis=1, inplace=True)
            
            
        means=mfpt.iloc[:, mfpt.columns.get_level_values(level='values')=='mean']
        stdevs=mfpt.iloc[:, mfpt.columns.get_level_values(level='values')=='stdev']
        
        #print('Before: ', means.isna().sum().sum())
        counts=0
        ####Filter the mean values to the error percentage of stdev
        for mean, stdev in zip(means, stdevs):
            ratio = mfpt[stdev] / mfpt[mean]
            for r, v in zip(ratio, means[mean]):
                if r >= error:
                    index_means=means.index[means[mean] == v]
                    means.loc[index_means, mean] =np.nan
                    counts+=1
                        
            #means.dropna(how='all', inplace=True)
            #means.dropna(how='all', axis=1, inplace=True)
            
        #print('After: ', means.isna().sum().sum())
        #print('Counts: ', counts)
            
        #minimum_notzero, maximum= means[means.gt(0)].min().min(), means.max().max()
        
        return means
        
# =============================================================================
#     @staticmethod
#     def plot_MFPT(mfpt_df, scheme, feature, parameters, error, regions=None, labels=None):
#         """Function to plot heatmap of MFPTS between all states."""
#         
#         # Taken from matplotlib documentation. Make images respond to changes in the norm of other images (e.g. via the
#         # "edit axis, curves and images parameters" GUI on Qt), but be careful not to recurse infinitely!
#         def update(changed_image):
#             for im in images:
#                 if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
#                     im.set_cmap(changed_image.get_cmap())
#                     im.set_clim(changed_image.get_clim())     
#         
# 
#         images = []
#                 
#         fig, axes = plt.subplots(4,2, constrained_layout=True, figsize=(12,9))
#         fig.suptitle(f'Discretization: {scheme}\nFeature: {feature} (error tolerance {error:.1%})', fontsize=14)  
#         
#         cmap_plot=plt.cm.get_cmap("gist_rainbow")
#         #cmap_plot.set_under(color='red')
# 
#         #cmap_plot.set_over(color='yellow')
#         cmap_plot.set_bad(color='white')
#         
#         
#         vmins, vmaxs=[], []
#             
#         for plot, parameter in zip(axes.flat, parameters):
#             
#             means=MSM.mfpt_filter(mfpt_df, scheme, feature, parameter, error) 
#             
#             try:
#                 if scheme == 'combinatorial':
#                 
#                     contour_plot=plot.pcolormesh(means, edgecolors='k', linewidths=1, cmap=cmap_plot) 
#                     label_names=Functions.sampledStateLabels(regions, sampled_states=means.index.values, labels=labels)
#                     positions=np.arange(0.5, len(label_names)+0.5)                    
#                     plot.set_xticks(positions)
#                     plot.set_xticklabels(label_names, fontsize=7)
#                     plot.set_yticks(positions)
#                     plot.set_yticklabels(label_names, fontsize=7)
#                 else:
#                     contour_plot=plot.pcolormesh(means, cmap=cmap_plot)
#                     ticks=plot.get_xticks()+0.5
#                     plot.set_xticks(ticks)
#                     plot.set_xticklabels((ticks+0.5).astype(int))
#                     plot.set_yticks(ticks[:-1])
#                     plot.set_yticklabels((ticks+0.5).astype(int)[:-1])
#                 plot.set_facecolor('white')
#                 plot.set_title(parameter) #, fontsize=8)
#                 plot.set_xlabel('From state', fontsize=10)
#                 plot.set_ylabel('To state', fontsize=10)
#                 images.append(contour_plot)
# 
#             except:
#                 print('No values to plot')
# 
#         fig.delaxes(axes[3][1])
#         
#         # Find the min and max of all colors for use in setting the color scale.   
#         vmins=[]
#         vmaxs=[]
#         for image in images:
#             array=image.get_array()
#             try:
#                 vmin_i=np.min(array[np.nonzero(array)])
#             except:
#                 vmin_i=1
#             try:
#                 vmax_i=np.max(array[np.nonzero(array)])
#             except:
#                 vmax_i=1e12
#             vmins.append(vmin_i)
#             vmaxs.append(vmax_i)
#         
#         vmin=min(vmins)
#         vmax=max(vmaxs)
#         #vmax = max(image.get_array().max() for image in images)
# 
#         norm = colors.LogNorm(vmin=vmin, vmax=vmax)
# 
#         for im in images:
#             im.set_norm(norm)
#              
#         print(f'limits: {vmin:e}, {vmax:e}')
#          
#         cbar=fig.colorbar(images[-1], ax=axes)
#         cbar.set_label(label=r'MFPT (ps)', size='large')
#         cbar.ax.tick_params(labelsize=12)
#         for im in images:
#             im.callbacksSM.connect('changed', update)
# 
#                         
#         return images
# =============================================================================

        
class Functions():
    
    @staticmethod
    def regionLabels(regions, labels):
        """Evaluate number of defined regions and corresponding indetifiers.
        Instance will have *region_labels* as provided or with default numerical indetifiers if not."""
        
        try:
            if (len(labels) == len(regions) + 1):
                region_labels=labels
        except TypeError:
            print(f'Region labels not properly defined. Number of regions is different from number of labels.') 
            region_labels=None 
                
        if region_labels == None:
            print('No region labels defined. Using default numerical identifiers.')
            region_labels=[]
            for idx, value in enumerate(regions, 1):
                region_labels.append(idx)
            region_labels.append(idx+1)
            
        return region_labels 
    
    @staticmethod
    def sampledStateLabels(regions, sampled_states=None, labels=None):
        """Static method that generates state labels for sampled states.
        Requires the *regions* to be provided and optionally the *sampled_states* and *labels*."""        
        
        state_names=Functions.stateEncoder(regions, labels)[1] #Call to stateEncoder[0] = state_names
        
        if sampled_states is not None:
            sampled=[]
            for i in sampled_states:
                sampled.append(state_names[i])
        
            return sampled
        else:
            return state_names
        
    @staticmethod    
    def stateEncoder(regions, labels):
        """Create a mapping of state index to combination of labels."""
            
        region_labels=Functions.regionLabels(regions, labels)
        state_encodings = list(itertools.product([0, 1], repeat=len(regions)+1))

        state_names=[[] for i in range(len(state_encodings))]
        
        for i, encoding in enumerate(state_encodings):
            for f, value in enumerate(encoding):
                if value == 1:
                    state_names[i].append(region_labels[f])
        for i, name in enumerate(state_names):
            state_names[i]=''.join(map(str,name))
        
        return state_encodings, state_names  
    
    
    @staticmethod
    def state_mapper(regions, array, labels=None):
        """Static method that converts an *array* of values into a unique state based on occupancy of regions.
        Requires the *regions*, input *array* and *labels* to be provided.""" 
        
  
        def state_evaluator(x):
            """Combinatorial encoding of regions into a single state."""
            for c,v in enumerate(Functions.stateEncoder(regions, labels)[0]): #Call to stateEncoder[0] = state_encodings
                if np.array_equal(v, x.values):
                    return c 
                         

        state_map=np.digitize(array, regions, right=True) #Discretize array values in state bins.
            #state_map=NAC.applymap(lambda x:np.digitize(x, states, right=True)) #Deprecated. Much slower.
 
                
        state_comb=pd.DataFrame()
        for s in range(0, len(regions)+1):
            state_comb[s]=(state_map == s).any(1).astype(int)
        state_df=state_comb.apply(state_evaluator, axis=1)


        return state_df
            
            
