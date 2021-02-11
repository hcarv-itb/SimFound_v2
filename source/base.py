# -*- coding: utf-8 -*-
"""

base.py
===================================
The core Classes of the package

Created on Fri Jul  3 11:30:16 2020

@author: hca
"""


def template(function):
    """
    The value of a function.
    
    Parameters
    ----------
    function: variable
        Some function.
    Returns
    -------
        The value of the function.
    """
    
    
def template2(function2):
    """
    The template2.

    Parameters
    ----------
    function2 : variable
        Another function.

    Returns
    -------
    the value of function2.

    """

import numpy as np
import pandas as pd
import mdtraj as md
import pyemma
import pickle
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import multiprocessing
from multiprocessing import Manager, Process

n_cpus=multiprocessing.cpu_count()
temp_dir='/tmp'

class Project():
    """Base class to build the paths and files of a given project (protein-ligand-parameter).
    
    
    
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
    
        
    
    def create(self, optionals=None):
        
        systems={}
        systems.update(self.setFileLocations())
        systems.update(self.setProperties())
        
        return systems
    
    def setFileLocations(self):
        locations={}
        for p in self.parameter:
            p_folder=f'{self.workdir}{self.parent_folder}-{p}'
            trj=[]
            top=[]
            nac=[]
            folder=[]
            for i in self.replicas:
                folder_i=f"{p_folder}/sim{i}"
                trj_i=f"{folder_i}/{self.trajectory}"
                top_i=f"{folder_i}/{self.topology}"
                nac_i=f"{folder_i}/results/NAC2-MeOH-{p}-i100000-o-1-s1-rep{i}.npy"
                trj.append(trj_i)
                top.append(top_i)
                nac.append(nac_i)
                folder.append(folder_i)
            locations[p]={'parent_folder':p_folder, 'folders':folder, 'trjs':trj, 'tops':top, 'nacs':nac}
        return locations
    
    def getProperties(self, *args):
       """Checks if queried properties is defined in self."""
       
       properties={}
       for prop in args:
           if prop in self.__dict__.keys():
               properties[prop]=self.__dict__[prop]
           else:
               print(f'Property "{prop}" not defined in system.')
       return properties
    
    def setProperties(self):
        """Sets by defaults the values in "properties" and can add further properties "optionals" defined in self"""
        
        properties=['protein', 'ligand', 'timestep', 'parameter']
        p_dict={}
        for p in properties:
            p_dict.update(self.getProperties(p))
        
        return p_dict


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
    
    def dist(self, dist1, dist2, stride, skip_frames=0):
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
                    
                    #i=md.iterload(trj,  skip=39000)
                    #print(i)
                    
                    distance=pyemma.coordinates.load(trj, skip=skip_frames, features=feature, stride=stride)*10
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
                if os.path.exists(i):
                    i_arr=np.load(i)
                    raw_data.append(i_arr)
                else:
                    print(f'\tWarning: file {i} not found.')
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
                if os.path.exists(i):
                    i_arr=np.load(i)
                    raw_data.append(i_arr)
                else:
                    print(f'\tWarning: file {i} not found.')
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
    
    def __init__(self, scheme, feature, parameter, values, protein, ligand, ps, results='results'):
        self.scheme=scheme
        self.feature=feature
        self.parameter=parameter
        self.values=values
        self.protein=protein
        self.ligand=ligand
        self.ps=int(ps)
        self.results=results
        self.name=f'{scheme}-{feature}-{parameter}'
        
    def ITS(self, lags):
        """Function to create ITS plot using as input 'lags' and the 'discretized' values.
        The corresponding 'png' file is checked in the 'results' folder. If found, calculations will be skipped."""
        
        
        its_name=f'{self.results}/its_bayesMSM_{self.scheme}-{self.feature}-{self.protein}-{self.ligand}-{self.parameter}-{self.ps}ps.png'
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
            while its == None and its_stride < 10000:
                try:
                    print(f'Trying stride {its_stride}')
                    data_i=np.ascontiguousarray(data[0::its_stride])
                    its=pyemma.msm.its(data_i, lags=lags, errors='bayes')
                except:
                    print(f'Could not generate ITS. Increasing stride.')
                    its_stride=its_stride*2
                    
            its_plot=pyemma.plots.plot_implied_timescales(its, units='ps', dt=self.ps)
            its_plot.set_title(f'Discretization: {self.scheme}\nFeature: {self.feature}\n{self.parameter}')
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
            bayesMSM_name=f'{self.results}/bayesMSM_{self.scheme}-{self.feature}-{self.protein}-{self.ligand}-{self.parameter}-{self.ps}ps-lag{lag_model}.npy'
            print(bayesMSM_name)
            if os.path.exists(bayesMSM_name):
                bayesMSM=pyemma.load(bayesMSM_name)
            else:
                data=np.array(np.load(self.values['discretized'][0]))
                bayesMSM=None
                msm_stride=1
                print(bayesMSM_name)
                while bayesMSM == None and msm_stride < 10000:
                    try:
                        print(f'Trying stride {msm_stride}')
                        data_i=np.ascontiguousarray(data[0::msm_stride])
                        bayesMSM=pyemma.msm.bayesian_markov_model(data_i, lag=lag_model, dt_traj=f'{self.ps} ps', conf=0.95, statdist=statdist_model)
                        bayesMSM.save(bayesMSM_name, overwrite=True) 
                    except:
                        print(f'Could not generate bayesMSM. Increasing stride.')
                        msm_stride=msm_stride*2
            
        return bayesMSM
        
    def stationaryDistribution(self, model):
        """Calculates the stationary distribution for input model."""
        
        #print(self.scheme, self.feature, self.parameter)
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
    
        column_index=pd.MultiIndex.from_product([[self.scheme], [self.feature], [self.parameter]], names=['Scheme', 'feature', 'parameter'])
        pi_model=pd.DataFrame(statdist, index=states, columns=column_index)

        return pi_model

    
    
    def CKTest(self, model, lag, mt, mlags=5):
        """Function to calculate CK test. Takes as input name of the model, the variant (name and model), dictionary of lags
        the number of coarse grained states, and the defail*cg_states* and *mlags*."""
        
        lag_model=lag[self.scheme][self.feature][self.parameter]
        
        
        if lag_model == None:
            print('No suitable lag time found. Skipping model')
            cktest_fig=None
        
        else:
            cktest_fig=f'{self.results}/cktest_{self.scheme}-{self.feature}-{self.parameter}-{self.ps}ps-lag{lag_model}-cg{mt}.png'
        
            if not os.path.exists(cktest_fig):
                try:
                    cktest=model.cktest(mt, mlags=mlags)
                    pyemma.plots.plot_cktest(cktest, units='ps')
                    plt.savefig(cktest_fig, dpi=600)
                    plt.show()
                except:
                    print(f'CK test not calculated for {mt} states')
            else:
                print("CK test already calculated:")
                plt.axis("off")
                plt.imshow(mpimg.imread(cktest_fig))
                plt.show()    
             
        return cktest_fig
        
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
        

    def MFPT(self, model):
        """Calculation of mean first passage times for all models.
        TODO: Method requires nested loops (i,j) to fill each cell of the MFPT matrix. 
        TODO: Check deprecated for list compreehension method (does not work multiIindex df)."""
        
        if model == None:
            print('no model')
        
        else:
            states=model.active_set
            
            index_row=pd.MultiIndex.from_product([[self.scheme], [self.feature], [s for s in states]], 
                                                 names=['Scheme', 'feature', 'state_source'])
            index_col=pd.MultiIndex.from_product([[self.parameter], [s for s in states], ['mean', 'stdev']], names=['parameter', 'state_sink', 'values'])
            
            mfpt=pd.DataFrame(index=index_row, columns=index_col)
            
            for i, state_source in enumerate(states):
                for j, state_sink in enumerate(states):
                    mfpt.loc[(self.scheme, self.feature, state_source), (self.parameter, state_sink, 'mean')]= model.sample_mean("mfpt", i,j)
                    mfpt.loc[(self.scheme, self.feature, state_source), (self.parameter, state_sink, 'stdev')]= model.sample_std("mfpt", i,j)
            
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
                        
        means.dropna(how='all', inplace=True)
        means.dropna(how='all', axis=1, inplace=True)
            
        #print('After: ', means.isna().sum().sum())
        #print('Counts: ', counts)
            
        #minimum_notzero, maximum= means[means.gt(0)].min().min(), means.max().max()
        
        return means
    
    
class Trajectory():
    """Class to perform transformations on trajectories using MDTRAJ."""
    
    def __init__(self, systems):
        self.systems=systems
    
    def density(self, selection='not protein', convert=True, unit='Molar'):
        """Function to calculate the 3D density of a "selection" in a trajectory which has been previously center/fit-ed. 
        unit = 'Molar', 'A^3', 'SPC', 'TIP4P', 'others' (check documentation)"""    
        
        
        from MDAnalysis import Universe
        from MDAnalysis.analysis.density import DensityAnalysis
        
        protein=self.systems['protein']
        print(f'protein is: {protein}')
        for parameter,v in self.systems.items():
            
            working_folder=v['parent_folder']
            print(f'working folder is: {working_folder}')

            struct=(f'{working_folder}/{protein}-cat.gro')
            trj=(f'{working_folder}/{protein}-cat.xtc')
            
            ###Heavy load
            try:
                u= Universe(struct, trj)
                selection=u.select_atoms(selection)
                D = DensityAnalysis(selection)
                D.run()
            ###
                if convert == True:
                    D.density.convert_density(unit)
                    D.density.export(f'{working_folder}/{protein}-{unit}.dx', type="double")
                else:
                    D.density.export(f'{working_folder}/{protein}.dx', type="double")
                
                prob_density= D.density.grid / D.density.grid.sum()
                np.save(f'{working_folder}/{protein}-prob_density.npy', prob_density)
                
                # Histogram. Ensure that the density is A^{-3}
                D.density.convert_density("A^{-3}")
                dV = np.prod(D.density.delta)
                atom_count_histogram = D.density.grid * dV
                np.save(f'{working_folder}/{protein}-histogram.npy', atom_count_histogram)
                
                
            except:
                print('Could not find file. Skipping calculations.')

        return print('Calculations complete.')  
    
    
    
    
    
    def superposeTrajs(self, selection='name CA', skip_frames=0):
        """Generate a new trajectory with all frames superposed to the reference one.
        TODO: Find quick way to determine how many frames a replica has and replace the value in chunk_size (40000)"""
    
    
        def superpose_wMDTRAJ(t):
            """Function to supoerpose and center/fit all frames of a replica."""
            print(t)
            t.atom_slice(atom_indices_noW, inplace=True)
            t.image_molecules(inplace=True)
            t.superpose(reference=trj_ref, frame=0, atom_indices=atom_indices)
                
            return t
            
        output_folder= self.systems['parent_folder']
        print(output_folder)
        protein=self.systems['protein']
        for parameter,v in self.systems.items():
            
            trjs=v['trjs']
            tops=v['tops']
            folders=v['folders']

            print(folders[0])
            
            chunk_size=int(40000/(n_cpus/2))
            #try:
            print(chunk_size)
            file_superposed=f'{output_folder}/{protein}-cat'
            try:
                if not os.path.exists(file_superposed+'.xtc'):
                    print(f'File {file_superposed}.xtc not found. Superposing replicas')
                    trj_list=[]
                    trj_ref=md.load_frame(trjs[0], 0, top=tops[0]) 
                    atom_indices_noW=trj_ref.topology.select('not water')
                    atom_indices=trj_ref.topology.select(selection)
                
                    trj_ref.atom_slice(atom_indices_noW, inplace=True)
                    trj_ref.save_gro(file_superposed+'.gro')
                
                    for top, trj, in zip(tops, trjs):
                        print(f'Superposing replica {tops.index(top)+1}.')
                        #traj=md.iterload(trj, chunk=chunk_size, top=top)
                        traj=md.load(trj, top=top,stride=1000)   
                        #traj_i=[t for t in traj]
                                             
                        #superposed_chunks=Functions.parallel_task(input_function=superpose_wMDTRAJ, collection=traj_i)
                        
                        traj_superposed=superpose_wMDTRAJ(traj)
                        #print(superposed_chunks)
                        #print(type(superposed_chunks))
                        #traj_superposed=None
                        #for k, v in superposed_chunks.items():
                        #    print(k,v)
                        #    traj_i[k]=v
                        
                        #traj_superposed=md.join(traj_i)
                            
                        #traj_superposed=md.join(traj_chunks)
                        trj_list.append(traj_superposed)
                
                    superposed=md.join(trj_list)
                    print(superposed)
                    superposed.save_xtc(file_superposed+'.xtc')
            
                else:
                    print(f'File {file_superposed}.xtc found')
            
            except:
                print(f'Could not superpose for {parameter}')
    

        
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
    
    
    
    @staticmethod
    def parallel_task(input_function=None, collection=None):
        """Core function for multiprocessing of a given task.Takes as input a function and an iterator.
        Warning: Spawns as many threads as the number of elements in the iterator."""
        
        
        def task_exec(manager, input_function, idx, c):
            """Execution of the input function. Output is stored in the manager, which is handled to the job"""
            
            task=manager
            function=input_function
            task[idx]=function(c)           
        
        if input_function != None and collection != None:
        
            manager=Manager().list()
            job=[Process(target=task_exec, args=(manager, input_function, idx, c)) for idx, c in enumerate(collection)]
            _=[p.start() for p in job]
            _=[p.join() for p in job]
       
            return manager
        
        else:
            print('No input function or container provided')
        
        
        
        
