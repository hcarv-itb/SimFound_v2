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
import glob
import nglview
import Trajectory
import tools_plots
import tools
import mdtraj as md
from mdtraj.utils import in_units_of
from simtk.unit import angstrom
from itertools import compress
from functools import wraps

class MSM:
    """
    Base class to create a pyEMMA *features* object.
    
    
    """
    
    skip=0
    c_stride=1
    subset='not water'
    superpose='protein and backbone'
    var_cutoff=0.95
    msm_var_cutoff = 0.95
    report_units = 'angstrom'
    report_flux_units = 'second'
    ITS_lags=[1, 2, 5, 10, 20, 40, 100, 250, 500, 750, 1000, 1500, 2000]
    ref_samples=2000
    




    def __init__(self,
                 project,
                 regions,
                 timestep,  
                 chunksize=0, 
                 stride=1,
                 skip=0,
                 confirmation=False,
                 _new=False,
                 warnings=False,
                 pre_process=True,
                 equilibration=False,
                 production=True,
                 def_top=None,
                 def_traj=None,
                 heavy_and_fast=False):
        """
        

        Parameters
        ----------
        systems : TYPE
            DESCRIPTION.
        timestep : TYPE
            DESCRIPTION.
        results : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.project = project
        self.systems = self.project.systems
        self.regions=regions
        self.features={}
        self.data = {}
        self.chunksize=chunksize
        self.stride=stride
        self.skip=skip
        self.timestep=timestep
        self.unit=timestep.unit.get_symbol()
        self.warnings=warnings
        self.pre_process=pre_process
        self.w_equilibration = equilibration
        self.w_production = production
        self.def_top = def_top
        self.def_traj = def_traj
        self.heavy_and_fast=heavy_and_fast
        result_paths = []
        for name, system in self.systems.items(): 
            result_paths.append(system.project_results)
        if len(list(set(result_paths))) == 1:
            result_path = list(set(result_paths))[0]
            self.results = os.path.abspath(f'{result_path}/MSM') 
        else:
            self.results=os.path.abspath(f'{self.project.results}/MSM')
            print('Warning! More than one project result paths defined. reverting to default.')
        self.stored=os.path.abspath(f'{self.results}/MSM_storage')
        print('Results will be stored under: ', self.results)
        print('PyEMMA calculations will be stored under: ', self.stored)
        tools.Functions.fileHandler([self.results, self.stored], confirmation=confirmation, _new=_new)
 
    
    def model_comparison(self, get='mean', w_vamp=True):
        
        def get_mean(vamp_):
            vamp_str = [i for i in vamp_.strip(' []').split(' ')]
            vamps = []
            for i in vamp_str:
                try:
                    vamps.append(float(i))
                except:
                    pass
            return np.mean(vamps)
        
        files=glob.glob(f'{self.results}/ModelSelection_*.csv')
        df_=pd.DataFrame()
        print(f'{len(files)} files collected.')
        for f in files:
            df=pd.read_csv(f, index_col=[0,1,2])
            df_=pd.concat([df_, df])

        if get == 'mean':
            df_['VAMP2'] = df_['VAMP2'].apply(get_mean)
        
# =============================================================================
#         if get_top != None:
#             highest_set = (-np.asarray(means)).argsort()[:get_top]
#             top_index = df_.iloc[highest_set]
#             print(df_.index[highest_set])
#             top_df = pd.concat([top_df, df_.index[highest_set]])
#             axes.errorbar(x=x, y=means, yerr=errors, label=labels)
# =============================================================================
            
        return df_
    
 
    
    def set_model(func):
        """Decorator for system setup"""
        
        @wraps(func)
        def model(*args, **kwargs):
            print(func.__name__ + " was called")
            return func(*args, **kwargs)
        return model
    
    @set_model
    def set_specs_wrap(self, lag, ft_name, feature, n_state):
        """sets systems specifications"""
        if self.disc_lag != None:
           disc_lag_=str(self.disc_lag*self.timestep).replace(" ", "")
           append = f'@{disc_lag_}'
        else:
            append = ''
        lag_ =str(lag*self.timestep).replace(" ", "")
        self.selections = self.regions[ft_name]
        self.ft_name = ft_name
        self.feature = feature
        self.lag = lag
        self.n_state = n_state
        self.feature_name = f'{feature}{append}'
        self.ft_base_name = f'{ft_name}_{self.feature_name}_s{self.stride}'
        self.full_name = f'{self.ft_base_name}_{n_state}@{lag_}'
        
        return self.full_name

    def set_specs(self, ft_name, feature, lag=1, n_state=0, set_mode='all'):
        if self.disc_lag != None:
            disc_lag_=str(self.disc_lag*self.timestep).replace(" ", "")
            append = f'@{disc_lag_}'
        else:
            append = ''
        lag_ =str(lag*self.timestep).replace(" ", "")
        
        feature_name = f'{feature}{append}'
        
        if set_mode == 'all':
            self.selections = self.regions[ft_name]
            self.ft_name = ft_name
            self.feature = feature
            self.lag = lag
            self.n_state = n_state
            self.feature_name = feature_name
            self.ft_base_name = f'{ft_name}_{self.feature_name}_s{self.stride}'
            self.full_name = f'{self.ft_base_name}_{n_state}@{lag_}'
        else:
            ft_base_name = f'{ft_name}_{feature_name}_s{self.stride}'
            return ft_base_name

    def set_systems(self):
        systems_specs=[]
        for name, system in self.systems.items():
            results_folder=system.results_folder
            topology=Trajectory.Trajectory.fileFilter(name, 
                                                        system.topology, 
                                                        self.w_equilibration, 
                                                        self.w_production, 
                                                        def_file=self.def_top,
                                                        warnings=self.warnings, 
                                                        filterW=self.heavy_and_fast)
            trajectory=Trajectory.Trajectory.fileFilter(name, 
                                                        system.trajectory, 
                                                        self.w_equilibration, 
                                                        self.w_production,
                                                        def_file=self.def_traj,
                                                        warnings=self.warnings, 
                                                        filterW=self.heavy_and_fast)
            systems_specs.append((trajectory, topology, results_folder, name))            
        #Gather all trajectories and handover to pyEMMA
        trajectories_to_load = []
        tops_to_load = []
        for system in systems_specs: #trajectory, topology, results_folder, name
            #TODO: check if only trj_list[0] is always the enough.
            trj_list, top=system[0], system[1]
            if len(trj_list):
                trajectories_to_load.append(trj_list)
                tops_to_load.append(top)
        self.tops_to_load=tops_to_load[0]
        self.trajectories_to_load=trajectories_to_load
        
    def load_models(self):

        self.set_systems()
        models = {}
        for input_params in self.inputs:
            (ft_name, feature, n_state_sets, lag_sets) = input_params[:4]
            n_states = self.check_inputs(n_state_sets)
            lags = self.check_inputs(lag_sets)
            self.ft_base_name = self.set_specs(ft_name, feature, set_mode='base')
            self.load_discretized_data(feature=feature) #placed here so that it's loaded only once. Bad if all done, good if not all done yet.
            for lag in lags:    
                for n_state in n_states:
                    #self.set_specs_wrap(self, lag, ft_name, feature, n_state)
                    self.set_specs(ft_name, feature, lag=lag, n_state=n_state)
                    model_status = self.check_model()
                    if model_status:
                        print(f'Model {self.full_name} found and passed filter(s)')
                        model = self.bayesMSM_calculation(lag) 
                        models[self.full_name] = (model, ft_name, feature, n_state, lag)    
                    elif model_status == False:
                        print(f'Model {self.full_name} found but not passed filter(s)')
                    elif model_status == None:
                        model = self.bayesMSM_calculation(lag)
                        if self.filter_model(model):
                            models[self.full_name] = (model, ft_name, feature, n_state, lag)   

        return models
    
    def filter_model(self, msm, vamp_iters=3):
        
        print('Filtering models based on filters:' , self.filters)
        def evaluate():
            keep = []
            for filter_ in self.filters:
                if filter_ == 'connectivity':
                    if fraction_state == 1.000:
                        keep.append(True)
                    else:
                        keep.append(False)
                        print(f'\tWarning! Model {self.full_name} is disconnected: {fraction_state} states, {fraction_count} counts')
                elif filter_ =='time_resolved':
                    if resolved_processes > 1:
                        keep.append(True)
                    else:
                        keep.append(False)
                        print(f'\tWarning! Model {self.full_name} has no processes resolved above lag time.')
                elif filter_ =='first_eigenvector':
                    eigvec = msm.eigenvectors_right()
                    allclose = np.allclose(eigvec[:, 0], 1, atol=1e-15)
                    if allclose:
                        keep.append(True)
                    else:
                        keep.append(False)
                        print('\tWarning! first eigenvector is not 1 within error tol. 1e-15)')
            return keep
        
        df_name = f'{self.results}/ModelSelection_{self.ft_base_name}.csv'
        try:
            df=pd.read_csv(df_name, index_col=[0,1,2])
        except FileNotFoundError:
            df=pd.DataFrame()
        
        lag_ =str(self.lag*self.timestep).replace(" ", "")
        msm_name = f'{self.n_state}@{lag_}'
        index_row= pd.MultiIndex.from_product([[self.ft_name], [self.feature_name], [msm_name]], names=['name', 'feature', 'model'])
            
        fraction_state = np.around(msm.active_state_fraction, 3)
        fraction_count = np.around(msm.active_count_fraction, 3)
        resolved_processes = self.spectral_analysis(msm, self.lag, plot=False)
        
        df_ = pd.DataFrame(index=index_row, columns=['VAMP2', 'Test', 'Filters', 'Processes'])
        
        if self.get_vamp:
            discretized_data = self.discretized_data
            clusters=self.clusterKmeans_standalone(discretized_data, self.n_state)
            try:
                score_model = pyemma.msm.estimate_markov_model(clusters.dtrajs, lag=self.lag, dt_traj=str(self.timestep))
                vamp = score_model.score_cv(clusters.dtrajs, n=vamp_iters, score_method='VAMP2', score_k=min(10, self.n_state))
            except Exception as v:
                print(v)
                vamp = [np.nan]*vamp_iters
                
        else:
            vamp = [np.nan]*vamp_iters
        
        keep = evaluate()
        #filters_ = list(compress(filters, keep))
        filters_ = ' '.join(list(compress(self.filters, keep)))
        if all(keep):
            print(f'\tModel {self.full_name} passed all filters')
            passed = True
            df_.loc[(self.ft_name, self.feature_name, msm_name),'Test'] = True
        else:
            passed = False
            df_.loc[(self.ft_name, self.feature_name, msm_name),'Test'] = False
        df_.loc[(self.ft_name, self.feature_name, msm_name),'VAMP2'] = vamp
        df_.loc[(self.ft_name, self.feature_name, msm_name),'Filters'] = filters_
        df_.loc[(self.ft_name, self.feature_name, msm_name),'Processes'] = f'{resolved_processes}/{self.n_state}'
        
        df=pd.concat([df, df_])
        df.to_csv(df_name)

        if passed:
            return True
        else:
            return False
        

    def check_model(self):
        if len(self.filters) > 0:
            df_name = f'{self.results}/ModelSelection_{self.ft_base_name}.csv'
    
            lag_ =str(self.lag*self.timestep).replace(" ", "")
            msm_name = f'{self.n_state}@{lag_}'
            feature_name = self.feature_name
            try:
                df=pd.read_csv(df_name, index_col=[0,1,2])
                try:
                    passed = df.loc[(self.ft_name, feature_name, msm_name),'Test']
                    #print('Found entry for: ', self.full_name, passed)
                    if passed:
                        return True
                    else: 
                        return False
                except KeyError:
                    print('No entry for: ', self.full_name)
                    return None
            except FileNotFoundError:
                print('DataFrame not found: ', df_name)
                return None
        else:
            return True
        return passed

   
    def load_discretized_data(self, feature=None):
        if self.disc_lag != None:
            self.discretized_data=self.TICA_calculation(self.disc_lag, feature)
        else:
            #TODO: make here entry for other discretization schemes
            self.discretized_data = None

    def load_discTrajs(self):
        #TODO: some function calls might need np.concatenate(disc_trajs) others don't.
        #discretized_data = self.load_discretized_data(feature=self.feature)
        if self.get_tica:
            tica=self.discretized_data  
            data_concat = np.concatenate(tica.get_output())
            clusters=self.clusterKmeans_standalone(tica, self.n_state) #tica, n_clusters
            disc_trajs = clusters.dtrajs #np.concatenate(clusters.dtrajs)
        else:
            #TODO: make here flexible for other types of disc traj. Evolve to own
            data_concat = []
            disc_trajs = []
            clusters = []
        
        return data_concat, disc_trajs, clusters




    def analysis(self, 
                 inputs, 
                 disc_lag=None, 
                 method=None, 
                 dim=-1,
                 sample_frames = 10000,
                 filters=['connectivity', 'time_resolved', 'first_eigenvector'],
                 compare_pdbs = [],
                 eval_vamps=False,
                 overwrite=False):
        
        self.overwrite=overwrite
        self.viewer = {}
        self.dim = dim
        self.get_tica=False
        self.filters = filters
        self.get_vamp = eval_vamps
        self.method = method
        if disc_lag != None:
            self.disc_lag = disc_lag
            self.get_tica=True
        
        if isinstance(inputs, list):
            for v in inputs:
                if not 3 < len(v) <= 6:
                    raise ValueError('Input_params has to be tuple of kind: (region, feature, n_state(s), lag(s), *[n. macrostate(s), flux input(s)])')
            self.inputs = inputs
        else:
            raise TypeError('Input has to be a list of kind: region : [(input_params1), (input_params2)]')

        flux, committor, pathway = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        models = self.load_models() #(model, ft_name, feature, n_state, lag)

        for name, model in models.items():
            (msm, ft_name, feature, n_state, lag) = model
            self.set_specs(ft_name, feature, lag, n_state)
            #self.discretized_data = self.load_discretized_data(feature=feature)
            
            name_ = f'{ft_name}-{feature}'
            for input_params in self.inputs:
                if f'{input_params[0]}-{input_params[1]}' == name_:
                    try:
                        macrostates = self.check_inputs(input_params[4])
                    except IndexError:
                        macrostates = None
                        pass
            if method == 'PCCA':
                self.PCCA_calculation(msm, macrostates, auto=False, dims_plot=10)
                
            elif method == 'CKtest':
                self.CKTest_calculation(msm, macrostates)

            elif method == 'Spectral':
                self.spectral_analysis(msm, lag, plot=True)
            elif method == 'MFPT':
                self.MFPT_calculation(msm, macrostates)
            elif method == 'Visual':
                for macrostate in macrostates:
                    state_samples = self.extract_states(msm, macrostate, n_total_frames=40, visual=True) 
                    self.viewer[f'{self.full_name}-{macrostate}'] = self.visualize_metastable(state_samples)
            elif method == 'RMSD':
                state_samples = self.extract_states(msm, macrostates, visual=False)
                self.RMSD_source_sink(state_samples, msm, ref_sample=sample_frames, pdb_files = compare_pdbs)
            elif method == 'flux':
                flux_inputs_ = self.inputs[self.ft_name][5]
                flux_inputs = self.check_inputs(flux_inputs_)
                flux_df, committor_df, pathway_df = self.flux_calculation(msm, macrostate=macrostates, between=flux_inputs)
                flux = pd.concat([flux, flux_df], axis=0)
                committor = pd.concat([committor, committor_df], axis=0)
                pathway = pd.concat([pathway, pathway_df], axis=0)
            elif method == 'inspect':
                pass
            else:
                raise SyntaxError('Analysis method not defined')
        
        if method == 'flux':
            print(flux)
            flux.to_csv(f'{self.results}/flux_all.csv')
            committor.to_csv(f'{self.results}/committor_all.csv')
            pathway.to_csv(f'{self.results}/pathway_all.csv')
            print(committor)
            print(pathway)
        if method == 'Visual':
            return self.viewer



    def calculate(self,
                  inputs=None,
                  method=None,
                  evaluate=[None], 
                  opt_feature={}, 
                  features=['torsions', 'positions', 'distances'],
                  TICA_lag=None,
                  cluster_lags=None,
                  VAMP_lags=None,
                  dim=-1,
                  overwrite=False):
        """
        Wrapper function to calculate VAMP2 scores using pyEMMA. 
        


        inputs : list or tuple
            Either a list of 1 selection or atuple of 2 selections.
        evaluate : string
            How to calculate VAMP scores. The default is None.
        ft_name: string, optional.
            The optional name of the inputs. The default is None.
        lags : list, optional.
            A list of lag values to calculate VAMP2 scores. The defaults is [1].
        dim : int, optional.
            The number of dimensions to use for VAMP2 calculation in some methods. 
            If fractionary and < 1, will calculate the corresponding precentage of kinetic variance.
            More details in pyEMMA pyemma.vamp.


        Returns
        -------
        dataframe
            A dataframe containing all featurized values across the list of iterables*replicas*pairs.

        """

        # Define system definitions and measurement hyperparameters
        self.features=features
        self.overwrite=overwrite
        self.inputs = inputs
        self.get_tica=False
        if TICA_lag != None:
            self.disc_lag = TICA_lag
            self.get_tica=True
        
        self.set_systems()
        
        #Use optional feature not defined at __init__
        try:
            ft_names = self.regions
        except:
            if not isinstance(ft_names, dict):
                raise TypeError('Provide a dictionary of type {region : inputs}')
            ft_names = opt_feature  
 
        #Loop through features
        for ft_name, ft_inputs_ in ft_names.items():
            self.ft_name=ft_name 
            self.selections = ft_inputs_ #The resid selections from input in __init__

            #VAMPs
            if method == 'VAMP':
                for ev in evaluate:
                    if ev == 'features':
                        self.VAMP2_features(VAMP_lags, dim)
                    elif ev == 'dimensions': 
                        self.VAMP2_dimensions(VAMP_lags)   
                    else:
                        print('No evaluation defined.')
            else:
                ft_inputs = inputs[ft_name] #reads from input in __init__ and maps to current input
                self.ft_inputs=ft_inputs #whatever inputs are
                for feature in self.features:
                    self.feature = feature

                    print('Feature: ', feature)
                    
                    if TICA_lag != None:
                        tica_lag_ = str(TICA_lag*self.timestep).replace(" ", "")
                        self.ft_base_name = f'{ft_name}_{feature}@{tica_lag_}_s{self.stride}'
                        self.discretized_data = self.TICA_calculation(TICA_lag, feature, dim=dim)
                    else:
                        self.discretized_data = []
                        self.ft_base_name = f'{ft_name}_{feature}_s{self.stride}'

                    if method == 'TICA':
                            pass
                        
                    elif method == 'Clustering':
                        self.cluster_calculation(lags=cluster_lags, TICA_lag=TICA_lag, method='kmeans')

                    elif method == 'ITS':
                        if isinstance(self.ft_inputs, int):
                             self.n_state = self.ft_inputs
                             self.ITS_calculation(MSM.ITS_lags, c_stride=1)
                        else:
                             raise TypeError('Input values have to be a integer "number of states"')

                    elif method == 'bayesMSM':
                        if isinstance(self.ft_inputs, tuple) and len(self.ft_inputs) == 2:
                            (n_states, msm_lag) = inputs[ft_name]
                            msm_lag_ = str(msm_lag*self.timestep).replace(" ", "")
                            self.full_name = f'{self.ft_base_name}_{n_states}@{msm_lag_}'
                            self.n_state = n_states
                            self.bayesMSM_calculation(msm_lag)
                        else:
                            raise TypeError('Input values have to be a tuple of integers (number of states, lag)')
                    else:
                        raise ValueError('No method defined')





    def flux_calculation(self, 
                         msm, 
                         macrostate=None, 
                         between=([], []), 
                         top_pathways=-1):
        """
        Function to calculate flux of model. A and B need to provided.

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        data_tuple : TYPE
            DESCRIPTION.
        macrostate : TYPE, optional
            DESCRIPTION. The default is None.
        between : TYPE, optional
            DESCRIPTION. The default is ([], []).
        top_pathways : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        None.

        """

        lag_ =str(self.lag*self.timestep).replace(" ", "")

        A, B = between[0], between[1]
        A = [a-1 for a in A]#since user starts at index 1.
        B = [b-1 for b in B]
        if len(A) > 0 and len(B) > 0:
            pass
        else:
            raise("Source or sink states not defined (between =([A], [B])")
            
        if isinstance(macrostate, int): 
             msm.pcca(macrostate)
             sets = msm.metastable_sets

             A_ = np.concatenate([sets[a] for a in A]) 
             B_ = np.concatenate([sets[b] for b in B])
             print(f'\tPerforming coarse-grained TPT from A: {A} ({A_}) to B: {B} ({B_})')
             label_names = [f'MS {i}' for i in range(1, macrostate+1)]  
             flux_ =pyemma.msm.tpt(msm, A_, B_)
             cg, flux = flux_.coarse_grain(msm.metastable_sets)
             #TODO
             #make mapping of I and colors.
             not_I = list(A_)+ list(B_)
             I = list(set(not_I) ^ set(msm.active_set))
             label_I = []
             for idx, set_ in enumerate(sets, 1):
                 for i in I:
                     if i in set_:
                         label_I.append(f'MS {idx}')
             label_I = list(set(label_I))         
             label_A=[f'MS {idx}' for idx, set_ in enumerate(sets,1) if A_ in set_]
             label_B=[f'MS {idx}' for idx, set_ in enumerate(sets,1) if B_ in set_]
             
             label_I.reverse()
             label_names_reord = label_A+label_I+label_B
             
             cmap = pl.cm.Set1(np.linspace(0,1,macrostate))
             colors = []
             for reord in label_names_reord:
                 idx = label_names.index(reord)
                 colors.append(cmap[idx])

        else:
            flux=pyemma.msm.tpt(msm, A, B)
            label_names = [f'state {i}' for i in range(1, self.n_state+1)]
        
        #Calculate commmittors
        index_row_comm=pd.MultiIndex.from_product([[self.ft_name], [self.feature], [f'{self.n_state}@{lag_}'], label_names_reord], names=['name', 'feature', 'model', 'states'])
        f_committor=flux.forward_committor
        b_committor=flux.backward_committor
        print(label_B, label_names_reord)
        
        committor_df=pd.DataFrame(zip(f_committor, b_committor), index=index_row_comm, columns=['Forward', 'Backward'])
        
        #Calculate fluxes
        index_row= pd.MultiIndex.from_product([[self.ft_name], [self.feature], [f'{self.n_state}@{lag_}']], 
                                              names=['name', 'feature', 'model'])
        net_flux = flux.net_flux
        net_flux_s = in_units_of(net_flux, f'second/{self.timestep.unit.get_name()}', 'second/second')
        rate_s = in_units_of(flux.rate, f'second/{self.timestep.unit.get_name()}', 'second/second')
        major_flux = flux.major_flux()
        gross_flux = flux.gross_flux
        total_flux = flux.total_flux
        if (net_flux.all() != major_flux.all() != gross_flux.all() != total_flux.all()):
            print('\tWarning! fluxes are not the same')
        unit = r'$s^{-1}$'
        report_flux = "{:.1e} {}".format(net_flux_s.sum(), unit)
        report_rate = "{:.1e} {}".format(rate_s, unit)

        
        #In units of gets the conversion of timestep to second
        flux_df=pd.DataFrame(columns=['net flux', 'rate'], index=index_row)
        flux_df.loc[:, 'net flux'] = net_flux_s.sum(1).sum()
        flux_df.loc[:, 'rate'] = rate_s
        label_names_reord = label_A+label_I+label_B
        fig, axes = plt.subplots(1,1, figsize=(10, 8))
        pyemma.plots.plot_flux(flux,
                               flux_scale = in_units_of(1, f'second/{self.timestep.unit.get_name()}', 'second/second'),
                               pos=None, 
                               state_sizes = flux.stationary_distribution,
                               show_committor=True,
                               state_labels = label_names_reord,
                               show_frame=False,
                               state_colors = colors,
                               arrow_curvature = 2,
                               state_scale = 0.5,
                               figpadding=0.1,
                               max_height=6,
                               max_width=8,
                               arrow_label_format='%.1e  / s',
                               ax=axes)
        axes.set_title(f'Flux {self.full_name}' +
                       f'\n{" ".join(label_A)} -> {" ".join(label_B)}' + 
                       f'\nRate: {report_rate} Net Flux: {report_flux}', 
                  ha='center',
                  fontsize=12)
        axes.axvline(0, linestyle='dashed', color='black')
        axes.axvline(1, linestyle='dashed', color='black')
        axes.grid(True, axis='x', linestyle='dashed')
        axes.set_xlim(-0.15, 1.15)
        axes.set_xticks(np.arange(0,1.1,0.1))
        plt.show()
        fig.savefig(f'{self.results}/Flux_{self.full_name}_{label_names_reord[0]}_{label_names_reord[-1]}')
        
        #Calculate the pathways	
        paths, path_fluxes = flux.pathways(fraction=0.95)
        path_labels=[[] for _ in paths]

        for idx, path in enumerate(paths):
            for p in path:
                path_labels[idx].append(label_names_reord[p])
        path_labels=[' -> '.join(k) for k in path_labels]
        index_col_path = pd.MultiIndex.from_product([[self.ft_name], [self.feature], [f'{self.n_state}@{lag_}'], path_labels], 
                                                    names=['name', 'feature', 'model', 'pathway'])
        #path_fluxes_s = in_units_of(path_fluxes, f'second/{self.timestep.unit.get_name()}', 'second/second')
        if len(path_fluxes) > 1:
            pathway_df=pd.DataFrame(((path_fluxes/np.sum(path_fluxes))[:top_pathways])*100, index=index_col_path[:top_pathways], columns=['% Total'])
        else:
            pathway_df=pd.DataFrame((path_fluxes/np.sum(path_fluxes))*100, index=index_col_path, columns=['% Total'])

# =============================================================================
#             if scheme == 'combinatorial':    
#                 label_names=Functions.sampledStateLabels(regions, sampled_states=states, labels=labels)
# =============================================================================               
        return flux_df, committor_df, pathway_df


    def MFPT_calculation(self, msm, macrostates=None):
        
        file_name=f'MFPT_{self.full_name}'
        mfpt = self.mfpt_states(msm, file_name)
        if macrostates != None:
            for macrostate in macrostates:
                msm.pcca(macrostate)
                self.mfpt_cg_states(mfpt, msm, file_name, macrostate)
    
    
    def mfpt_cg_states(self, mfpt_all, msm, file_name, n_macrostate):
        
        macrostates= range(1, n_macrostate+1)
        file_name = f'{file_name}_{n_macrostate}macrostates'
        
        mfpt = np.zeros((n_macrostate, n_macrostate))
        for i in macrostates:
            for j in macrostates:
                print(f'Transition MS: {i} -> {j}', end='\r')
                mfpt[i-1, j-1] = msm.mfpt(msm.metastable_sets[i-1], msm.metastable_sets[j-1])
        print('\n')
        cmap_plot=plt.cm.get_cmap("gist_rainbow")
        fig, axes = plt.subplots(1,1, figsize=(8,6), constrained_layout=True)
        matrix = axes.pcolormesh(mfpt, 
                                 cmap=cmap_plot, 
                                 shading='auto',
                                 norm=ml_colors.LogNorm(vmin = self.lag, vmax=mfpt.max().max()),
                                 vmin=self.lag,
                                 vmax=mfpt.max().max())
                                 #vmin=mfpt_all[mfpt_all > 0].min().min(),
                                 #vmax=mfpt_all.max().max(),
                                 #norm=ml_colors.LogNorm(vmin = mfpt_all[mfpt_all > 0].min().min(), vmax = mfpt_all.max().max())) 
        colorbar = plt.colorbar(matrix, orientation='vertical')
        colorbar.set_label(label=f'MFPT ({self.unit})', size='large')
        colorbar.ax.tick_params(labelsize=14)
        axes.set_title(f'MFPT {self.full_name} -> {n_macrostate} macrostates')
        axes.set_xticks(macrostates)
        axes.set_yticks(macrostates)
        cmap_plot.set_bad(color='white')
        axes.set_xlabel('From macrostate')
        axes.set_ylabel('To macrostate')
        cmap_plot.set_under(color='black')
        cmap_plot.set_bad(color='white')
        cmap_plot.set_over(color='grey')
        #plt.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
        
        #pyemma.plots.plot_network(inverse_mfpt, arrow_label_format=f'%f. {self.unit}', arrow_labels=mfpt, size=12) 
        plt.show()
        
        return mfpt
    
    
    def mfpt_states(self, msm, file_name):
        
        states = range(1, self.n_state+1)
        index_row=pd.MultiIndex.from_product([[s for s in states]], names=['source'])
        index_col=pd.MultiIndex.from_product([[s for s in states], ['mean', 'stdev']], names=['sink', 'values'])
        
        if not os.path.exists(f'{self.results}/{file_name}.csv') or self.overwrite:
            mfpt=pd.DataFrame(index=index_row, columns=index_col)
            print('\tCalculating MFPT matrix...')
            for i in states:
                for j in reversed(states):
                    mfpt.loc[(i), (j, 'mean')]= msm.sample_mean("mfpt", i-1,j-1)
                    mfpt.loc[(i), (j, 'stdev')]= msm.sample_std("mfpt", i-1,j-1)
                    print(f'\t\ttransition: {i} -> {j} ', end = '\r')
            print('\n')
            mfpt.to_csv(f'{self.results}/{file_name}.csv')
        
        #reload again otherwise cannot be processed pcolormesh.
        mfpt = pd.read_csv(f'{self.results}/{file_name}.csv', index_col = 0, header = [0,1]) 
        means = MSM.mfpt_filter(mfpt, 0)
        cmap_plot=plt.cm.get_cmap("gist_rainbow")
        
        fig, axes = plt.subplots(1,1, figsize=(8,6), constrained_layout=True)
        matrix = axes.pcolormesh(means, 
                                 cmap=cmap_plot,
                                 shading='auto',
                                 vmin=self.lag,
                                 vmax=means.max().max(),
                                 norm=ml_colors.LogNorm(vmin = self.lag, vmax = means.max().max()))    #means[means > 0].min().min()
        colorbar = plt.colorbar(matrix, orientation='vertical')
        colorbar.set_label(label=f'MFPT ({self.unit})', size='large')
        colorbar.ax.tick_params(labelsize=14)
        axes.set_title(f'MFPT {self.full_name}')
        axes.set_xticks(states)
        axes.set_yticks(states)
        cmap_plot.set_bad(color='white')
        axes.set_xlabel('From state')
        axes.set_ylabel('To state')
        cmap_plot.set_under(color='black')
        cmap_plot.set_bad(color='white')
        cmap_plot.set_over(color='grey')
        #plt.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")

        return mfpt
    

    def extract_states(self, msm, macrostate, n_total_frames=20, visual=False):

        msm_file_name=f'bayesMSM_{self.full_name}'
        
        self.set_systems()
        
        topology=self.tops_to_load
        trajectories=self.trajectories_to_load
        
        superpose_indices=md.load(topology).topology.select(MSM.superpose)
        subset_indices=md.load(topology).topology.select(MSM.subset)
        md_top=md.load(topology) #, atom_indices=ref_atom_indices )
        
        if self.pre_process:
            trajectories=Trajectory.Trajectory.pre_process_MSM(trajectories, md_top, superpose_to=superpose_indices)
            md_top.atom_slice(subset_indices, inplace=True)
        state_samples= {} 
        #TODO: get total number of frames, to get proper n_samples
        if macrostate != None:
            
            h_members = []
            msm.pcca(macrostate)
            for idx, idist in enumerate(msm.metastable_distributions, 1):
                
                highest_member = idist.argmax()
                #TODO: resolve conflict of low assign states take frames from same micro. only problematic for lowe assig cases anyways for now.
                #if highest_member in h_members:
                    #idist = np.delete(idist, highest_member)
                    #highest_member = idist.argmax()
                h_members.append(highest_member)
                dist = msm.pi[msm.metastable_sets[idx-1]].sum()
                n_frames = int(np.rint(dist*n_total_frames))
                if n_frames == 0:
                    n_frames +=1
                n_state_frames = int(np.rint(dist*MSM.ref_samples))
                if n_state_frames == 0:
                    n_state_frames += 1

                if visual:
                    file_name=f'{self.stored}/{msm_file_name}_{idx}of{macrostate}macrostates_v.pdb'
                    sample = msm.sample_by_state(n_frames, subset=[highest_member], replace = False)
                    if not os.path.exists(file_name) or self.overwrite:
                        print(f'\tGenerating {n_frames} samples for macrostate {idx} using frames assigned to state {highest_member+1}')
                        try:
                            pyemma.coordinates.save_traj(trajectories, sample, outfile=file_name, top=md_top)
                        except ValueError as v:
                            print(v)
                    state_samples[idx] = (file_name, file_name)
                else:
                    file_name=f'{self.stored}/{msm_file_name}_{idx}of{macrostate}macrostates_{n_state_frames}'
                    sample = msm.sample_by_state(n_state_frames, subset=[highest_member], replace = False)
                    if not os.path.exists(f'{file_name}.dcd') or self.overwrite:
                        print(f'\tGenerating {n_state_frames} samples for macrostate {idx} using frames assigned to state {highest_member+1}')
                        pyemma.coordinates.save_traj(trajectories, sample, outfile=f'{file_name}.dcd', top=md_top)
                        md.load_frame(f'{file_name}.dcd', index=0, top=md_top).save_pdb(f'{file_name}.pdb')
                    state_samples[idx] = (f'{file_name}.dcd', f'{file_name}.pdb')
        #TODO: make single state extraction.
        
        return state_samples


    def visualize_metastable(self, samples, opt_index=1449, opt_index_end=1704):
        """
        

        Parameters
        ----------
        samples : TYPE
            DESCRIPTION.
        opt_index : TYPE, optional
            DESCRIPTION. The default is 1449.
        opt_index_end : TYPE, optional
            DESCRIPTION. The default is 1704.

        Returns
        -------
        widget : TYPE
            DESCRIPTION.

        """

        from matplotlib.colors import to_hex
    
        cmap = pl.cm.get_cmap('Set1', len(samples))
    
        widget = nglview.NGLWidget()
        widget.clear_representations()
        #TODO: make opt_index robust. query input topology for first residue value

        ref = md.load_pdb(self.project.input_topology)
        n_residues = ref.n_residues
        resid_range_ = [int(word) for word in self.selections.split() if word.isdigit()]
        if len(resid_range_) == 0:
            print('Could not assign inputs to resid ranges, reverting to "all".')
            resid_range_ = [1, n_residues]
        
        resid_range = '-'.join([str(resid+opt_index) for resid in resid_range_])
        prev_range_ = [opt_index, resid_range_[-1]+opt_index]
        prev_range = '-'.join([str(resid) for resid in prev_range_])
        post_range_ = [resid_range_[-1]+opt_index, opt_index_end]
        post_range = '-'.join([str(resid) for resid in post_range_])
        
        ref_struct = widget.add_trajectory(ref, default=False)
        ref_struct.add_cartoon(resid_range, color='black')
        ref_struct.add_cartoon(prev_range, color='black', opacity=0.3)
        ref_struct.add_spacefill('_ZN', color='black') #'246-248'
        ref_struct.add_licorice('SAM', color='black')

        x = np.linspace(0, 1, num=len(samples))
        for idx, s in samples.items():
            (traj_file, top) = s

            c = to_hex(cmap(x[idx-1]))
            print('\tGenerating images of (macro)state', idx, end='\r')
            traj = md.load(traj_file, top=top)
            traj.superpose(traj[0])
            

            component = widget.add_trajectory(traj, default=False)
            component.add_cartoon(resid_range, color=c)
            component.add_cartoon(prev_range, color=c, opacity=0.2)
            component.add_cartoon(post_range, color=c, opacity=0.2)
            component.add_spacefill('_ZN', color=c) #'246-248' #ZNB
            component.add_licorice('SAM', color=c) #SAM

        print('\n')
        widget.center()
        widget.camera = 'perspective'
        widget.stage.set_parameters(**{
                                 "clipNear": 0, "clipFar": 100, "clipDist": 1,
                                 "backgroundColor": "white",})   

        return widget


    
    def spectral_analysis(self, msm, lag, plot=False):
    #from pyEMMA notebook
        
        def its_separation_err(ts, ts_err):
            """
            Error propagation from ITS standard deviation to timescale separation.
            """
            return ts[:-1] / ts[1:] * np.sqrt(
                (ts_err[:-1] / ts[:-1])**2 + (ts_err[1:] / ts[1:])**2)


        dt_scalar=int(str(self.timestep*self.stride).split(' ')[0])
        
        timescales_mean = msm.sample_mean('timescales')
        timescales_std = msm.sample_std('timescales')



        #Filter number os states above 0.95 range of ITS
        state_cutoff=-1
        factor_cutoff =  (1 - MSM.msm_var_cutoff) * (max(timescales_mean)-min(timescales_mean))
        for idx, t in enumerate(timescales_mean):
            if t < factor_cutoff:
                #print(f'Factor {factor_cutoff}, {t} {idx}')
                state_cutoff = idx
                break
            
        imp_ts=range(1, len(timescales_mean)+1)
        if factor_cutoff > dt_scalar*lag:
            timescales_mean_=timescales_mean[:state_cutoff]
            timescales_std_=timescales_std[:state_cutoff]

            imp_ts_factor=imp_ts[:state_cutoff]
            factor = timescales_mean_[:-1]/timescales_mean_[1:]
            factor_labels = [f'{k}/{k+1}' for k in imp_ts_factor[:-1]]
            prop_error=its_separation_err(timescales_mean_, timescales_std_)

            if plot:
                print(f'\t{len(imp_ts_factor)} processes resolved above {MSM.msm_var_cutoff*100}% ITS ({self.timestep*lag})')
                fig, axes = plt.subplots(1, 2, figsize=(8,6))
                
                axes[0].plot(imp_ts, timescales_mean, marker='o')
                axes[0].fill_between(imp_ts, timescales_mean-timescales_std, timescales_mean+timescales_std, alpha=0.8)
                #axes[0].set_xticks(imp_ts)
                #axes[0].axvline(state_cutoff, lw=1.5, color='green', linestyle='dotted')
                axes[0].axhline(factor_cutoff, lw=1.5, color='red', linestyle='dashed')
                axes[0].axhline(dt_scalar*lag, lw=1.5, color='red', linestyle='solid')
                axes[0].axhspan(0, dt_scalar*lag, alpha=0.5, color='red')
                axes[0].axhspan(dt_scalar*lag, factor_cutoff, alpha=0.2, color='red')
                axes[0].set_xlabel('implied timescale index')
                axes[0].set_ylabel(self.unit)
                axes[0].ticklabel_format(axis='y', style='sci')
                axes[0].set_yscale('log')
                axes[0].set_ylim(bottom=1)
                axes[0].set_title('Implied timescales')
                
                axes[1].errorbar(imp_ts_factor[:-1], factor, yerr=prop_error, marker='o')
                #axes[1].fill_between(imp_ts_factor[:-1], factor-prop_error, factor+prop_error, alpha=0.5)
                axes[1].grid(True, axis='x')
                axes[1].set_xticks(imp_ts_factor[:-1])
                axes[1].set_xticklabels(factor_labels, rotation=45)
                axes[1].set_xlabel('implied timescale indices')
                axes[1].set_ylabel('Factor')
                axes[1].set_title('Timescale separation')
                fig.suptitle(f'ITS decompositon of {self.full_name}', weight='bold', fontsize=12)
                fig.tight_layout()
    
                fig.savefig(f'{self.results}/SpectralITS_{self.full_name}.png', dpi=600, bbox_inches="tight")
                plt.show()
            
            print(f'\t{len(imp_ts_factor)} processes with {MSM.msm_var_cutoff*100}% ITS resolved above lag time ({self.timestep*lag})')
            return len(imp_ts_factor)

        else:
            return 0

       
    def RMSD_source_sink(self, file_names, ref_sample=2000, pdb_files=[]):
        
        df = pd.DataFrame()
        #TODO: make init robust to initial residue of structure
        selection = self.convert_resid_residue(self.selections, init=1449)
        source_traj = md.load(self.project.input_topology)
        colors= pl.cm.Set1(np.linspace(0,1,len(file_names)))
        
        rows, columns=tools_plots.plot_layout(pdb_files)
        fig, axes = plt.subplots(rows, columns, sharey=True, sharex=True, constrained_layout=True, figsize=(12,8))
        
        flat = axes.flat
        for pdb, ax in zip(pdb_files, flat):
            for idx, file_name in file_names.items():
                (traj, top) = file_name
                traj= md.load(traj, top=top)
                frames=traj.n_frames
                weight= frames / ref_sample
                
                indexes_source=[['RMSD'], [idx], ['source']]
                indexes_sink=[['RMSD'], [idx], [pdb]]
                names=['name', 'macrostates', 'reference']
                sink_traj = md.load(f'{self.project.def_input_struct}/pdb/{pdb}.pdb')
                
                atom_indices=traj.topology.select(selection)
                source_atom_indices=source_traj.topology.select(selection)
                sink_atom_indices=sink_traj.topology.select(selection)
                
                ax.set_title(pdb)
                
                try:                    
                    rmsd_sink=in_units_of(md.rmsd(traj,
                                     sink_traj,
                                     atom_indices =atom_indices,
                                     ref_atom_indices = sink_atom_indices,
                                     frame=0,
                                     precentered=False,
                                     parallel=True), 'nanometers', MSM.report_units)
                    ax.hist(rmsd_sink, 
                            label=f'MS {idx}',
                            density=True,
                            weights=[weight]*len(rmsd_sink), 
                            color=colors[idx-1], 
                            alpha=0.6,
                            histtype='barstacked',
                            stacked=True)
                    rows=pd.Index(np.arange(1, len(rmsd_sink)+1), name='sample')
                    df_macro = pd.DataFrame(rmsd_sink, index=rows, columns=pd.MultiIndex.from_product(indexes_sink, names=names))
                    df=pd.concat([df, df_macro], axis=1)
                    
                except ValueError:
                    pass
        #[-1].legend()
        #axes[1].legend()
        handles, labels = flat[0].get_legend_handles_labels()
        def_locs=(1.1, 0.6)
        fig.legend(handles, labels, bbox_to_anchor=def_locs)
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.suptitle(f'RMSD: {ref_sample} samples\nModel: {self.full_name}')
        fig.text(0.5, -0.04, f'RMSD ({MSM.report_units})', ha='center', va='center', fontsize=12)
        fig.text(-0.01, 0.5, 'Probability density', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.tight_layout()
        plt.show()
        
        return df


    def PCCA_calculation(self, msm, macrostates, auto=False, dims_plot=10):
        """
        

        Parameters
        ----------
        macrostates : TYPE
            DESCRIPTION.

        Returns
        -------
        pccas : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.mute = True
        if macrostates == None or auto:
            print('Retrieving number of macrostates from spectral analysis')
            macrostates = [self.spectral_analysis(self.full_name, msm, self.lag, plot=False)]
        
        data_concat, disc_trajs, clusters = self.load_discTrajs()
        #loop through sets of macrostates
        for macrostate in macrostates:
            if macrostate != 0:
                print(f'\tPCCA with {macrostate} MS')
                msm.pcca(macrostate)
                self.plot_PCCA(msm, macrostate, data_concat, disc_trajs, clusters)
            else:
                print('\tPCCA skipped')

    def plot_PCCA(self, msm, macrostate, data_concat, disc_trajs, clusters):
        
        #data_concat, disc_trajs, clusters = self.load_concatData_discTrajs(self.discretized_data)
        #disc_trajs = np.concatenate(clusters.dtrajs)
        metastable_traj = msm.metastable_assignments[np.concatenate(disc_trajs)]
        highest_membership = msm.metastable_distributions.argmax(1)
        
        f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))
        
        file_name=f'PCCA_{self.full_name}_{macrostate}MS'
        fig=plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(nrows=4, ncols=3)
        colors= pl.cm.Set1(np.linspace(0,1,macrostate))
        fig.suptitle(f'MSM {self.full_name}\n PCCA+ : {self.n_state} states to {macrostate} MS', 
                     ha='center',
                     weight='bold')
        
        statdist = msm.stationary_distribution
        assignments=msm.metastable_assignments
        
        plot_statdist_states = fig.add_subplot(gs[0, :2]) 
        plot_statdist_states.set_xlabel('State index')
        plot_statdist_states.set_ylabel('Stationary distribution')
        plot_statdist_states.set_ylim(0,1)
        plot_statdist_states.yaxis.grid(color='grey', linestyle='dashed')
        #plot_statdist_states.set_yscale('log')
        for x, stat in zip(range(1, len(statdist)+1), statdist):
            plot_statdist_states.bar(x, stat, color=colors[assignments[x-1]])
        
        plot_assign_ms = fig.add_subplot(gs[1,:2], sharex=plot_statdist_states, sharey=plot_statdist_states) 
        plot_assign_ms.set_xlabel('State index')
        plot_assign_ms.set_ylabel('MS assignment')
        plot_assign_ms.yaxis.grid(color='grey', linestyle='dashed')
        for idx, met_dist in enumerate(msm.metastable_distributions):
                plot_assign_ms.bar(range(1, len(statdist)+1), met_dist, color=colors[idx], label=f'MS {idx+1}')
        plot_assign_ms.legend()
        
        statdist_cg=[]
        plot_statdist_ms = fig.add_subplot(gs[:2,2], sharey=plot_statdist_states)
        plot_statdist_ms.set_xlabel('Macrostate')
        plot_statdist_ms.set_ylabel('MS stationary distribution')
        plot_statdist_ms.yaxis.grid(color='grey', linestyle='dashed')
        plot_statdist_ms.set_xticks(range(1,macrostate+1))
        #plot_statdist_ms.set_yscale('log')
         
        for idx, s in enumerate(msm.metastable_sets):
            statdist_m = msm.pi[s].sum()
            plot_statdist_ms.bar(idx+1, statdist_m, color=colors[idx]) #, label=f'MS {idx+1}')
            statdist_cg.append(statdist_m)
        #plot_statdist_ms.legend() 

        plot_ms_trajs = fig.add_subplot(gs[2,:])
        plot_ms_trajs.plot(range(1,len(metastable_traj)+1), metastable_traj+1, lw=0.5, color='black')
        plot_ms_trajs.set_yticks(range(1, macrostate+1))
        plot_ms_trajs.set_ylabel('Macrostate')
        plot_ms_trajs.set_xlabel('Discretized frames')
        plot_ms_trajs.xaxis.set_major_formatter(mtick.FuncFormatter(g))
        plot_ms_trajs.set_xlim(0, len(metastable_traj+1))

# =============================================================================
#         plot_tica = fig.add_subplot(gs[3, 0])
#         pyemma.plots.plot_free_energy(*self.tica_concat[:, :2].T, ax=plot_tica) #, legacy=False)
#         plot_tica.set_xlabel('IC 1')
#         plot_tica.set_ylabel('IC 2')
#         plot_tica.set_title('TICA')
# =============================================================================
        
        plot_msm_tica = fig.add_subplot(gs[3, 0])
        pyemma.plots.plot_free_energy(*data_concat[:, :2].T, 
                                      weights=np.concatenate(msm.trajectory_weights()), 
                                      ax=plot_msm_tica) #, legacy=False)
        plot_msm_tica.set_xlabel('IC 1')
        plot_msm_tica.set_ylabel('IC 2')
        plot_msm_tica.set_title('MSM')
        
        plot_eigen = fig.add_subplot(gs[3, 1])
        eigvec = msm.eigenvectors_right()        
        coarse_state_centers = clusters.clustercenters[msm.active_set[highest_membership]]

        pyemma.plots.plot_contour(*data_concat[:, :2].T, 
                                  eigvec[np.concatenate(disc_trajs), 1], 
                                  ax=plot_eigen, 
                                  cbar_label='2nd right eigenvector', 
                                  mask=True, 
                                  cmap='PiYG') 
        #another way to plot them all without colors
        #plot_eigen.scatter(*coarse_state_centers[:, :2].T, s=15, c='C1')
        plot_eigen.set_xlabel('IC 1')
        plot_eigen.set_ylabel('IC 2')
        
        
        
        
        plot_ms_tica = fig.add_subplot(gs[3, 2])
        _, _, misc = pyemma.plots.plot_state_map(*data_concat[:, :2].T, 
                                                 metastable_traj, 
                                                 ax=plot_ms_tica, 
                                                 cmap='Set1',
                                                 alpha=0.9)
        plot_ms_tica.set_xlabel('IC 1')
        plot_ms_tica.set_ylabel('IC 2')
        plot_ms_tica.set_title('PCCA+')
        for center, color in zip(coarse_state_centers, colors):
            plot_ms_tica.scatter(x=center[0], y=center[1], edgecolors='black', color=color)
        #misc['cbar'].set_ticklabels([r'$\mathcal{S}_%d$' % (i + 1)for i in range(macrostate)])
        misc['cbar'].set_ticklabels([i for i in range(1,macrostate+1)])
        misc['cbar'].set_label('MS')
        fig.tight_layout()
        plt.show()
        try:
            fig.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
        except OverflowError:
            print('Warning! Figure is getting too complex. Reducing dpi to 300.')
            fig.savefig(f'{self.results}/{file_name}.png', dpi=300, bbox_inches="tight")


    def CKTest_calculation(self, msm, macrostates):
        """
        

        Parameters
        ----------
        macrostates : TYPE
            DESCRIPTION.
        mlags : TYPE, optional
            DESCRIPTION. The default is 7.

        Returns
        -------
        ck_tests : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.show_progress_bars = False
        
        f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))
        for macrostate in macrostates:
            file_name = f'CKtest_{self.full_name}_{macrostate}macrostates'
            if not os.path.exists(f'{self.stored}/{file_name}.npy') or self.overwrite:
                print(f'\tPerforming CK test for {self.full_name} -> {macrostate} macrostates')
                cktest=msm.cktest(macrostate)
                cktest.save(f'{self.stored}/{file_name}.npy', overwrite=True)
            else:
                print('\tCK test found for: ', self.full_name)
                cktest=pyemma.load(f'{self.stored}/{file_name}.npy')
    
            dt_scalar=int(str(self.timestep*self.stride).split(' ')[0])
            ck_plot, axes=pyemma.plots.plot_cktest(cktest, 
                                                   dt=dt_scalar, 
                                                   layout='wide', 
                                                   marker='.',  
                                                   y01=True, 
                                                   units=self.unit)
            for ax in axes:
                for subax in ax:
                    subax.xaxis.set_major_formatter(mtick.FuncFormatter(g))    
                #TODO (maybe?): play around with subplot properties
    # =============================================================================
    #             for ax in axes:
    #                 for subax in ax:
    #                     #subax.set_xscale('log')
    # =============================================================================
            ck_plot.suptitle(f'CK test: {self.full_name} -> {macrostate} macrostates', va='top', ha='center', weight='bold', fontsize=12)
            ck_plot.tight_layout()
            ck_plot.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
            plt.show()

    def bayesMSM_calculation(self,   
                             lag, 
                             msm_stride=1,
                             statdist=None) -> dict:
        """
        

        Parameters
        ----------
        discretized_data : TYPE
            DESCRIPTION.
        states : TYPE
            DESCRIPTION.
        lags : TYPE
            DESCRIPTION.
        statdist : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        #Important legacy
        #data=np.array(np.load(self.values['discretized'][0]))
        #data_i=np.ascontiguousarray(data[0::msm_stride])

        
        #TODO: make statdist into arguments for msm construction.
        file_name=f'{self.stored}/bayesMSM_{self.full_name}.npy'
        if not os.path.exists(file_name) or self.overwrite:
            _, disc_trajs, _ = self.load_discTrajs()
            try:
                print(f'\tGenerating Bayesian MSM for {self.full_name}')
                bayesMSM=pyemma.msm.bayesian_markov_model(disc_trajs[0::msm_stride], lag=lag, dt_traj=str(self.timestep), conf=0.95)
                bayesMSM.save(file_name, overwrite=True)
            except Exception as v:
                print(v)
             
        else:
            bayesMSM=pyemma.load(file_name)
            print(f'\tBayesian MSM found for {self.full_name}')

        return bayesMSM
        
    def ITS_calculation(self, c_stride=1) -> dict:

        pyemma.config.show_progress_bars = False
        data_concat, disc_trajs, clusters = self.load_discTrajs(self.discretized_data)
        file_name = f'ITS_{self.ft_base_name}'            
        print(f'Generating ITS profile for {self.ft_base_name}')
        its=None
        its_stride=1
        while its == None and its_stride < 10000:
            try:
                its_name=f'{self.stored}/{file_name}_{its_stride}sits.npy'
                if not os.path.exists(its_name) or self.overwrite:                
                    print(f'\tCalculating ITS with trajectory stride of {its_stride}', end='\r')
                    its=pyemma.msm.its(disc_trajs[0::its_stride], lags=MSM.ITS_lags, errors='bayes')
                    its.save(f'{self.stored}/{file_name}_{its_stride}sits.npy', overwrite=self.overwrite)  
                else:
                    print(f'ITS profile found for: {self.ft_base_name} {its_stride} sits')
                    its=pyemma.load(its_name)
            except:
                print('\tWarning!Could not generate ITS. Increasing stride.')
                its_stride=its_stride*2      
        dt_scalar=int(str(self.timestep*(self.stride*its_stride)).split(' ')[0])
        its_plot=pyemma.plots.plot_implied_timescales(its, units=self.unit, dt=dt_scalar)
        its_plot.set_title(f'ITS of {self.ft_base_name}', ha='center', weight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{self.results}/{file_name}.png', dpi=600, bbox_inches="tight")
        plt.show()
    
    def TICA_calculation(self,
                         lag,
                         feature,
                         dim=-1):
        """
        

        Parameters
        ----------
        lag : TYPE
            DESCRIPTION.
        feature : TYPE
            DESCRIPTION.
        dim : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        tica : TYPE
            DESCRIPTION.

        """
        file_name = f'{self.stored}/TICA_{self.ft_base_name}.npy'
        if not os.path.exists(file_name) or self.overwrite:                  
            data = self.load_features(self.tops_to_load, 
                                      self.trajectories_to_load, 
                                      [feature], 
                                      inputs=self.selections)[0] #[0] because its yielding a list.
            print(f'\tCalculating TICA of {self.ft_base_name}')
            try:
                tica = pyemma.coordinates.tica(data, 
                                               lag=lag, 
                                               dim=dim,
                                               var_cutoff=MSM.var_cutoff,
                                               skip=self.skip, 
                                               stride=self.stride, 
                                               chunksize=self.chunksize)
                tica.save(file_name, save_streaming_chain=True)
                self.plot_TICA(tica, lag, dim, feature)
            except ValueError:
                print(f'Warning! Failed for {self.ft_base_name}') 
                print('Probably trajectories are too short for selected lag time and stride')
                pass
        else:
            print(f'\tFound TICA of {self.ft_base_name}')
            tica = pyemma.load(file_name)
        
        n_total_frames = tica.n_frames_total()
        n_trajs = tica.number_of_trajectories()
        min_traj, max_traj = tica.trajectory_lengths().min(), tica.trajectory_lengths().max()
        print(f'\t\tTotal number of frames: {n_total_frames}\n\t\tNumber of trajectories: {n_trajs}\n\t\tFrames/trajectory: [{min_traj}, {max_traj}]')     
 
        return tica

    def plot_TICA(self, 
                  tica,
                  lag,
                  dim,  
                  name, 
                  opt_dim=10):
        """
        

        Parameters
        ----------
        tica : TYPE
            DESCRIPTION.
        dim : TYPE
            DESCRIPTION.
        lag : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        opt_dim : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        Notes
        -----
        
        tica_concat is the trajectories concatenated.
        ticas is the individual trajectories
        """
         
        f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))
        
        if tica.dimension() > 10 and dim == -1:
            print(f'\tWarning: TICA for {MSM.var_cutoff*100}% variance cutoff yields {tica.dimension()} dimensions.')
            print(f'Reducing to {opt_dim} dimensions for visualization only.')
            dims_plot=opt_dim
        elif tica.dimension() > dim and dim > 0:
            dims_plot=tica.dimension()
        colors= pl.cm.Accent(np.linspace(0,1,dims_plot))
        tica_concat = np.concatenate(tica.get_output())
        
        fig=plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(nrows=1, ncols=2)
            
        #plot histogram
        ax0 = fig.add_subplot(gs[0, 0])
        pyemma.plots.plot_feature_histograms(tica_concat[:, :dims_plot], ax=ax0)
        ax0.set_ylabel('IC') #, ylog=True)
        ax0.set_title('Histogram')
            
        #plot projection along main components
        ax1 = fig.add_subplot(gs[0, 1])
        pyemma.plots.plot_free_energy(*tica_concat[:, :2].T, ax=ax1, legacy=False) #, logscale=True)
        ax1.set_xlabel('IC 1')
        ax1.set_ylabel('IC 2')
        ax1.set_title('IC density')

        fig.suptitle(fr'TICA: {self.ft_name} {name} @ $\tau$ ={self.timestep*lag}, {tica.dimension()} dimensions', ha='center', weight='bold')
        fig.tight_layout()   
        fig.savefig(f'{self.results}/TICA_{self.ft_base_name}.png', dpi=600, bbox_inches="tight")
        plt.show()

        #plot discretized trajectories
        ticas=tica.get_output()
        
        rows, columns=tools_plots.plot_layout(len(ticas)+1)
        fig_trajs,axes_=plt.subplots(columns, rows, sharex=True, sharey=True, constrained_layout=True, figsize=(12,12))
        #fig_trajs.subplots_adjust(hspace=0, wspace=0)

        try:
            flat=axes_.flat
        except AttributeError:
            flat=[axes_]     

        #loop for the trajectories
        for (idx_t, trj_tica), ax in zip(enumerate(ticas), flat): #len trajs?
            
            
            x = self.timestep * np.arange(trj_tica.shape[0])
    
            for idx, tic in enumerate(trj_tica.T):
                
                if idx < dims_plot:
                    ax.plot(x, tic, label=f'IC {idx}', color=colors[idx])
                    ax.text(.5,.85, idx_t+1, horizontalalignment='center', transform=ax.transAxes)
                    ax.xaxis.set_major_formatter(mtick.FuncFormatter(g))
        fig_trajs.suptitle(fr'TICA: {self.ft_name} {name} @ $\tau$ ={self.timestep*lag}', weight='bold')
        handles, labels = ax.get_legend_handles_labels()
        def_locs=(1.1, 0.6)
        fig_trajs.legend(handles, labels, bbox_to_anchor=def_locs)
        fig_trajs.tight_layout()
        fig_trajs.subplots_adjust(hspace=0, wspace=0)
        fig_trajs.text(0.5, 0, f'Trajectory time ({self.unit})', ha='center', va='center', fontsize=12)
        fig_trajs.text(-0.03, 0.6, 'IC value', ha='center', va='center', rotation='vertical', fontsize=12)
        fig_trajs.savefig(f'{self.results}/TICA_{self.ft_base_name}_discretized.png', dpi=600, bbox_inches="tight")
        plt.show()

    
    def VAMP2_dimensions(self, lags):
        """
        

        Parameters
        ----------
        lags : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        rows, columns=tools_plots.plot_layout(self.features)
        fig, axes = plt.subplots(rows, columns, figsize=(9,6))

        try:
            flat=axes.flat
        except AttributeError:
            flat=axes
            flat=[flat]

        for ax, feature in zip (flat, self.features):
            
            data, dimensions = self.load_features(self.tops_to_load, 
                                      self.trajectories_to_load, 
                                      [feature], 
                                      inputs=self.selections,
                                      get_dims=True)[0] #[0] because its yielding a list.

                        
            dims = [int(dimensions*i) for i in np.arange(0.1, 1.1, 0.1)]
            legends = []

            for lag in lags:
                print(f'\tLag: {lag*self.timestep}\n' )
                scores_dim= []
                #scores_=np.array([self.score_cv(data, dim, lag)[1] for dim in dims])

                for dim in dims:
                    try:
                        scores_dim.append(self.score_cv(data, dim, lag, number_of_splits=10)[1]) #discard vamps [0]

                    except MemoryError:
                        print('\tWarning! Failed due to memory error.')
                        scores_dim.append(np.full(10, 0))

                    except ValueError:
                        print('\tWarning! Failed due to too large lag, stride combination.')
                        scores_dim.append(np.full(10, 0))

                scores_ = np.asarray(scores_dim)
                scores = np.mean(scores_, axis=1)
                errors = np.std(scores_, axis=1, ddof=1)
                #color = 'C{lags.index(lag)}'
                ax.fill_between(dims, scores - errors, scores + errors, alpha=0.3) #, facecolor=color)
                ax.plot(dims, scores, '--o')
                legends.append(r'$\tau$ = '+f'{self.timestep*lag}')
            #ax.legend()
            ax.set_title(feature)
            ax.set_xlabel('number of dimensions')
            ax.set_ylabel('VAMP2 score')
        
        fig.suptitle(f'VAMP2 of {self.ft_name}\nstride {self.stride*self.timestep}', ha='center', weight='bold')
        fig.legend(legends, ncol=1, bbox_to_anchor=(1.03, 0.5), loc='center left')
        
        #if not axes[-1].lines: 
        #    axes[-1].set_visible(False)

        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/VAMP2_evalDimensions_{self.ft_name}_stride{self.stride}.png', dpi=600, bbox_inches="tight")
    
        
    def VAMP2_features(self, lags, dim):
        """
        

        Parameters
        ----------
        lags : TYPE
            DESCRIPTION.
        dim : TYPE
            DESCRIPTION.

        Returns
        -------
        vamps : TYPE
            DESCRIPTION.

        """
        
        vamps={}
        rows, columns=tools_plots.plot_layout(self.features)
        fig, axes = plt.subplots(rows, columns, figsize=(8,6), constrained_layout=True, sharex=True, sharey=True)
        try:
            flat=axes.flat
        except AttributeError:
            flat=axes
            flat=[flat]

        legends = []
        colors= pl.cm.Set2(np.linspace(0,1,len(self.features)))
        
        for feature, ax, color in zip(self.features, flat, colors):
            print('Feature: ', feature)
            data = self.load_features(self.tops_to_load, 
                                      self.trajectories_to_load, 
                                      [feature], 
                                      inputs=self.selections)[0] #[0] because its yielding a list.
            
            legends.append(feature)
            scores=[]
            errors=[]
            
            lags_scalar=[]
            for lag in lags:
                try:
                    vamps[feature], ft_scores=self.score_cv(data, lag=lag, dim=dim)
                    scores.append(ft_scores.mean())
                    errors.append(ft_scores.std())
                except ValueError:
                    print('\tWarning! Failed due to too large lag, stride combination.')
                lags_scalar.append(int(str(self.timestep*lag).split(' ')[0]))
                
            errors_=np.asarray(errors)
            ax.plot(lags_scalar, scores, color=color, marker='o')
            ax.fill_between(lags_scalar, scores-errors_, scores+errors_, color=color, alpha=0.5)
            #colors=[f'C{c}' for c in range(len(self.features[ft_name]))]
                
            #ax.bar(labels, scores, yerr=errors, color='green') #, color=colors)
            ax.set_xlabel(f'Lag time (in {self.unit})')
            ax.set_ylabel('VAMP2 score')
            ax.set_xscale('log')
            ax.set_title(feature)
            #ax.tick_params('x', labelrotation=45)
            #vamp_bars_plot = dict(labels=self.labels, scores=scores, errors=errors, dim=dim, lag=lag)
        
        #fig.legend(legends) #, ncol=1, bbox_to_anchor=(1.03, 0.5), loc='center left')
        fig.suptitle(f'VAMP2 of features for {self.ft_name} (stride {self.stride*self.timestep})', weight='bold')
        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/VAMP2_evalFeatures_{self.ft_name}_stride{self.stride}.png', dpi=600, bbox_inches="tight")
        
        return vamps
        

    def load_features(self, 
                      topology, 
                      trajectories, 
                      features,
                      inputs=None,
                      get_dims=False):
        """
        

        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        trajectories : TYPE
            DESCRIPTION.
        features : TYPE
            DESCRIPTION.
        inputs : TYPE, optional
            DESCRIPTION. The default is None.
        name : TYPE, optional
            DESCRIPTION. The default is None.
        subset : TYPE, optional
            DESCRIPTION. The default is 'not water'.
        superpose : TYPE, optional
            DESCRIPTION. The default is 'protein and backbone'.
        get_dims : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        features_list : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.show_progress_bars = False
        
        features_list=[]
        superpose_indices=md.load(topology).topology.select(MSM.superpose)
        subset_indices=md.load(topology).topology.select(MSM.subset)
        md_top=md.load(topology) #, atom_indices=ref_atom_indices )
        if self.pre_process:
            trajectories=Trajectory.Trajectory.pre_process_MSM(trajectories, md_top, superpose_to=superpose_indices)
            md_top.atom_slice(subset_indices, inplace=True)
        for f in features:
            
            try:
                feat=pyemma.coordinates.featurizer(md_top)
                if type(inputs) is tuple:
                    if f == 'distances':
                        print(inputs[0], inputs[1])
                        print(feat.select(inputs[1]))
                        feat.add_distances(indices=feat.select(inputs[0]), indices2=feat.select(inputs[1]))
                    elif f == 'contacts':
                        feat.add_contacts(indices=feat.select(inputs[0]), indices2=feat.select(inputs[1]))
                    elif f == 'min_dist':
                        sel1 = feat.select(inputs[0])
                        sel2 = feat.select(inputs[1])
                        
                        for s1 in sel1:
                            for s2 in sel2:
                                #pairs_sel=np.concatenate((sel1, sel2))
                                res_pairs=feat.pairs([s1,s2], excluded_neighbors=2)
                                feat.add_residue_mindist(residue_pairs=res_pairs, 
                                                 scheme='closest-heavy', 
                                                 ignore_nonprotein=False, 
                                                 threshold=0.3)
                else:
                    
                    if f == 'torsions':
                        feat.add_backbone_torsions(selstr=inputs, cossin=True)       
                    elif f == 'positions':
                        feat.add_selection(feat.select(inputs))
                    elif f == 'chi':
                        for idx in inputs:
                            try:
                                feat.add_chi1_torsions(selstr=f'resid {idx}', cossin=True)
                            except ValueError:
                                pass
# =============================================================================
#                     elif f == 'RMSD':
#                         feat.add_minrmsd_to_ref(inputs)
# =============================================================================
                if len(feat.describe()) > 0:
                    #TODO: make this more fancy and robust
                    print(f'\tFrom: {feat.describe()[0]}\n\tTo: {feat.describe()[-1]}')
                    
                    dimensions=feat.dimension()
                    feature_file = f'{self.stored}/feature_{self.ft_name}_stride{self.stride}_dim{dimensions}.npy'
                    if not os.path.exists(feature_file):
                        data = pyemma.coordinates.load(trajectories, features=feat, stride=self.stride)
                        print(f'Loading {f} data with stride {self.stride} ({feat.dimension()} dimensions)')
                        ft_list=open(feature_file, 'wb')
                        pickle.dump(data, ft_list)
                    else:
                        print(f"Loading featurized {f} file: {feature_file}")
                        ft_list=open(feature_file, 'rb')
                        data=pickle.load(ft_list)                                 
                    if not get_dims:
                        features_list.append(data)           
                    else:
                        features_list.append((data, feat.dimension()))
                else:
                    print(f'Warning! Failed to featurize {f}. Check inputs and features.')
        
            except OSError:
                print('PyEMMA could not load data. Check input topologies and trajectories.')
        

        return features_list
        
        
    #From pyEMMA notebook
    def score_cv(self,
                 data, 
                 dim, 
                 lag, 
                 number_of_splits=10, 
                 validation_fraction=0.5):
        """Compute a cross-validated VAMP2 score.

        We randomly split the list of independent trajectories into
        a training and a validation set, compute the VAMP2 score,
        and repeat this process several times.

        Parameters
        ----------
        data : list of numpy.ndarrays
            The input data.
        dim : int
            Number of processes to score; equivalent to the dimension
            after projecting the data with VAMP2.
        lag : int
            Lag time for the VAMP2 scoring.
        number_of_splits : int, optional, default=10
            How often do we repeat the splitting and score calculation.
        validation_fraction : int, optional, default=0.5
            Fraction of trajectories which should go into the validation
            set during a split.
        """
        #print(f'\tCalculating VAMP2 scores for {dim} dimensions \n')
        nval = int(len(data) * validation_fraction)
        scores = np.zeros(number_of_splits)
        
        
        if 0 < dim <= 1:
            string=f'({dim} kinetic var.)'
        else:
            string=f'({dim} dimensions)'
        for n in range(number_of_splits):
        
            print(f'\tCalculating VAMP2 scores @lag {lag*self.timestep} {string}: cycle {n+1}/{number_of_splits}', end='\r')
            ival = np.random.choice(len(data), size=nval, replace=False)
            vamp = pyemma.coordinates.vamp([d for i, d in enumerate(data) if i not in ival], 
                                       lag=lag, 
                                       dim=dim, 
                                       stride=self.stride, 
                                       skip=self.skip, 
                                       chunksize=self.chunksize)
        
            scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])
            
            #make pretty
            if n+1 == number_of_splits:
                print(' ')

        return vamp, scores


    def clusterKmeans_standalone(self, tica, n_clusters):

        pyemma.config.show_progress_bars = False
        cluster_name=f'{self.stored}/Cluster_{self.ft_base_name}_{n_clusters}clusters.npy'
        if not os.path.exists(cluster_name) or self.overwrite:
            print(f'\tClustering for {n_clusters} cluster centres')
            clusters=pyemma.coordinates.cluster_kmeans(tica, 
                                                       max_iter=50, 
                                                       k=n_clusters, 
                                                       stride=MSM.c_stride, 
                                                       chunksize=self.chunksize)
            clusters.save(cluster_name, save_streaming_chain=True)
        else:
            print(f'\tCluster found for {n_clusters} cluster centers')
            clusters=pyemma.load(cluster_name)
        
        return clusters

    #from pyEMMA notebook
    def cluster_calculation(self, 
                            lags=[],
                            TICA_lag=None,
                            n_centres=[2, 5, 10, 30, 75, 200, 450], 
                            n_iter=5, 
                            method='kmeans',
                            overwrite=True):
        """
        

        Parameters
        ----------
        features : TYPE
            DESCRIPTION.
        lag : TYPE, optional
            DESCRIPTION. The default is [1,2].
        n_centres : TYPE, optional
            DESCRIPTION. The default is [1, 5, 10, 30, 75, 200, 450].
        n_iter : TYPE, optional
            DESCRIPTION. The default is 7.
        method : TYPE, optional
            DESCRIPTION. The default is 'kmeans'.

        Returns
        -------
        optk_features : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.show_progress_bars = False
            
        optk_features={}      
        
        rows, columns=tools_plots.plot_layout(lags)
        fig, axes=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        try:
            subplots=axes.flat
        except AttributeError:
            subplots=axes
            subplots=[subplots]
        
        for lag, ax in zip(lags, subplots):
            name = f'{self.feature}@{lag*self.timestep}'
            print(name)
            
            data = self.discretized_data
            #data.n_chunks(self.chunksize)
            tica=data.get_output()
            tica_concatenated = np.concatenate(tica)
             
            scores = np.zeros((len(n_centres),n_iter))
            
            for n, k in enumerate(n_centres):                     
                rows_, columns_ =tools_plots.plot_layout(n_iter)
                fig_, axes_ = plt.subplots(rows_, columns_, sharex=True, sharey=True, constrained_layout=True, figsize=(8, 6))                    
                try:
                    subplots_=axes_.flat
                except AttributeError:
                    subplots_=axes_
                    subplots_=[subplots_]
                
                for m, subplot in zip(range(n_iter), subplots_):
                    print(f'\tCalculating VAMP2 score for {k} cluster centres ({m+1}/{n_iter})', end='\r')
                    if method == 'kmeans':
                        clusters=pyemma.coordinates.cluster_kmeans(tica, max_iter=50, k=k, stride=MSM.c_stride, chunksize=self.chunksize)
                    #TODO: more methods
                    
                    try:
                        msm=pyemma.msm.estimate_markov_model(clusters.dtrajs, lag=lag, dt_traj=str(self.timestep))
                        scores[n,m]=msm.score_cv(clusters.dtrajs, n=1, score_method='VAMP2', score_k=min(10, k))
                    except Exception:
                        print(f'\tWarning! could not estimate MSM at iteration {m}/{n_iter-1}')
                        pass
                    
                    pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=subplot, cbar=False, alpha=0.3)
                    subplot.scatter(*clusters.clustercenters[:, :2].T, s=5, c='C1')
                    #subplot.set_title(f'Iteration {m+1}/{n_iter}')
                
                if len(subplots_) > n_iter:
                    axes_.flat[-1].axis("off")

                fig_.subplots_adjust(wspace=0, hspace=0)
                fig_.suptitle(f'{self.ft_base_name} : {k} centres')
                fig_.text(0.5, -0.02, 'IC 1', ha='center', va='center')
                fig_.text(-0.02, 0.5, 'IC 2', ha='center', va='center', rotation='vertical')
                fig_.tight_layout()                        

                fig_.savefig(f'{self.results}/ClusterTICA_{self.ft_base_name}_{method}_Elag{self.timestep*lag}_{k}centres.png', 
                             dpi=600, 
                             bbox_inches="tight")
            
            lower, upper=pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
            ax.plot(n_centres, np.mean(scores, axis=1), '-o', label=self.feature)
            ax.fill_between(n_centres, lower, upper, alpha=0.3)
            ax.semilogx()
            ax.set_xlabel('Number of cluster centres')
            ax.set_ylabel('VAMP2 score')

            optk=dict(zip(n_centres, np.mean(scores, axis=1)))
            #print(f'\tVAMP2 scores for feature {feature}', optk)
            optk_features[f'{self.feature}-{lag}']=optk
            print('\n')
        ax.legend()
        ax.set_title(r'$\tau$ = '+f'{self.timestep*lag}')
        
        fig.suptitle(f'Clustering (TICA) for {self.ft_base_name}\nMethod: {method}', weight='bold', ha='center')
        fig.tight_layout()
        fig.savefig(f'{self.results}/ClusterTICA_{self.ft_base_name}_Elag{self.timestep*lag}.png', dpi=600, bbox_inches="tight")

                
        plt.show()        
        
        self.clusters=optk_features
        
        return optk_features


    @staticmethod
    def check_inputs(input_):
        
        if isinstance(input_, int):
            input_ =[input_]
        elif isinstance(input_, tuple) and len(input_) == 2:
            input_ =range(input_[0], input_[1]+1)
        elif isinstance(input_, list):
            pass
        else:
            raise TypeError(input_+'should be either a single integer, a tuple (min, max) or a list of integers')
        
        return input_

    @staticmethod
    def mfpt_filter(mfpt, error=0):
        """Function to filter out MFPT values whose standard deviations are above *error* value.
            Default value of *error* is 0"""
            
        #mfpt=mfpt_df.loc[(scheme, feature), (parameter)]
        mfpt.dropna(how='all', inplace=True)
        mfpt.dropna(how='all', axis=1, inplace=True)
            
        idx = pd.IndexSlice    
        means=mfpt.loc[:, idx[:, 'mean']]
        stdevs=mfpt.loc[:, idx[:, 'stdev']]
        
        ratios = pd.DataFrame()
        if error != 0:
            print('Before: ', means.isna().sum().sum())
            counts=0
            ####Filter the mean values to the error percentage of stdev
            for mean, stdev in zip(means, stdevs):
                ratio = mfpt[stdev] / mfpt[mean]
                #print(mfpt[mean], mfpt[stdev], ratio)
                for r, v in zip(ratio, means[mean]):
                    if r >= error:
                        index_means=means.index[means[mean] == v]
                        means.loc[index_means, mean] =np.nan
                        counts+=1
                            
            #means.dropna(how='all', inplace=True)
            #means.dropna(how='all', axis=1, inplace=True)
                

            
        #minimum_notzero, maximum= means[means.gt(0)].min().min(), means.max().max()
        
        return means


    @staticmethod
    def convert_resid_residue(selection, init=0):
        
        selection_ = selection.split()
        resid_range = [int(word) for word in selection.split() if word.isdigit()]
        selection_new = f'residue {resid_range[0]+init} to {resid_range[-1]+init} {" ".join(selection_[-2:])}'
        
        return selection_new

