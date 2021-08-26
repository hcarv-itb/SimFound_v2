# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:52:02 2021

@author: hcarv
"""

import os
import pyemma
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import Trajectory
import tools_plots
import tools

import mdtraj as md


class MSM:
    """
    Base class to create a pyEMMA *features* object.
    
    
    """
    
    skip=0
    
    
    def __init__(self, 
                 systems, 
                 timestep, 
                 results, 
                 chunksize=None, 
                 stride=1,
                 skip=0,
                 confirmation=False,
                 _new=False,
                 warnings=False,
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
        
        self.systems=systems
        self.results=os.path.abspath(f'{results}/MSM')
        self.stored=os.path.abspath(f'{self.results}/MSM_storage')
        tools.Functions.fileHandler([self.results, self.stored], confirmation=confirmation, _new=_new)
        self.features={}
        self.chunksize=chunksize
        self.stride=stride
        self.skip=skip
        self.timestep=timestep
        self.unit='ps'
        self.warnings=warnings
        self.heavy_and_fast=heavy_and_fast
        
        print('Results will be stored under: ', self.results)
        print('PyEMMA calculations will be stored under: ', self.stored)
 
        
    def calculate(self,
                       inputs,
                       method=None,
                       evaluate=None, 
                       ft_name=None, 
                       features=['torsions', 'positions', 'distances'], 
                       lags=[1], 
                       dim=-1,
                       equilibration=False,
                       production=True,
                       def_top=None,
                       def_traj=None):
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
        
        #TODODTOTOTODOTOTODOTODODODO
        # Define system definitions and measurement hyperparameters
        #Note: system cannot be sent as pickle for multiproc, has to be list
        systems_specs=[]
        
        
        for name, system in self.systems.items():
            
            results_folder=system.results_folder
            timestep=system.timestep
            
            topology=Trajectory.Trajectory.fileFilter(name, 
                                                        system.topology, 
                                                        equilibration, 
                                                        production, 
                                                        def_file=def_top,
                                                        warnings=self.warnings, 
                                                        filterW=self.heavy_and_fast)

            trajectory=Trajectory.Trajectory.fileFilter(name, 
                                                        system.trajectory, 
                                                        equilibration, 
                                                        production,
                                                        def_file=def_traj,
                                                        warnings=self.warnings, 
                                                        filterW=self.heavy_and_fast)
            
            
            

            systems_specs.append((trajectory, topology, results_folder, name))            
        

        #Pack fixed measurement hyperparameters
        #measurements=(inputs, start, stop, timestep, stride, self.chuncksize, self.stride)
                
        
        #Gather all trajectories and handover to pyEMMA
        trajectories_to_load = []
        tops_to_load = []
        
        for system in systems_specs: #trajectory, topology, results_folder, name 
            
            #TODO: Fix this for differente "separator" value. This is now hardcorded to '-'.
            #TODO: de-hardcode for SAM
            indexes=[n for n in system[3].split('-')]
            
            if indexes[1] == 'SAM':
                
                
                trj_list, top=system[0], system[1]
                if len(trj_list):
                    trajectories_to_load.append(trj_list[0])
                    tops_to_load.append(top)
        
        features_=self.load_features(tops_to_load[0], trajectories_to_load, features, inputs=inputs)
           
        
        if method == 'VAMP':
            if evaluate == 'features':
                vamps=self.VAMP2_features(features_, lags, dim, ft_name)
            elif evaluate == 'dimensions': 
                vamps = self.VAMP_dimensions(features_, lags, ft_name)   
            else:
                print('No evaluation defined.')
            
            self.features[f'{method}_{evaluate}'] = vamps
   
        if method == 'TICA':
            
            out_data=self.TICA_calculation(features_, lags, ft_name=ft_name, dim=dim)
            self.features[f'{method}'] = out_data
        
        else:
            print('No method defined')
            out_data=None
        
         
        
    def TICA_calculation(self, 
                         features, 
                         lag,
                         ft_name=None,
                         dim=-1, 
                         overwrite=False):
        
        ticas = {}
        
        if type(dim) is float:  
            dim = -1

        for data_tuple in features:
            (data, name, dimensions)=data_tuple
            
            file_name = f'{self.stored}/TICA_{ft_name}_{name}_lag{lag}_dim{dimensions}.npy'
            
            if not os.path.exists(file_name) or overwrite:
                print(f'\tCalculating TICA of {name} with lag {lag*self.timestep} ({dimensions}).')
                
                try:
                    tica = pyemma.coordinates.tica(data, 
                                                   lag=lag, 
                                                   dim=dim, 
                                                   skip=self.skip, 
                                                   stride=self.stride, 
                                                   chunksize=self.chunksize)
                    tica.save(file_name)
                    
                    self.plot_TICA(tica, lag, dim, ft_name, name) #TODO: Consider pushing back var_cutoff and opt_dim up.
                    
                except ValueError:
                    print(f'Failed for {name}. Probably trajectories are too short for selected lag time.')
                    break

            else:
                print(f'\tFound TICA of {name} with lag {lag*self.timestep} ({dimensions}).')
                tica = pyemma.load(file_name)
            
            ticas[f'{name}-{lag}'] = tica        

        self.tica=ticas
        return ticas

    def plot_TICA(self, 
                  tica,
                  lag,
                  dim, 
                  ft_name, 
                  name, 
                  var_cutoff=0.95, 
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
        ft_name : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        var_cutoff : TYPE, optional
            DESCRIPTION. The default is 0.95.
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
        
        tica_concat = np.concatenate(tica.get_output())
        
        if tica.dimension() > 10 and dim == -1:
            print(f'\tWarning: TICA for {var_cutoff*100}% variance cutoff yields {tica.dimension()} dimensions. \n\tReducing to {opt_dim} dimensions.')
            dims_plot=opt_dim
        elif tica.dimension() > dim and dim > 0:
            dims_plot=tica.dimension()
                
            
        fig=plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(nrows=1, ncols=2)
            
        #plot histogram
        ax0 = fig.add_subplot(gs[0, 0])
        pyemma.plots.plot_feature_histograms(tica_concat[:, :dims_plot], ax=ax0, ylog=True)
        ax0.set_title('Histogram')
            
        #plot projection along main components
        ax1 = fig.add_subplot(gs[0, 1])
        pyemma.plots.plot_density(*tica_concat[:, :2].T, ax=ax1, logscale=True)
        ax1.set_xlabel('IC 1')
        ax1.set_ylabel('IC 2')
        ax1.set_title('IC density')

        fig.suptitle(fr'TICA: {ft_name} {name} @ $\tau$ ={self.timestep*lag}', weight='bold')
        fig.tight_layout()
            
        fig.savefig(f'{self.results}/TICA_{ft_name}_{name}_lag{self.timestep*lag}.png', dpi=600, bbox_inches="tight")
        plt.show()


        #plot discretized trajectories
        ticas=tica.get_output()
        n_trajs=len(ticas)
        
        rows, columns, fix_layout=tools_plots.plot_layout(ticas)
        fig_trajs,axes_=plt.subplots(columns, rows*2, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))

        try:
            flat=axes_.flat
        except AttributeError:
            flat=axes_
            
            flat=[flat]
        

        #loop for the trajectories
        for (idx_t, trj_tica), ax in zip(enumerate(ticas), flat): #len trajs?
            
            
            x = self.timestep * np.arange(trj_tica.shape[0])
    
            for idx, tic in enumerate(trj_tica.T):
                
                if idx <= dims_plot:
                    ax.plot(x, tic, label=f'IC {idx}')
                    ax.set_ylabel('Feature values')
        
                    ax.set_xlabel(f'Total simulation time  (in {self.unit})')
                    #ax_trj.set_title('Discretized trajectory: ', idx)
        
        #fig_trajs.legend()
        fig_trajs.suptitle(fr'TICA: {ft_name} {name} @ $\tau$ ={self.timestep*lag}', weight='bold')
        fig_trajs.tight_layout()
            
        fig_trajs.savefig(f'{self.results}/TICA_{ft_name}_{name}_lag{self.timestep*lag}_discretized.png', dpi=600, bbox_inches="tight")
        plt.show()
        
        
    
    def VAMP_dimensions(self, 
                        features, 
                        lags, 
                        ft_name):
        
        rows, columns, fix_layout=tools_plots.plot_layout(features)
        fig, axes = plt.subplots(rows, columns, figsize=(9,6))

        try:
            flat=axes.flat
        except AttributeError:
            flat=axes
            flat=[flat]

        
        for ax, data_tuple in zip (flat, features):
            
            (data, name, dimensions)=data_tuple #as defined in load features
            
            print('Feature: ', name)
                        
            dims = [int(dimensions*i) for i in np.arange(0.1, 1.1, 0.1)]
            legends = []

            for lag in lags:
                print(f'\tLag: {lag*self.timestep}\n' )
                scores_dim= []
                #scores_=np.array([self.score_cv(data, dim, lag)[1] for dim in dims])

                for dim in dims:
                    try:
                        scores_dim.append(self.score_cv(data, dim, lag)[1]) #discard vamps [0]
                    except MemoryError:
                        scores_dim.append([np.nan()])
                scores_ = np.asarray(scores_dim)
                scores = np.mean(scores_, axis=1)
                errors = np.std(scores_, axis=1, ddof=1)
                #color = 'C{lags.index(lag)}'
                ax.fill_between(dims, scores - errors, scores + errors, alpha=0.3) #, facecolor=color)
                ax.plot(dims, scores, '--o')
                legends.append(r'$\tau$ = '+f'{self.timestep*lag}')
            #ax.legend()
            ax.set_title(name)
            ax.set_xlabel('number of dimensions')
            ax.set_ylabel('VAMP2 score')
        
        if ft_name != None:
            fig.suptitle(f'VAMP2: {ft_name}', weight='bold')

        fig.legend(legends, ncol=1, bbox_to_anchor=(1.08, 0.5), loc='center left')
        
        #if not axes[-1].lines: 
        #    axes[-1].set_visible(False)

        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/VAMP2_evalDimensions_{ft_name}.png', dpi=600, bbox_inches="tight")
    
        
    def VAMP2_features(self, features, lags, dim, ft_name):
        
        vamps={}
        rows, columns, fix_layout=tools_plots.plot_layout(lags)
        fig, axes = plt.subplots(rows, columns, figsize=(9,6), sharex=True, sharey=True)
        for ax, lag in zip(axes.flat, lags):
            print(f'lag time: {lag} \n')
            scores=[]
            errors=[]
            labels=[]
            
            for data_tuple in features:
                (data, name, dimensions)=data_tuple #as defined in load features
                labels.append(name)
                
                print(f'Feature: {name}')
                vamps[name], ft_scores=self.score_cv(data, lag=lag, dim=dim)
                scores += [ft_scores.mean()]
                errors += [ft_scores.std()]
            #colors=[f'C{c}' for c in range(len(self.features[ft_name]))]
                
            ax.bar(labels, scores, yerr=errors) #, color=colors)
            ax.set_title(r'$\tau$ = '+f'{self.timestep*lag}')
            ax.set_ylabel('VAMP2 score')
            ax.tick_params('x', labelrotation=45)
            #vamp_bars_plot = dict(labels=self.labels, scores=scores, errors=errors, dim=dim, lag=lag)
            
        if ft_name != None:
            fig.suptitle(f'VAMP2 : {ft_name}', weight='bold')
                
        if not axes.flat[-1].lines: 
            axes.flat[-1].set_visible(False)
            
        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.results}/VAMP2_evalFeatures_{ft_name}.png', dpi=600, bbox_inches="tight")
        
        return vamps
        

        
                    
    @staticmethod
    def load_features(topology, 
                      trajectories, 
                      features, 
                      subset='not water', 
                      inputs=None):
        """
        

        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        trajectories : TYPE
            DESCRIPTION.
        features : TYPE
            DESCRIPTION.
        subset : TYPE, optional
            DESCRIPTION. The default is 'not water'.
        inputs : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        features_list : TYPE
            DESCRIPTION.

        """
        
        pyemma.config.show_progress_bars = False
        
        features_list=[]
        
        
        ref_atom_indices=md.load(topology).topology.select(subset)
        md_top=md.load(topology, atom_indices=ref_atom_indices)
        
        for f in features:
            try:
                feat=pyemma.coordinates.featurizer(md_top)
                print('\tWarning: subset loaded')
            except:
                feat=pyemma.coordinates.featurizer(topology)
                print('default')
            print('Extracting feature: ', f)
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
                elif f == 'RMSD':
                    feat.add_minrmsd_to_ref(inputs)
            
            if len(feat.describe()) > 0:               
                #print(feat.describe())
                print(f'\tLoading data for {feat.dimension()} dimensions')
                data = pyemma.coordinates.load(trajectories, features=feat)
                features_list.append((data, f, feat.dimension()))


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
        print(f'\tCalculating VAMP2 scores for {dim} dimensions \n')
        nval = int(len(data) * validation_fraction)
        scores = np.zeros(number_of_splits)
        for n in range(number_of_splits):
            try:
                print(f'\tcycle {n+1}/{number_of_splits}', end='\r')
                ival = np.random.choice(len(data), size=nval, replace=False)
                vamp = pyemma.coordinates.vamp([d for i, d in enumerate(data) if i not in ival], 
                                           lag=lag, 
                                           dim=dim, 
                                           stride=self.stride, 
                                           skip=self.skip, 
                                           chunksize=self.chunksize)
            
                scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])
                #frames[n] = vamp.n_frames_total(self.stride, self.skip)
            except 'ZeroRankError':
                pass
        
        return vamp, scores




def cluster_calculation(self, 
                        features,
                        lag=[1,2],
                        n_centres=[1, 5, 10, 30, 75, 200, 450], 
                        n_iter=7, 
                        method='kmeans'):
    
    try:
        tica = self.tica
        print('TICA loaded: ', tica)
    except:
        print('TICA not found. Calculating...')
        tica=self.TICA_calculation(features, 
                         lag,
                         ft_name=None,
                         dim=-1, 
                         overwrite=False)
        
    optk_features={}
    

    
    
    for feature in features:
        
        rows, columns, fix_layout=tools_plots.plot_layout(lag)
        fig, axes=plt.subplots(rows, columns, sharex=True, sharey=True, constrained_layout=True, figsize=(9,6))
        
        try:
            flat=axes.flat
        except AttributeError:
            flat=axes
            flat=[flat]
        
        for l, ax in zip(lag, flat):
        
            try:
                tica=tica[f'{feature}-{l}'].get_output()
    
            except KeyError:
                print(f'No {feature}-{l} combination found in stored TICA')    
                break
            
            if method == 'kmeans':
                scores = np.zeros((len(n_centres),1))
                for n, t in enumerate(n_centres):
                    print(f'Calculating score for {t} cluster centres.')
                    for m in range(n_iter):
                        _cl=pyemma.coordinates.cluster_kmeans(tica,
                                                              max_iter=200,
                                                              k=t)
                        
                        _msm=pyemma.msm.estimate_markov_model(_cl.dtrajs,
                                                              lag=l,
                                                              dt_traj=self.timestep,
                                                              maxerr=1e-08)
                        scores[n,m]=_msm.score_cv(_cl.dtrajs,
                                                  n=1,
                                                  score_method='VAMP2')
                        
                        print('Scores: ',scores)
                        
                fig, ax=plt.subplots()
                lower, upper=pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
                ax.fill_between(n_centres, lower, upper, alpha=0.3)
                ax.semilogx()
                ax.set_xlabel('Number of cluster centres')
                ax.set_ylabel('VAMP2 score')
                ax.plot(n_centres, np.mean(scores, axis=1), '-o')
                
                
                optk=dict(zip(n_centres, np.mean(scores, axis=1)))
                print(f'Cluster centres for feature {feature}', optk)
    
                
                optk_features[feature]=optk
            else:
                print('no method defined')
        
    self.clusters=optk_features
    
    return optk_features
        
        
        
        
        
        
        
        
        
        
        
# =============================================================================
#         #Spawn multiprocessing tasks    
#         data_t=tools.Tasks.parallel_task(method_function, systems_specs, measurements, n_cores=n_cores)
#         
#         # concat dataframes from multiprocessing into one dataframe
#         #Update state of system and features with the feature df.
#         out_df=pd.DataFrame()
#         
#         for data_df, system in zip(data_t, self.systems.values()):
#             system.data[f'{method}-{feature_name}']=data_df
#             system.features[f'{method}-{feature_name}']=inputs
#             out_df=pd.concat([out_df,  data_df], axis=1)
#         out_df.index.rename=data_df.index.name
#         out_df.to_csv(os.path.abspath(f'{self.results}/{method}_{feature_name}_{str(inputs)}.csv'))
# 
#         if feature_name != None:
#             self.features[f'{feature_name}']=out_df
#         else:
#             self.features['undefined']=out_df
#             
#         
#         out_df.name=method
#         
#         print('Featurization updated: ', [feature for feature in self.features.keys()])
#         
#         return out_df
# =============================================================================






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
        


      
    def bayesMSM(self, lag, variant=False, statdist_model=None):
        """Function to calculate model based on provided lag time. Must be evaluated suitable lag time based on ITS.
        If no suitable lag is found, model calculation will be skipped.
        Accepts variants. Currently, only *norm* is defined."""
        
        
        #TODO: Get lag directly from passed lagged object or dictionary.
        lag_model=lag


        if lag_model == None:
            print('No suitable lag time found. Skipping model')
            bayesMSM=None       
        else:
            bayesMSM_name=f'{system.results}/bayesMSM_{system.feature_name}-lag{lag_model}.npy'
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
        
        #Set up

        #TODO: iterate through features.
        discretized_name, discretized_data=[i for i in self.feature]
        
        
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
    