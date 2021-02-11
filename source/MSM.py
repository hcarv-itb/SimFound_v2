# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:52:02 2021

@author: hcarv
"""

class MSM:
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
    