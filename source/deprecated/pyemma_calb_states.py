#!/bin/python3
import os
os.system("source ../miniconda3/bin/activate pyemma")


import pyemma
from pyemma import config


import argparse
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.show(block=True)
plt.ion()
import glob
import pickle
import pandas as pd
from functions2 import *

parser = argparse.ArgumentParser(description='Script to featurize and analyze MD simulation data using PyEmma.',epilog='End')
parser.add_argument('-ft', metavar='Feature',type=str,nargs='+', help='To be analyzed.')
parser.add_argument('--cc', metavar='cluster centres or dmin',default=100,type=int,help='Number of clusters for k-mean clustering or minimal distance between clusters for regspace')
parser.add_argument('--lg', metavar='Lagtime',default=100, type=int, help='Lag time for TICA and VAMP.')
parser.add_argument('--mlg', metavar='msm_lagtime',default=500, type = int, help='Lag time for creating the MSM, minimal lag time with converging implied timescales (select with its plot)')
parser.add_argument('--n', metavar='nstates',default=4, type=int, help='Number of states for CK-Test and rates')
parser.add_argument('--met', metavar='Methode', default='hmm', type=str, help='Methode to use for building the model.')
parser.add_argument('--clu', metavar='Clustering', default='kmeans', type=str, help='Clustering method.')
parser.add_argument('--comb', action='store_true', help='Should the features be combined into one MSM?')
parser.add_argument('--c_plt', action='store_true', help='Should a plot to select the number of clusters be generated? (Takes long for kmeans)')
parser.add_argument('--va', metavar='Variation Approach',default='tica', type=str, help='Variational approach for the clustering to be based on. If no transformation should be performed before clustering, specify <raw>.')

sels=[['name 9C81', 'name OA'], ['resid 224 and name NE2', 'name HO']]

args=parser.parse_args()
cc=args.cc
feats=args.ft
lag=args.lg
msm_lag=args.mlg
nstates=args.n
comb=args.comb
met=args.met
clu=args.clu
c_plt=args.c_plt
va=args.va

#Go to the directory, where the simulation data can be found
wd="/home/esc/calb-meoh-1M_test2/"
os.chdir(wd)

#Defining the xtc and pdb files to be used
xtc=sorted(glob.glob("calb-100ns-*.xtc"))
pdb="calb-100ns.pdb"

print('xtc=',xtc)
print('pdb=',pdb)

sims=len(xtc)
v_f=0.8#Choose the validation fraction here
lags=[1,10,50,100,450]#Lags for implied timescales
lags_vd=[1,2,5,10,25,50,57,100,150,200,250]#Lags for selecting TICA/VAMP lag and dimensionality cutoff
n_clustercentres = [2,5,10,30,50,75,100,120]#Number of cluster centres to test for finding the optimum in case of kmeans clustering
reg_dmin=[1,10,15,20,22,25]#Minimal distance of cluster centres to tet for finding the optimum in case of regspace clustering
nsplits=10#Choose the number of splits here
labels=[]
conc='1M'#Define the methanol concentration in the simulation here
ps=5#Define the timestep here
if nstates == 3:
	pos_array=np.asarray([[2, 4], [4, 0], [6,4]])
elif nstates==2:
	pos_array=np.asarray([[2,4], [4,0]])
elif nstates == 4:
	pos_array=np.asarray([[0,0],[2, 4], [4, 0], [6,4]])
elif nstates == 5:
	pos_array=np.asarray([[0,0],[2, 4], [4, 0], [6,4], [1,3]])
elif nstates == 6:
	pos_array=np.asarray([[0,0],[2, 4], [4, 0], [6,4],[1,3],[1,5]])
else:
	print('Please define an array for the positions with the lenght of nstates here. Otherwise an error will occur lateron.')
#Define the dimensionality cutoff for the variational approach
for feat in feats:
	if feat=='tors':
		dim=0.9
		v_c=0.9
	if feat == 'pos':
		dim=0.8
		v_c=0.8
	if feat == 'sea_met':
		dim=0.9
		v_c=0.9
	if feat=='helix_dist':
		dim=0.999
		v_c=0.999
	if feat =='helix_dist_2':
		dim=0.9
		v_c=0.9
dim=0.95
v_c=0.95
print('Dimensionality cutoff:',dim)

#Featurization
d={}
if comb == False:
	for feat in feats:
		feature_nc='{}_data_sim{}.npy'.format(feat,sims)
		labels+=['{}'.format(feat)]
		if os.path.exists(feature_nc):
			print("Featurized {} detected, loading...".format(feature_nc))
			ft_list=open(feature_nc, 'rb')
			d[feat]=pickle.load(ft_list)
		else:
			ft_feat=pyemma.coordinates.featurizer(pdb)
			print('Featurizing', feat)
			if feat == "tors":
				ft_feat.add_backbone_torsions(cossin=False,deg=True)
			elif feat == 'dist':
				ft_feat.add_distances(ft_feat.pairs(ft_feat.select_Backbone(),excluded_neighbors=2),periodic = True)
			elif feat == 'pos':
				ft_feat.add_selection(ft_feat.select_Backbone())
			elif feat == 'sea_met':
				sea=ft_feat.select('resname SEA and index 1526')
				met=ft_feat.select('resname MeOH and name OA')
				ft_feat.add_distances(indices=met, indices2=sea)
			elif feat == 'helix_dist':
				leu_5=ft_feat.select('resname LEU and name CA and residue 144')
				helix_10=ft_feat.select('(resname LEU or resname ILE) and name CA and (residue 277 or residue 285)')
				ft_feat.add_distances(indices=leu_5, indices2=helix_10)
			elif feat == 'helix_dist_2':
				helix_5_1=ft_feat.select('residue 148 and name CA')
				helix_5_2=ft_feat.select('resname LEU and name CA and residue 144')
				helix_10_1=ft_feat.select('resname LEU and name CA and residue 277')
				helix_10_2=ft_feat.select('resname ILE and name CA and residue 285')
				ft_feat.add_distances(indices=helix_5_1, indices2=helix_10_1)
				ft_feat.add_distances(indices=helix_5_2, indices2=helix_10_2)
			d[feat]=pyemma.coordinates.load(xtc, features = ft_feat)
			ft_list=open(feature_nc, 'wb')
			pickle.dump(d[feat],ft_list)
else:
	value=['combined features']
	for feat in value:
		feature_c='comb{}_data_sim{}.npy'.format(feats,sims)
		if os.path.exists(feature_c):
			print("Featurized {} detected, loading...".format(feature_c))
			ft_list=open(feature_c, 'rb')
			d[feat]=pickle.load(ft_list)
		else:
			ft_feat=pyemma.coordinates.featurizer(pdb)
			for feat in feats:
				print('Featurizing and adding', feat)
				if feat == 'tors':	
					ft_feat.add_backbone_torsions(cossin=True,periodic=True)
				elif feat == 'dist':
					ft_feat.add_distances(ft_feat.pairs(ft_feat.select_Backbone(),excluded_neighbors=en),periodic = True)
				elif feat == 'pos':
					ft_feat.add_selection(ft_feat.select_Backbone())
				elif feat == 'sea_met':
					sea=ft_feat.select('resname SEA and index 1526')
					met=ft_feat.select('resname MeOH and name OA')
					ft_feat.add_distances(indices=met, indices2=sea)
			d[feat]=pyemma.coordinates.load(xtc,features=ft_feat)
			ft_list=open(feature_c, 'wb')
			pickle.dump(d[feat], ft_list)

print('Shape of ft_feat:', np.shape(d[feat]))

# =============================================================================
# #Cross validated VAMP2 score to rank features
# def score_cv(data,dim,lag,number_of_splits=nsplits,validation_fraction=v_f):
# 	nval=int(len(data)*validation_fraction)
# 	scores=np.zeros(number_of_splits)
# 	for n in range(number_of_splits):
# 		ival=np.random.choice(len(data), size=nval,replace=False)
# 		vamp=pyemma.coordinates.vamp([d for i, d in enumerate(data) if i not in ival],lag=lag,dim=dim,chunksize=1000)
# 		print('Parameter:', vamp.get_params())
# 		scores=vamp.score([d for i, d in enumerate(data) if i in ival])
# 		print(scores)
# 	return scores
# 
# def cov_cv(data,lag):
# 	if va == 'tica':
# 		vamp=pyemma.coordinates.tica([d for i, d in enumerate(data)],lag=lag,chunksize=1000)
# 	elif va == 'vamp':
# 		vamp=pyemma.coordinates.vamp([d for i, d in enumerate(data)],lag=lag,chunksize=1000)
# 	print('Parameter:', vamp.get_params())
# 	cov=vamp.cumvar
# 	return cov
# 
# if not va == 'raw':
# 	#sc={}
# 	#for l in lags:
# 	#	vamp_bars='vamp_bar_plot_{}_lg{}_vf{}_sim{}_dim{}.png'.format(feats,l,v_f,sims,dim)
# 	#	if os.path.exists("vamp_bar_plots/" + vamp_bars):
# 	#		print('{} already exists, skipping calculations.'.format(vamp_bars))
# 	#	else:
# 	#		for k,v in d.items():
# 	#			feat_sc='{}_sc_lg{}_vf{}_sim{}_dim{}.npy'.format(k,l,v_f,sims,dim)
# 	#			if os.path.exists("scores/" + feat_sc):
# 	#				print('{} already exists, skipping calculations'.format(feat_sc))
# 	#				sc[k]=np.load("scores/" + feat_sc)
# 	#			else:
# 	#				print('Calculating VAMP score {}.'.format(feat_sc))
# 	#				sc[k]=score_cv(v,lag=l,dim=dim)
# 	#				np.save("scores/" + feat_sc,sc[k])
# 	#			scores=[]
# 	#			err=[]
# 	#			for k,v in sc.items():
# 	#				scores+=[v.mean()]
# 	#				err+=[v.std()]
# 	#				print('This is scores for {}:'.format(k), scores)
# 	#		plt.bar(labels,scores,yerr=err)
# 	#		plt.title('lag time tau = {} ps'.format(l*ps))
# 	#		plt.savefig("vamp_bar_plots/" + vamp_bars)
# 	#		plt.clf()
# 	
# 	#Using different dimensions to select the best (TICA/VAMP) lag time
# 	for k,v in d.items():
# 		if comb == False:
# 			vamp_var='vamp_var_plt_{}_sim{}.png'.format(k,sims)
# 		else:
# 			vamp_var='vamp_var_plt_{}_sim{}.png'.format(feats,sims)
# 		if os.path.exists("vamp_dim_plots/" + vamp_var):
# 			print('{} already exists,skipping calculations.'.format(vamp_var))
# 		else:
# 			print('Calculating vamp_var plot...')
# 			fig, ax = plt.subplots()
# 			for i, l in enumerate(lags_vd):
# 				color = 'C{}'.format(i)
# 				cov_val=cov_cv(v,lag=l)
# 				ax.plot(cov_val, '--o', color=color, label='lag={:.1f}ps'.format(l * ps))
# 			ax.set_ylim(0, 1)
# 			#ax.set_xlim(0,1.5)
# 			ax.legend()
# 			ax.set_xlabel('Dimensions')
# 			ax.set_ylabel('Cumulative Variance')
# 			fig.tight_layout()
# 			fig.savefig("vamp_dim_plots/" + vamp_var)
# 			fig.clf()
# lag=lag
# 	
# #tICA/VAMP with histogram and density
# vamp_wo={}
# vamp={}
# vamp_conc={}
# vamp_wo_red={}
# vamp_red={}
# vdim={}
# for k,v in d.items():
# 	tdim=-1
# 	if va == 'tica':
# 		print('Calculating tica object...')
# 		vamp_wo[k]=pyemma.coordinates.tica(v,lag=lag,dim=tdim, var_cutoff=v_c)
# 	elif va == 'vamp':
# 		print('Calculating vamp object...')
# 		vamp_wo[k]=pyemma.coordinates.vamp(v,lag=lag,dim=dim)
# 		print('Singular values:',vamp_wo[k].singular_values)
# 	elif va == 'pca':
# 		print('Calculating PCA object...')
# 		vamp_wo[k]=pyemma.coordinates.pca(v,variance_cutoff=v_c)
# 	elif va == 'raw':
# 		vamp[k]=v
# 	if not va == 'raw':
# 		vamp[k]=vamp_wo[k].get_output()
# 		vdim[k]=vamp_wo[k].dimension()
# 		print('Parameters:', vamp_wo[k].get_params())
# 		print('Shape (sims, traj, output dimensions):', np.shape(vamp[k]))
# 	for k,v in vamp.items():
# 		vamp_conc[k]=np.concatenate(v)
# for k,v in vamp_conc.items():
# 	if comb == False:
# 		vamp_hist='{}_hist_{}_lg{}_vc{}_sim{}_dim{}.png'.format(va,k,lag,v_c,sims,dim)
# 	else:
# 		vamp_hist='{}_hist_comb{}_lg{}_vc{}_sim{}_dim{}.png'.format(va,feats,lag,v_c,sims, dim)
# 	fig,axes=plt.subplots(1,2,figsize=(10,4))
# 	pyemma.plots.plot_feature_histograms(v[:, :15],ax=axes[0],ylog=True)
# 	pyemma.plots.plot_density(*v[:, :2].T,ax=axes[1],logscale=True)
# 	axes[1].set_xlabel('Component 1')
# 	axes[1].set_ylabel('Component 2')
# 	fig.tight_layout()
# 	fig.savefig("histograms/" + vamp_hist)
# 	plt.close()
# 
# #Plotting histogram and Ramachandran plot for Phi and Psi values of torsions
# if feat == 'tors' and va == 'raw':
# 	for i in d[feat]:
# 		for j in i:
# 			phi=j[0::2]
# 			psi=j[1::2]
# 		fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# 		axs[0].hist(phi)
# 		axs[1].hist(psi)
# 		axs[0].set_xlabel(r'$\Phi$')
# 		axs[1].set_xlabel(r'$\Psi$')
# 		fig.savefig('histograms/torsions_hist_phi_psi_{}_lg{}_vc{}_sim{}_dim{}.png'.format(va,lag,v_c,sims,dim))
# 		plt.close()
# 
# 		fig, ax = plt.subplots(figsize=(4,4))
# 		ax.hist2d(phi,psi,density=True, bins=30)
# 		ax.set_xlabel(r'$\Phi$')
# 		ax.set_ylabel(r'$\Psi$')
# 		fig.savefig('histograms/torsions_density_phi_psi_{}_lg{}_vc{}_sim{}_dim{}.png'.format(va,lag,v_c,sims,dim))
# 		plt.close()
# 
# #Plotting trajectory in the space of the first 4 TICA/VAMP components
# if not va == 'raw':
# 	for k,v in vamp.items():
# 		if comb == False:
# 			vamp_traj='trajectory_{}_{}_lg{}_vc{}_sim{}_dim{}.png'.format(k,va,lag,v_c,sims,dim)
# 		else:
# 			vamp_traj='trajectory_comb{}_{}_lg{}_vc{}_sim{}_dim{}.png'.format(feats,va,lag,v_c,sims,dim)
# 		fig,axes=plt.subplots(4,1,figsize=(12,5),sharex=True)
# 		x=ps*np.arange(v[0].shape[0])
# 		for i,(ax, tic) in enumerate(zip(axes.flat,v[0].T)):
# 			ax.plot(x,tic)
# 			ax.set_ylabel('IC {}'.format(i+1))
# 		axes[-1].set_xlabel('time / ps')
# 		fig.tight_layout()
# 		fig.savefig("trajectories/" + vamp_traj)
# 		fig.clf()
# 
# #Determining the optimal number of cluster centres for k-clustering
# if c_plt == True:
# 	for c,v in vamp.items():
# 		if comb == False:
# 			cluster_centres='cluster_centres_{}_{}_lg{}_vf{}_vc{}_sim{}_dim{}_mlg{}_{}.png'.format(c,va,lag,v_f,v_c,sims,dim,msm_lag,clu)
# 		else:
# 			cluster_centres='cluster_centres_comb{}_{}_lg{}_vf{}_vc{}_sim{}_dim{}_mlg{}_{}.png'.format(feats,va,lag,v_f,v_c,sims,dim,msm_lag,clu)
# 		print('Determining the optimal number of cluster centres for {}...'.format(c))
# 		if clu == 'kmeans':
# 			scores = np.zeros((len(n_clustercentres),1))
# 			for n, t in enumerate(n_clustercentres):
# 				print('Calculating score for {} cluster centres.'.format(t))
# 				for m in range(7):
# 					_cl=pyemma.coordinates.cluster_kmeans(v,max_iter=200,k=t)
# 					_msm=pyemma.msm.estimate_markov_model(_cl.dtrajs,lag=msm_lag,dt_traj='{} ps'.format(ps),maxerr=1e-08)
# 					scores[n,m]=_msm.score_cv(_cl.dtrajs,n=1,score_method='VAMP2')
# 					print('Scores:',scores)
# 			fig, ax=plt.subplots()
# 			lower,upper=pyemma.util.statistics.confidence_interval(scores.T.tolist(),conf=0.9)
# 			ax.fill_between(n_clustercentres,lower,upper,alpha=0.3)
# 			ax.plot(n_clustercentres,np.mean(scores,axis=1),'-o')
# 			optk=dict(zip(n_clustercentres,np.mean(scores,axis=1)))
# 			print('Cluster centres:VAMP Score (for feature {})'.format(c),optk)
# 			ax.semilogx()
# 			ax.set_xlabel('Number of cluster centres')
# 		elif clu == 'regspace':
# 			scores = np.zeros((len(reg_dmin),1))
# 			for n, t in enumerate(reg_dmin):
# 				print('Calculating score for {} cluster centres.'.format(t))
# 				for m in range(1):
# 					_cl=pyemma.coordinates.cluster_regspace(v,dmin=t,max_centers=1000)
# 					_msm=pyemma.msm.estimate_markov_model(_cl.dtrajs,lag=msm_lag,dt_traj='{} ps'.format(ps))
# 					scores[n,m]=_msm.score_cv(_cl.dtrajs,n=1,score_method='VAMP2')
# 					print('score',scores)
# 			fig, ax=plt.subplots()
# 			lower,upper=pyemma.util.statistics.confidence_interval(scores.T.tolist(),conf=0.9)
# 			ax.fill_between(reg_dmin,lower,upper,alpha=0.3)
# 			ax.plot(reg_dmin,np.mean(scores,axis=1),'-o')
# 			optk=dict(zip(reg_dmin,np.mean(scores,axis=1)))
# 			print('Cluster centres:VAMP Score (for feature {})'.format(c),optk)
# 			ax.set_xlabel('dmin (distance of clusters)')
# 			ax.semilogx()
# 		ax.set_ylabel('VAMP-2 score')
# 		fig.tight_layout()
# 		fig.savefig("cluster_centres/" + cluster_centres)
# 		fig.clf()
# 
# #Creating relevant clusters/states with the predefined number of centres
# cluster={}
# for k,v in vamp.items():
# 	if clu == 'regspace':
# 		print('Regspace clustering using dmin {}...'.format(cc))
# 		cluster[k]=pyemma.coordinates.cluster_regspace(v, dmin=cc,max_centers=1000)
# 		print('Number of clusters:',cluster[k].max_centers)
# 	elif clu == 'kmeans':
# 		print('K-means clustering using {} cluster centres...'.format(cc))
# 		cluster[k]=pyemma.coordinates.cluster_kmeans(v,k=cc, max_iter=200)
# 
# #Plotting states in the first TICA dimensions
# for k,v in vamp_conc.items():
# 	print('Plotting states in IC for {}...'.format(k))
# 	if comb == False:
# 		ic_cluster='states_in_IC_{}_{}_lg{}_mlg{}_vc{}_sim{}_dim{}_cc{}_{}.png'.format(k,va,lag,msm_lag,v_c,sims,dim,cc,clu)
# 	else:
# 		ic_cluster='states_in_IC_comb{}_{}_lg{}_mlg{}_vc{}_sim{}_dim{}_cc{}_{}.png'.format(feats,va,lag,msm_lag,v_c,sims,dim,cc,clu)
# 	fig, ax=plt.subplots(figsize=(4, 4))
# 	pyemma.plots.plot_density(*v[:, :2].T,ax=ax,cbar=False,alpha=0.3)
# 	ax.scatter(*cluster[k].clustercenters[:, :2].T,s=5,c='C1')
# 	ax.set_xlabel('Component 1')
# 	ax.set_ylabel('Component 2')
# 	fig.tight_layout()
# 	fig.savefig("states_in_IC/" + ic_cluster)
# 	fig.clf()
# 
# #Estimation of implied timescales and Markov Models
# _msm={}
# its={}
# its_hmm={}
# model={}
# ac_s={}
# for k,v in cluster.items():
# 	its_out='implied_timescale/' + 'its_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}.png'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu)
# 	print('Plotting implied timescales for msm...')
# 	its[k]=pyemma.msm.its(v.dtrajs, lags=np.array([1, 2, 5, 10, 40, 125, 200, 500,800,1000]),nits=10, errors='bayes')
# 	pyemma.plots.plot_implied_timescales(its[k],outfile=its_out,units='ps',dt=ps)
# 	plt.close()
# 	if met =='hmm' or met =='bhmm':
# 		print('Plotting implied timescales for hmm...')
# 		its_hmm_out='implied_timescale/' +'its_hmm_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}.png'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu)
# 		its_hmm[k] = pyemma.msm.timescales_hmsm(v.dtrajs, nstates,lags=np.array([1, 2, 5, 10, 40, 125, 200, 500,800, 1000]),errors='bayes',mincount_connectivity=0.0)
# 		pyemma.plots.plot_implied_timescales(its_hmm[k], outfile=its_hmm_out,units='ps',dt=ps,marker='o', ylog=True)
# 	print('Building and analyzing bayesian MSM...')
# 	bmsm='bmsm_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}.npy'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu)
# 	_msm[k]=pyemma.msm.bayesian_markov_model(v.dtrajs,lag=msm_lag,dt_traj='{} ps'.format(ps),conf=0.95,connectivity='largest',count_mode='effective')
# 	_msm[k].save('models/' + bmsm, overwrite=True)
# 	print('fraction of states used for bayesian msm = {:.2f}'.format(_msm[k].active_state_fraction))
# 	print('fraction of counts used for bayesian msm = {:.2f}'.format(_msm[k].active_count_fraction))
# 	if met == 'coarse':
# 		print('Building coarse grained bayesian MSM...')
# 		model[k]=_msm[k].coarse_grain(nstates)
# 	elif met == 'bhmm':
# 		print('Building bayesian hidden markov model...')
# 		model[k]=pyemma.msm.bayesian_hidden_markov_model(v.dtrajs,nstates,lag=msm_lag,dt_traj='{} ps'.format(ps),conf=0.95,connectivity='largest',mincount_connectivity=0.0)
# 	elif met == 'hmm':
# 		print('Estimating hidden markov model...')
# 		model[k]=pyemma.msm.estimate_hidden_markov_model(v.dtrajs,nstates,lag=msm_lag,dt_traj='{} ps'.format(ps),connectivity='largest',mincount_connectivity=0.0)
# 	elif met == 'pcca':
# 		_msm[k].pcca(nstates)
# 		model[k]=_msm[k]
# 	ac_s[k]=len(model[k].active_set) - 1
# 	print('Active Set:', model[k].active_set)
# 	print('Eigenvalues of coarse model:',model[k].eigenvalues())
# 	model[k].save('models/model_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.npy'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), overwrite=True)
# 
# #Chapman-Kolmogorov test
# cktest_model={}
# cktest_bmsm={}
# for k,v in _msm.items():
# 	ck_bmsm='ck_bmsm_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met)
# 	print('Calculating CK test for bMSM...')
# 	cktest_bmsm[k] = v.cktest(nstates,mlags=4)
# 	pyemma.plots.plot_cktest(cktest_bmsm[k], dt=ps, units='ps')
# 	plt.savefig("ck_test/" + ck_bmsm)
# 	plt.clf()
# if not met == 'pcca':
# 	for k,v in model.items():
# 		print('Calculating CK test for {}...'.format(met))
# 		ck_model='ck_model_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met)
# 		cktest_model[k] = v.cktest(mlags=4)
# 		pyemma.plots.plot_cktest(cktest_model[k], dt=ps, units='ps')
# 		plt.savefig("ck_test/" + ck_model)
# 		plt.clf()
# 
# #MFPT,Rates,Histograms
# mdtraj={}
# stat={}
# for k,v in model.items():
# 	f=open('mfpt_csv/' + 'mfpt_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.csv'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), 'wb')
# 	fs=open('mfpt_csv/' + 'mfpt_rates_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.csv'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), 'wb')
# 	pi_c=pd.DataFrame()
# 	flux_c=pd.DataFrame()
# 	flux_d=pd.DataFrame()
# 	print('Calculating stationary distributions...')
# 	if met=='pcca':
# 		statdist=np.zeros(nstates)
# 		for i,s in enumerate(v.metastable_sets):
# 			statdist[i]=v.pi[s].sum()
# 	else:
# 		statdist=v.stationary_distribution
# 	print('Stationary distribution (pi) of states: \n', statdist)
# 	pi=pd.Series(statdist, name=k)
# 	pi_c=pd.concat([pi_c, pi], axis=1)
# 	pi_c.to_csv('pi_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.csv'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met))
# 	print('Sum of weights = {:f}'.format(v.pi.sum()))
# 	print("Mean first passage times - MFPT (ps) in the {} of state indexes".format(range(ac_s[k])))
# 	mfpt = np.zeros((nstates, nstates))
# 	mfpt_std = np.zeros((nstates, nstates))
# 	for i in range(nstates):
# 		for j in range(nstates):
# 			if met == 'bhmm':
# 				mfpt[i,j] = v.sample_mean("mfpt", i,j)
# 				mfpt_std[i,j] = v.sample_std("mfpt", i,j)
# 			elif met == 'coarse' or met == 'hmm':
# 				mfpt[i,j] = v.mfpt(i,j)
# 			elif met == 'pcca':
# 				mfpt[i,j] = v.mfpt(v.metastable_sets[i],v.metastable_sets[j])
# 	mfpt_table=pd.DataFrame(np.round(mfpt, decimals=2), index=range(1, nstates + 1), columns=range(1, nstates + 1))
# 	print("Mean MFPT values: \n", mfpt_table)
# 	if met=='bhmm':
# 		mfpt_table_std=pd.DataFrame(np.round(mfpt_std, decimals=2), index=range(1, nstates + 1), columns=range(1, nstates + 1))
# 		print("Std MFPT values: \n", mfpt_table_std)
# 		np.savetxt(f, mfpt_table_std, delimiter=",")
# 	np.savetxt(f, mfpt_table, delimiter=",")
# 	rate_mfpt=(1/mfpt_table)/1e-12
# 	rate_mfpt=rate_mfpt.replace(np.inf, 0)
# 	print("Rate constants (s-1) in the {} of state indexes".format(range(ac_s[k])))
# 	print("Means: \n", rate_mfpt)
# 	if met == 'bhmm':
# 		ave_std=1/(mfpt_table-(mfpt_table-((mfpt_table - mfpt_table_std)*1e-12)))-rate_mfpt
# 		ave_std=ave_std.replace(np.inf, 0)
# 		print("Standard deviations: \n", ave_std)
# 		np.savetxt(fs, ave_std, delimiter=",")
# 	np.savetxt(fs, rate_mfpt, delimiter=",")
# 	inverse_mfpt = np.zeros_like(mfpt)
# 	nz = mfpt.nonzero()
# 	inverse_mfpt[nz] = 1.0 / mfpt[nz]
# 	pyemma.plots.plot_network(inverse_mfpt, arrow_label_format='%.1f ns', state_sizes=statdist,pos=pos_array,arrow_labels=mfpt*0.001, arrow_scale=1.5,state_labels=range(1, nstates + 1), size=12)
# 	plt.savefig('mfpt/' + 'mfpt_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met))
# 	plt.clf()
# 
# 	print('Extracting frames for each state...')
# 	met_member= v.metastable_memberships
# 	set_full=v.metastable_sets
# 	bad_clusters=[]
# 	sets=[]
# 	if met == 'pcca':
# 		traj=v.discrete_trajectories_active
# 	elif met == 'hmm' or met =='bhmm':
# 		traj=v.discrete_trajectories_obs
# 	probability=0.75
# 	#Only use microclusters with at least this probability to be assigned to one of the macro clusters.
# 	for o,p in enumerate(met_member):
# 		if max(p)<probability:
# 			bad_clusters.append(o)
# 	print('Bad clustres:',bad_clusters)
# 	print('Number of discarded clustres:', len(bad_clusters))
# 	for r,q in enumerate(set_full):
# 		new=np.setdiff1d(q,bad_clusters)
# 		sets.append(new)
# 		print('Microstates with a probability below {}% were removed. In macrostate {} are now {} instead of {} microstate.'.format(probability*100,r,len(new),len(q)))
# 	my_samples = [[] for _ in range(nstates)]
# 	for a,b in enumerate(traj):
# 		for c,d in enumerate(b):
# 			for e in range(nstates):
# 				if d in sets[e]:
# 					my_samples[e].append([a,c])
# 	number_frames=0
# 	for number in my_samples:
# 		number_frames+=len(number)
# 	print(number_frames,' frames were extracted in total.')
# 	my_samples_2=[]
# 	for element in my_samples:
# 		print('{} frames were extracted from this state, implying a stationary distribution of {} '.format(len(element),len(element)/number_frames))
# 		my_samples_2.append(element[1])
# 
# 	my_samples_traj=[pyemma.coordinates.save_traj(xtc, idist, outfile=None, top=pdb) for idist in my_samples]
# 	my_samples_traj_2=pyemma.coordinates.save_traj(xtc, my_samples_2, outfile=None, top=pdb)
# 
# 	dG_c=pd.DataFrame()
# 	dist_c=pd.DataFrame()
# 	for x,y in enumerate(my_samples_traj_2):			
# 		print('Saving gro file...')
# 		y.save_gro('structures/' + 'samples_{}-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.gro'.format(k,x+1,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met))	
# 	for x,y in enumerate(my_samples_traj):
# 		print('saving xtc file...')
# 		y.save_xtc('structures/' + 'samples_{}-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.xtc'.format(k,x+1,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met))
# 		print('Calculating the NAC distances for each frame of state {}...'.format(x+1))
# 		dG_df=NAC_calculation(wd=wd,name='samples_{}-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}'.format(k,x+1,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met),  sels=sels, i=x+1, c=conc)
# 		dG_c=pd.concat([dG_c, dG_df], axis=1)
# 
# 		print('Calculating the helix distance for each frame of state {}...'.format(x+1))
# 		dist_df=distance_calculation(wd=wd,name='samples_{}-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}'.format(k,x+1,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), i=x+1)
# 		dist_c=pd.concat([dist_c, dist_df], axis=1)
# 
# 	print('Plotting NAC distribution for each state...')
# 	ranges_bulk=np.arange(30, 40.5, 0.5)
# 	results_dir=wd + 'structures/results/'
# 	binding=dG_c.filter(like='dG')
# 	fig=binding.plot(kind='line', logx=True, colormap='viridis')
# 	fig.set_xlabel(r'NAC ($\AA$)')
# 	fig.set_ylabel(r'$\Delta G (k_BT)$')
# 	fig.set_xlim(1,50)
# 	fig.axhline(y=0, color='black')
# 	plt.savefig('{}/dG_nac-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(results_dir,k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), dpi=600)
# 	plt.clf()
# 
# 	print('Plotting fitting of NAC for dG...')
# 	fig, ax=plt.subplots(sharey=True)
# 	ax2=ax.twinx()
# 	N_theoretic, N_adjusted, N_sim=dG_c.filter(like='{initial}'), dG_c.filter(like='{adjusted}'), dG_c.filter(like='state')
# 	N_adjusted.plot(kind='line', style="--", ax=ax,colormap='viridis')
# 	N_theoretic.plot(kind='line', style=":", ax=ax,colormap='viridis')
# 	N_sim.plot(kind='line', ax=ax2,colormap='viridis')
# 	ax.set_ylabel(r'ln $N_{theoretic}$')
# 	ax2.set_ylabel(r'ln $P_{}$'.format('{enzyme}'))
# 	ax.set_xlim(1,60)
# 	ax.set_xlabel(r'NAC ($\AA$)')
# 	ax.legend(ncol=1, bbox_to_anchor=(1.12, 0.5), loc='center left')
# 	ax.set_ylim(-8,3)
# 	ax2.set_ylim(ax.get_ylim())
# 	ax.axvline=(ranges_bulk[0])
# 	ax.axvline=(ranges_bulk[-1])
# 	ax.fill_betweenx(ax.get_ylim(), ranges_bulk[0], ranges_bulk[-1], alpha=0.5, color='gray')
# 	fig.savefig('{}/dG_nac_fitting-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(results_dir,k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), bbox_inches="tight",dpi=600)
# 	plt.clf()
# 		
# 	print('Plotting histogram of helix distance distribution for each state...')
# 	dist1_df, dist2_df=dist_c.filter(like='dist1'), dist_c.filter(like='dist2')
# 	fig, ax=plt.subplots(1,2,figsize=(10,4),sharey=True)	
# 	ax[0].set_xlabel('Distance ($\AA$)')
# 	ax[1].set_xlabel('Distance ($\AA$)')
# 	ax[0].title.set_text('Distance 1')
# 	ax[1].title.set_text('Distance 2')
# 	dist1_df.plot(kind='hist', ax=ax[0],alpha=0.5,colormap='viridis')
# 	dist2_df.plot(kind='hist', ax=ax[1],alpha=0.5,colormap='viridis')
# 	fig.savefig('{}/dist_hist-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(results_dir,k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), dpi=600)
# 	plt.clf()
# 
# 	print('Plotting boxplot of helix distance distribution for each state...')
# 	fig, ax=plt.subplots(1,2,figsize=(10,4))
# 	ax[0].set_ylabel('Distance ($\AA$)')
# 	ax[1].set_ylabel('Distance ($\AA$)')
# 	ax[0].title.set_text('Distance 1')
# 	ax[1].title.set_text('Distance 2')
# 	ax[0].set_ylim(12,30)
# 	ax[1].set_ylim(5,19)
# 	mean_1=dist1_df.mean()
# 	mean_2=dist2_df.mean()
# 	print('means of distance 1:\n', mean_1)
# 	print('means of distance 2:\n', mean_2)
# 	quantiles_1=dist1_df.quantile([0.25, 0.5, 0.75]).unstack()
# 	quantiles_2=dist2_df.quantile([0.25, 0.5, 0.75]).unstack()
# 	print('quantiles for distance 1:\n', quantiles_1)
# 	print('quantiles for distance 2:\n', quantiles_2)
# 	dist1_df.plot(kind='box', ax=ax[0], showfliers=False,colormap='viridis')
# 	dist2_df.plot(kind='box', ax=ax[1], showfliers=False,colormap='viridis')
# 	fig.savefig('{}/dist-box-{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(results_dir,k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met), dpi=600)
# 	plt.clf()
# 
# #Metastable states projected into data
# print('Plotting metastable states in {} data space...'.format(va))
# for k,v in vamp_conc.items():
# 	state_plt='state_proj_{}_{}_lg{}_vc{}_sim{}_dim{}_mlg{}_cc{}_n{}_{}_{}.png'.format(k,va,lag,v_c,sims,dim,msm_lag,cc,nstates,clu,met)
# 	fig, ax = plt.subplots(figsize=(10, 7))
# 	metastable_traj = model[k].metastable_assignments[np.concatenate(cluster[k].dtrajs)]
# 	_, _, misc = pyemma.plots.plot_state_map(*v[:, :2].T, metastable_traj, ax=ax, zorder=-1)
# 	misc['cbar'].set_ticklabels([k for k in range(1,nstates+1)])
# 	ax.set_xlabel('Component 1')
# 	ax.set_ylabel('Component 2')
# 	fig.tight_layout()
# 	fig.savefig('metastable_state_projection/' + state_plt)
# 	fig.clf()
# =============================================================================

