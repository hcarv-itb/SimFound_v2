#!/bin/python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns
plt.show(block=False)
plt.ion()
import numpy as np
import pandas as pd
import pyemma
import argparse
import os
import itertools
#import networkx as nx
import seaborn as sns
plt.style.use('seaborn-paper')

parser = argparse.ArgumentParser(
        description='Starts from discretized NAC arrays (kNAC) and builds MSMs.\nInitiates with a 4 state model and lag=100 x ps timesteps')
parser.add_argument('-prot', metavar='Protein', type=str, help='name of protein')
parser.add_argument('-mol', metavar='Substrate', type=str, help='name of substrate')

parser.add_argument('-conc', metavar='Mol', nargs='+', type=str, help='Concentrations (e.g. 0.3M)')
parser.add_argument('-concentrations', metavar='M', nargs='+', type=float, help='Concentrations NAC')
parser.add_argument('-lag', metavar='tau', type=int, nargs='+', default=[100], help='Lag time for MSM. Single or array. Default 100')
parser.add_argument('-ps', metavar='ps', type=int, help="Physical time of each timestep (ps). Must be specified")
parser.add_argument('--dt', metavar='i', type=int, default=1, help='Read each i th frame. Default 1')
parser.add_argument('--start', metavar='frame', type=int, default=20000, help='Start frame of kNAC / ps. Default 20000. i.e. kNAC starting at 20000*ps')
parser.add_argument('--stop', metavar='frame', type=int, default=-1, help='Stop frame of kNAC. Default -1')
parser.add_argument('--getFlux', action="store_true", help='Get net flux from state A to B. Requires A and B to be defined as strings')
parser.add_argument('--A', metavar='state', type=str, nargs='+', help='Source state(s) for flux analysis')
parser.add_argument('--B', metavar='state', type=str, nargs='+', help='Sink state(s) for flux analysis')
parser.add_argument('--coarseGrain', action="store_true", help='Get net flux from state A to B. Requires A and B to be defined as strings')
parser.add_argument('--cg', metavar='#Macrostate', type=int, default=2, help='Number of macrostates')
parser.add_argument('-states', metavar='Number', type=int, default=4, help='Number of states')
parser.add_argument('--stateLabels', metavar='L', type=str, nargs='+', help='State labels (states + 1)')
args = parser.parse_args()

prot=args.prot
conc=args.conc
lag=args.lag
dt=args.dt
start=args.start
stop=args.stop
mol=args.mol
ps=args.ps
concentrations=args.concentrations
nstates=args.states
state_labels=args.stateLabels


if args.getFlux == True:
	A_source=args.A
	B_sink=args.B
if args.coarseGrain == True:
	cg_states=args.cg

work_path=(os.getcwd()) + '/results/'
results_dir=work_path + 'MSM_byFrame/'
if not os.path.exists(results_dir):
	os.makedirs(results_dir)

#Set defaults
    
if state_labels == None:
    if nstates == 4:
        state_labels=['A', 'P', 'S', 'B'] #Optional
        print('Warning: No state labels were defined. Using {}'.format(state_labels))
        
    elif nstates == 5:
        state_labels=['A', 'T', 'E', 'S', 'B'] #Optional
        print('Warning: No state labels were defined. Using {}'.format(state_labels))
    else:
        print('Error: No state labels were defined. Failed since {} states not = {} labels'.format(nstates, len(state_labels)))    

#Lags for implied timescales
lags=np.array([1, 2, 5, 10, 20, 40, 100, 500, 1000, 2000, 5000])

if len(lag) != len(conc):
    print('Warning: lag times not defined for all concentrations. Using default value for of {} for all'.format(lag))
    lag=lag*len(conc)
    print(lag)

os.chdir(work_path + 'MSM_byFrame')
print("Working on {} in {} with stride {} x {} = {} ps physical time".format(prot, work_path, dt, ps, dt*ps))
print("Using {} lag steps and {} state combinations: {}".format(list(lags), nstates, state_labels))

###Mapping os states
state_encodings = list(map(list, itertools.product([0, 1], repeat=nstates)))

#names
state_names=[[] for i in range(len(state_encodings))]

for i, encoding in enumerate(state_encodings):
	for f, value in enumerate(encoding):
		if value == 1:
			state_names[i].append(state_labels[f])
for i, name in enumerate(state_names):
	state_names[i]=''.join(map(str,name))

#colors
state_color=[]
for s in range(len(state_encodings)):
	state_color.append(cm.tab20_r(s))
state_color=state_color*2

#positions
#Circle with total_states points distributed in 10x10 figure
total_states=2**nstates
theta=np.linspace(0, 2*np.pi, total_states)
a, b = 10 * np.cos(theta), 10 *np.sin(theta)
state_coords=list(zip(a,b))
###

flux_c=pd.DataFrame()
path_c=pd.DataFrame()

mfpt_c=pd.DataFrame()
rates_c=pd.DataFrame()
mfpt_cg_c=pd.DataFrame()
rates_cg_c=pd.DataFrame()
total_flux=pd.DataFrame()
net_flux=pd.DataFrame()


if concentrations == None:
    concentrations=conc


#Calculate implied timescales for all concentrations. 
for c, c_bulk in zip(conc, concentrations):
	msm_its='{}its_bmsm-{}-{}-{}-s{}.png'.format(results_dir, prot, mol, c, dt)
	if os.path.exists(msm_its):
		plot_its=False
		print("\tImplied timescales found ({}), skipping calculations".format(msm_its))
	else:
		plot_its=True
		print("\tImplied timescales...")
		kNAC=pyemma.coordinates.load('{}kNAC-{}-{}-{}-i{}-o{}-s{}.npy'.format(work_path, prot, mol, c, start*ps, stop, dt))
		msm_its=pyemma.msm.its(kNAC.flatten().astype(int), lags=lags, errors='bayes')
		its=pyemma.plots.plot_implied_timescales(msm_its, units='ps', dt=ps)
		its.set_title("[{}] = {} M".format(mol, c_bulk))
#		ax.set_xscale('log')
#		ax[conc.index(c)].label_outer()
		plt.savefig('{}its_bmsm-{}-{}-{}-s{}.png'.format(results_dir, prot, mol, c, dt), dpi=600)
		plt.clf()
		print("Warning: Check implied timescales plot to define appropriate lag time for MSM.")



if len(conc) == 7:
    columns= int(np.rint((len(conc)/2)+0.1))
else:
    columns= 2
fig_mfpt, axes = plt.subplots(2, columns, gridspec_kw={'hspace': 0.1, 'wspace':0}, sharey=True, sharex=True)
axes_list= [item for sublist in axes for item in sublist]
fig_flux, axes_flux = plt.subplots(2, columns, gridspec_kw={'hspace': 0.3, 'wspace':0}, sharey=True, sharex=True)
axesflux_list = [i for subl in axes_flux for i in subl]


for c,c_bulk,l in zip(conc, concentrations, lag):
	time=l*ps
	ax=axes_list.pop(0)
	ax_flux=axesflux_list.pop(0)
	index=pd.MultiIndex.from_product([[c], state_labels], names=['[{}]'.format(mol), 'State'])
	print("Calculating models for {} and lag {} timesteps = {} ps".format(c,l, time))


	###Calculate bMSM model with lag l
	model="{}bMSM-{}-{}-{}-s{}-{}ps.npy".format(work_path, prot, mol, c, dt, time)
	if os.path.exists(model):
		bmsm=pyemma.load(model)
		print("\tbMSM model found ({}), skipping calculations \n".format(model))
	else:
		print("\tReading data...")
		kNAC=pyemma.coordinates.load('{}kNAC-{}-{}-{}-i{}-o{}-s{}.npy'.format(work_path, prot, mol, c, start*ps, stop, dt)) #, stride=2)
		print("\tBayesian MSM with lag time {} ({} ps)....\n".format(l, time))
		bmsm=pyemma.msm.bayesian_markov_model(kNAC.flatten().astype(int), lag=l, dt_traj='{} ps'.format(ps), conf=0.95)
		bmsm.save(model, overwrite=True)		

	print(bmsm)
	states=bmsm.active_set
	statdist=bmsm.stationary_distribution
	nstates=len(states)

	print("\tActive set, length:", states, nstates)
	print('\tfraction of counts used = {:f}'.format(bmsm.active_count_fraction))
	###

	###Perform CK test for given l value
	cktest='{}cktest_bmsm-{}-{}-{}-{}ps-st{}-l{}.png'.format(results_dir, prot, mol, c, time, nstates, l)
	if os.path.exists(cktest):
		print("\tCK test found ({}), skipping calculations \n".format(cktest))
	else:
		print("\tCK test of bMSM with 5 multiples of {} ps ({} lag) and {} states....\n".format(time, l, nstates))
		cktest=bmsm.cktest(nstates, mlags=5)
		pyemma.plots.plot_cktest(cktest, dt=ps, units='ps')
		plt.savefig('{}cktest_bmsm-{}-{}-{}-{}ps-st{}-l{}.png'.format(results_dir, prot, mol, c, time, nstates, l))
#               plt.clf()
		print("Warning: Check CK test plot to validate employed lag time of MSM.")
	###


	###Prepare state labeling
	state_labels, state_colors, state_pos=[], [], []
	for v in states:
		state_labels.append(state_names[int(v)])
		state_colors.append(state_color[int(v)])
		state_pos.append(state_coords[v])	
	state_pos=np.array(state_pos)
	index=pd.MultiIndex.from_product([[c_bulk], state_labels], names=['[{}] (M)'.format(mol), 'State'])
	###

	pi=pd.DataFrame(statdist, columns=[r'$\pi$'],  index=index)
	
	###Calculate MFPTs and corresponding rates
	#Replaces the method shown in the pyEMMA documentation with nested for loops
	mfpt_table=pd.DataFrame([[bmsm.sample_mean("mfpt", i, j) for j in range(nstates)] for i in range(nstates)], index=index, columns=[c for c in state_labels])
	mfpt_table_std=pd.DataFrame([[bmsm.sample_std("mfpt", i, j) for j in range(nstates)] for i in range(nstates)], index=index, columns=[c+'_std' for c in state_labels])
	mfpt=pd.concat([mfpt_table, mfpt_table_std], axis=1)
	mfpt_c=pd.concat([mfpt_c, mfpt], axis=0, sort=True)

	rates_table=mfpt_table.apply(lambda x: (1 / x) / 1e-12)

	rates_table.replace([np.inf], np.nan, inplace=True)
	rates_table.fillna(value=0, inplace=True)

	########Inelegant way to be able to substract both dataframes since the name is different.
	########If the names are equal then concat mfpt_table and mfpt_table_std fails.
	mfpt_table_std.rename(columns={c+'_std':c for c in state_labels}, inplace=True)
	rates_table_std= 1/(mfpt_table-(mfpt_table-((mfpt_table - mfpt_table_std)*1e-12))) - rates_table
	rates_table_std.rename(columns={c:c+'_std' for c in state_labels}, inplace=True)
	########


	pyemma.plots.plot_network(rates_table.values, pos=state_pos, state_sizes=bmsm.stationary_distribution, size=12, state_labels=state_labels, arrow_labels=None, state_colors=state_colors, fontsize=7, ax=ax, arrow_scale=0.2)
	ax.set_title('[{}] = {} M\n'.format(mol, c_bulk) + r'$\tau$= {} ps'.format(time), fontsize=6) #, fontweight='bold')

#	plt.savefig('Transitions-{}-{}-{}-lag{}ps.png'.format(prot, mol, c, time), dpi=600)
#	plt.clf()

	rates=pd.concat([rates_table, rates_table_std, pi], axis=1)
	rates_c=pd.concat([rates_c, rates], axis=0, sort=True)
	###
	
	'''
	#Get lifetimes of states
	kNAC=pyemma.coordinates.load('{}kNAC-{}-{}-{}-i{}-o{}-s{}.npy'.format(work_path, prot, mol, c, start*ps, stop, dt))
	msm=pyemma.msm.MaximumLikelihoodHMSM(kNAC.flatten().astype(int), lag=l, dt_traj='{} ps'.format(ps))
	print('Lifetime expectation of states used = {:f}'.format(msm.expectation(msm.lifetimes)))
	'''

	###Coarse grain into #cg_states macrostates

	if args.coarseGrain == True:

		bmsm.pcca(cg_states)
		assignments=bmsm.metastable_assignments
		print("Metasble assignments \n", assignments)

		groups=[[] for f in np.arange(cg_states)]
		for k,v in enumerate(assignments):
			groups[v].append(state_labels[k])
		print(groups)

		index_cg=pd.MultiIndex.from_product([[c], np.arange(cg_states)], names=['[{}]'.format(mol), 'Macrostate'])

		statdist_cg=[]
		for i, s in enumerate(bmsm.metastable_sets):
			print('b_{} = {:f}'.format(i + 1, bmsm.pi[s].sum()))
			statdist_cg.append(bmsm.pi[s].sum())

		pi_cg=pd.DataFrame(statdist_cg, columns=[r'$\pi$'],  index=index_cg)

		mfpt_cg_table=pd.DataFrame([[bmsm.mfpt(bmsm.metastable_sets[i], bmsm.metastable_sets[j]) for j in range(cg_states)] for i in range(cg_states)], index=index_cg, columns=[c for c in np.arange(cg_states)])
		mfpt_cg_c=pd.concat([mfpt_cg_c, mfpt_cg_table], axis=0, sort=True)


		rates_cg_table=mfpt_cg_table.apply(lambda x: (1 / x) / 1e-12)
		rates_cg_table.replace([np.inf], np.nan, inplace=True)
		rates_cg_table.fillna(value=0, inplace=True)

		rates_cg_table=pd.concat([rates_cg_table, pi_cg], axis=1)
		rates_cg_c=pd.concat([rates_cg_c, rates_cg_table], axis=0, sort=True)


	###Flux analysis
	if args.getFlux == True:

		#Remove states in A or B that are not sampled at c. NOTE: The set of states will become different!
		A_filter = list(filter(lambda k: k in state_labels, A_source))
		A=([state_labels.index(k) for k in A_filter])

		B_filter = list(filter(lambda k: k in state_labels, B_sink))
		B=([state_labels.index(k) for k in B_filter])

		if len(A_filter) != len(A_source) or len(B_filter) != len(B_sink): 
			print("Warning: not all --A or --B states were sampled at {}".format(c))
			print("set A indexes: {} \nset B indexes: {}".format(A, B))

		flux=pyemma.msm.tpt(bmsm, A, B)
		flux_rev=pyemma.msm.tpt(bmsm, B, A)
		pos_flux=np.stack((flux.forward_committor, states/(len(state_encodings)*1.5))).T

		flux_df=pd.DataFrame(flux.flux*1e12, index=index, columns=[c for c in state_labels])
		flux_c=pd.concat([flux_c, flux_df], axis=0, sort=True)
		pyemma.plots.plot_flux(flux, state_sizes=flux.stationary_distribution, state_labels=state_labels, pos=pos_flux, show_committor=True, state_colors=state_colors, size=6, arrow_labels=None, ax=ax_flux, arrow_scale=0.2, max_height=np.max(states)) #arrow_label_format='%2.E ps$^{-1}$'
		ax_flux.set_xticks(np.arange(0,1.25,0.25))
		ax_flux.set_xticklabels(list(np.round(np.arange(0,1.25,0.25), decimals=2)), fontsize=5)
		ax_flux.set_xticks(np.arange(0,1.125,0.125), minor=True)
		ax_flux.set_title('[{}] = {} M\n'.format(mol, c_bulk) + r'$\tau$= {} ps'.format(time), fontsize=6)
#		plt.savefig('{}flux_A-B-{}-{}-{}-{}.png'.format(results_dir, prot, mol, c, time), dpi=150, bbox_inches='tight')
#		plt.clf()

		#Calculate the pathways		
		paths, path_fluxes = flux.pathways(fraction=0.9)
		path_fluxes_s=[x*1e12 for x in path_fluxes]
		path_labels=[[] for _ in paths]
		for k,v in enumerate(paths):
			for p in v:
				path_labels[k].append(state_labels[p])
#		path_labels=[k[1:-1] for k in path_labels]
		path_labels=[' -> '.join(k) for k in path_labels]
		index_flux=pd.MultiIndex.from_product([[c_bulk], path_labels], names=['[{}] (M)'.format(mol), 'pathway'])
		path_df=pd.DataFrame({'flux': path_fluxes_s}, index=index_flux)
		path_c=pd.concat([path_c, path_df], axis=0)
        #print(path_labels)
        #print(path_fluxes_s)
		print("Total flux:", path_labels) #flux.total_flux)
		print("Net flux:", path_fluxes_s) #np.sum(flux.net_flux))
		#print("gross flux:", np.sum(flux.gross_flux))

		total_c=pd.DataFrame({'total_flux': flux.total_flux}, index=[c])
		total_flux=pd.concat([total_flux, total_c*1e12])
		net_c=pd.DataFrame({'net_flux': np.sum(flux.net_flux)}, index=[c_bulk])
		net_flux=pd.concat([net_flux, net_c*1e12])

#print("Rates \n", rates_c)
#print("Rates CG \n", rates_cg_c)
#print("Flux \n", flux_c)
print("Pathway \n", path_c)


for ax in axes_list: #remove the extra subplots in fig_mfpt
	ax.remove()

#transition rates subplots
fig_mfpt.savefig('{}transitions_rates-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600, bbox_inches='tight')
#plt.clf()


for ax_flux in axesflux_list:
        ax_flux.remove()

fig_flux.tight_layout()
#fig_flux.text(0.1, 0, 'Committor Probability', ha='center', va='bottom')
fig_flux.savefig('{}fluxesAtoB-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600, bbox_inches='tight')
plt.clf()




#Stationary distributions
statDist=rates_c[r'$\pi$'].unstack(level=0)
statDist=statDist[concentrations]
statDist.plot(kind='bar', logy=True)
plt.xticks(fontsize=8, rotation=45, horizontalalignment='right')
plt.ylabel(r'Stationary Distribution ($\pi$)')
plt.savefig('{}stationaryDistribution-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600, bbox_inches='tight')
plt.clf()


pathPlot=path_c.unstack(level=1)

#Total flux
pathway_sum=pathPlot.sum(axis=1)
pathway_fraction=pathPlot.div(pathway_sum, axis=0)

pathway_sum.reindex(index=conc)


print(net_flux)
print(total_flux)
plt.plot(concentrations, net_flux.values, marker='o')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('[{}] (M)'.format(mol))
plt.ylabel(r'Net Flux ($s^{-1}$)')
plt.savefig('{}NetFlux-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600)
plt.clf()

plt.plot(concentrations, total_flux.values, marker='o')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('[{}] (M)'.format(mol))
plt.ylabel(r'Total Flux ($s^{-1}$)')
plt.savefig('{}totalFlux-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600)
plt.clf()




#Make a heatmap of all transitions
drop_list=list(rates_c.filter(regex='_std'))
drop_list.append(r'$\pi$')

rates_heatmap=rates_c.drop(columns=drop_list)
rates_heatmap=rates_heatmap.applymap(np.log10) #Get log of rates for easier visualization
rates_heatmap.replace([np.inf, -np.inf], np.nan, inplace=True)
rates_heatmap.fillna(value=0, inplace=True)

sns.set(font_scale=0.8)
sns.heatmap(rates_heatmap, linewidths=0.1, cbar_kws={'label': r'rates (log $s^{-1}$)'})
plt.xticks(rotation=45, horizontalalignment='right')

plt.savefig('{}RatesHeatmap-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600, bbox_inches='tight')
plt.clf()


#Heatmap of all fluxes
flux_heatmap=flux_c.applymap(np.log10)
flux_heatmap.replace([np.inf, -np.inf], np.nan, inplace=True)
flux_heatmap.fillna(value=0, inplace=True)

sns.set(font_scale=0.8)
sns.heatmap(flux_heatmap, linewidths=0.1, cbar_kws={'label': r'flux (log $s^{-1}$)'})
plt.xticks(rotation=45, horizontalalignment='right')
plt.savefig('{}FluxHeatmap-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600, bbox_inches='tight')
plt.clf()

#plot pathways
pathway_fraction.loc[concentrations].plot(kind='bar') #, x='[{}] (M)'.format(mol))
#plt.xtickslabels(rotation=90, horizontalalignment='right')
plt.ylabel('Flux fraction')
plt.legend(ncol=1, bbox_to_anchor=(1.05, 0.8)) #, loc='upper center')
plt.savefig('{}pathway-{}-{}-s{}.png'.format(results_dir, prot, mol, dt), dpi=600, bbox_inches='tight')



pathPlot.to_csv('pathplot-{}-{}-s{}.csv'.format(prot, mol, dt))
mfpt_c.to_csv('transitions_MFPT-{}-{}-s{}.csv'.format(prot, mol, dt))
rates_c.to_csv('transitions_rates-{}-{}-s{}.csv'.format(prot, mol, dt))
flux_c.to_csv('flux-{}-{}-s{}.csv'.format(prot, mol, dt))
net_flux.to_csv('netFlux-{}-{}-s{}.csv'.format(prot, mol, dt), header='net flux')
total_flux.to_csv('totalFlux-{}-{}-s{}.csv'.format(prot, mol, dt), header='total flux')
path_c.to_csv('pathway_flux-{}-{}-s{}.csv'.format(prot, mol, dt))




