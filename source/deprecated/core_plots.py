#!/bin/python3
#Function usage is described with the -h flag
import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import re
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc, ticker
plt.style.use('seaborn-paper')

def NAC_angle_plot_mean(angleNAC_c, NAC_bulkRange, P_list, results_dir, sels, protein, mol):
	"""Plots the angle distributions as a function of NAC value by mean and standard deviation. Requires the angleNAC_c dataframe and the ranges_bulk"""

	mean, std=angleNAC_c.filter(like='Mean'), angleNAC_c.filter(like='std')

	fig=mean.plot(kind='line', logx=True, legend=False)


	fig.set_xlim(1, NAC_bulkRange[-1])
	fig.yaxis.set_major_locator(ticker.MultipleLocator(20))
	fig.yaxis.set_minor_locator(ticker.MultipleLocator(5))

	fig.set_xlabel(r'$d_{NAC}$ ($\AA$)')
	fig.set_ylabel(r'$\Theta$ ($\degree$)')

	for a,b in zip(mean, std):
		lower, higher=mean[a].values - std[b].values, mean[a].values + std[b].values
		fig.fill_between(mean.index.values, lower, higher, alpha=0.5, linewidth=2)
	plt.savefig('{}/angle_nac_distMean{}-{}-{}.png'.format(results_dir, len(sels), protein, mol), dpi=600)


def NAC_distance_plot_mean(distanceNAC_c, NAC_bulkRange, P_list, results_dir, sels, protein, mol):
    """Plots the distance distributions as a function of NAC value by mean and standard deviation. Requires the distanceNAC_c dataframe and the ranges_bulk"""

#    mean, std=distanceNAC_c.filter(like='Mean'), distanceNAC_c.filter(like='std')

    mean, Q1, Q3=distanceNAC_c.filter(like='Median'), distanceNAC_c.filter(like='Q1'), distanceNAC_c.filter(like='Q3')

    fig=mean.plot(kind='line', logx=True, legend=False)


    fig.set_xlim(1, NAC_bulkRange[-1])
    fig.yaxis.set_major_locator(ticker.MultipleLocator(5))
    fig.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    fig.set_xlabel(r'$d_{NAC}$ ($\AA$)')
    fig.set_ylabel(r'$d_B - d_A$  ($\AA$)')

    '''
    for a,b in zip(mean, std):
        lower, higher=mean[a].values - std[b].values, mean[a].values + std[b].values
        fig.fill_between(mean.index.values, lower, higher, alpha=0.5, linewidth=2)
    plt.savefig('{}/distance_nac_distMean{}-{}-{}.png'.format(results_dir, len(sels), protein, mol), dpi=600)
    '''

    for a,b,c in zip(mean, Q1, Q3):
        lower, higher=Q1[b].values, Q3[c].values
        fig.fill_between(mean.index.values, lower, higher, alpha=0.5, linewidth=2)
    plt.savefig('{}/distance_nac_distQuartile{}-{}-{}.png'.format(results_dir, len(sels), protein, mol), dpi=600)









#Plot fitting of NAC for dG
def NAC_fitting_plot(dG_c, NAC_bulkRange, results_dir, sels, protein, mol):
        """Plots the N_theoretic and P_enzyme in the same plot with the y-axis.
        Requires the ranges_bulk and dG_c dataframe"""
	fig, ax=plt.subplots(sharey=True)

        N_adjusted, N_sim=dG_c.filter(like='bulk'), dG_c.filter(like='P_{enzyme}')

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(N_sim.columns)]

        fig=N_adjusted.plot(kind='line', style="--", color=colors)
        fig.set_xlim(1,NAC_bulkRange[-1] + 20)
        fig.set_ylim(-10, np.max(N_sim.values) + 2) #this is to avoid plotting large negatives in x --> 0
        #fig.legend(ncol=1, bbox_to_anchor=(1.12, 0.5), loc='center left')

        N_sim.plot(kind='line', legend=False, ax=fig, color=colors)
        fig.set_ylabel('ln N')
        fig.set_xlabel(r'$d_{NAC}$ ($\AA$)')

        #Show region used for fitting
        plt.axvline=(NAC_bulkRange[0])
        plt.axvline=(NAC_bulkRange[-1])
        plt.fill_betweenx(plt.ylim(), NAC_bulkRange[0], NAC_bulkRange[-1], alpha=0.5, color='gray')
        plt.xscale('log')
        plt.savefig('{}/dG_nac{}-fitting-{}-{}.png'.format(results_dir, len(sels), protein, mol), bbox_inches="tight", dpi=600)
        plt.clf()




#Plot dG of NAC
def NAC_dG_plot(dG_c, ranges_bulk, results_dir, sels, protein, mol):
	'''Plot the NAC dG profile. Requires the dG_c dataframe. Uses the ranges_bulk to define x-axis'''

	binding=dG_c.filter(like=mol)
	binding_err=dG_c.filter(like='err')
	fig=binding.plot(kind='line', logx=True)

	fig.set_xlabel(r'$d_{NAC}$ ($\AA$)')
	fig.set_ylabel(r'$\Delta G (k_BT)$')

	fig.axhline(y=0, color='black')

	fig.set_xlim(1, ranges_bulk[-1])
	fig.set_ylim(-4, 5)
#	fig.yaxis.set_major_locator(ticker.MultipleLocator(1))
#	fig.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

	for b, e in zip(binding.columns, binding_err.columns):
	        lower, higher=binding[b].values - binding_err[e].values, binding[b].values + binding_err[e].values
	        fig.fill_between(dG_c.index.values, lower, higher, alpha=0.5)

	plt.savefig('{}/dG_nac{}-{}-{}.png'.format(results_dir, len(sels), protein, mol), dpi=600)
	plt.clf()



def distances_hist(distance_c, results_dir, dists, protein, mol, P_list):
	'''Plot the histogram of distances. Requires the --calcDists to be invoked and the distance_c df to be generated
	NOTE: It plots two vertical lines corresponding to the distances on PDB ID 5a71'''

	plt.rc('legend', fontsize=8)
	fig, ax=plt.subplots(len(P_list), len(dists), sharey=True, sharex=True)

	for i in range(len(dists)):
		dist_index=distance_c.filter(like='d{}@'.format(i+1))

		for c in P_list:
			dist_c=dist_index.filter(like=c)
			if len(P_list) > 1:
				dist_c.plot(kind='hist', ax=ax[P_list.index(c)][i], density=True)
				ax[P_list.index(c)][i].legend(['{} {}'.format(mol, c)])
				ax[0][i].title.set_text('Distance {}'.format(i+1))
				ax[-1][i].set_xlabel(r'$\AA$')
                                #ax[P_list.index(c)][i].set_ylim(0, 0.6)
				ax[P_list.index(c)][i].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
				ax[P_list.index(c)][i].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                                #Useful only for CALB-acyl
				if i+1 == 1:
					ax[P_list.index(c)][i].axvline(x=11.1, color='red', linestyle='--')
					ax[P_list.index(c)][i].axvline(x=7.7, color='green', linestyle='--')
				elif i+1 == 2:
					ax[P_list.index(c)][i].axvline(x=11.5, color='red', linestyle='--')
					ax[P_list.index(c)][i].axvline(x=2.8, color='green', linestyle='--')
                                ###
			else:
				dist_c.plot(kind='hist', ax=ax[i], density=True, alpha=0.5)
				ax[i].legend(['NAC frames', '[{}]={}'.format(mol, c)])
				ax[i].title.set_text('Distance {}'.format(i+1))
				ax[-1].set_xlabel(r'$\AA$')
                                #ax[i].set_ylim(0, 0.6)
				ax[i].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
				ax[i].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        #fig.tight_layout()
	fig.suptitle(mol)
	fig.savefig('{}/distances_hist{}-{}-{}.png'.format(results_dir, len(dists), protein, mol), dpi=600, bbox_inches='tight')
	plt.clf()


def distances_hist_wNAC(distance_NAC_c, results_dir, dists, protein, mol, P_list, co):
	'''Plot the histogram of distances. Requires the distances_df_c dataframe and the --calcDists to be invoked'''


	print(P_list)

	print(distance_NAC_c)


	plt.rc('legend', fontsize=6)
	fig, ax=plt.subplots(len(P_list), len(dists), sharey=True, sharex=True)

	for i in range(len(dists)):
		dist_index=distance_NAC_c.filter(like='d{}@'.format(i+1))
		
		for c in P_list:
			dist_c=dist_index.filter(like=c)
			if len(P_list) > 1:			
				dist_c.plot(kind='hist', ax=ax[P_list.index(c)][i], density=True, alpha=0.5, legend=False)
#				ax[P_list.index(c)][i].legend(['NAC frames', '[{}]={}'.format(mol, c)])
				ax[0][i].title.set_text('Distance {}'.format(i+1))
				ax[P_list.index(c)][i].set_ylim(0, 0.6)
				ax[-1][i].set_xlabel(r'$\AA$')
				ax[P_list.index(c)][i].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
				ax[P_list.index(c)][i].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
				#Useful only for CALB-acyl
				if i+1 == 1:
					ax[P_list.index(c)][i].axvline(x=11.1, color='red', linestyle='--') 
					ax[P_list.index(c)][i].axvline(x=7.7, color='green', linestyle='--')
				elif i+1 == 2:
					ax[P_list.index(c)][i].axvline(x=11.5, color='red', linestyle='--')
					ax[P_list.index(c)][i].axvline(x=2.8, color='green', linestyle='--')
					
				###
			else:
				dist_c.plot(kind='hist', ax=ax[i], density=True, alpha=0.5)
				ax[i].legend(['NAC frames', '[{}]={}'.format(mol, c)])
				ax[i].title.set_text('Distance {}'.format(i+1))
				ax[-1].set_xlabel(r'$\AA$')
				ax[i].set_ylim(0, 0.6)
				ax[i].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
				ax[i].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
	#fig.tight_layout()		
	fig.text(0.5, 0.04, 'Frequency', ha='center')
	fig.suptitle(r'NAC= {} $\AA$'.format(co))
	fig.savefig('{}/distances_hist{}-{}-{}_NAC{}.png'.format(results_dir, len(dists), protein, mol, co), dpi=600)
	plt.clf()

	
def kNAC_plot(kNAC_hist, results_dir, protein, mol, sels, states):
	'''Plots the distributions of states as a function of concentration
	Legacy: Generate directly the histogram and plot it. Can get confusing due to overlaps. Use the kNAC_c as input instead'''

	fig=kNAC_hist.plot(kind='bar', title='States', logy=True)

	fig.set_ylabel('Sampled Fraction')
	fig.xaxis.label.set_visible(False)
	plt.xticks(fontsize=8, rotation=45, horizontalalignment='right')

	plt.tight_layout()	
	plt.savefig('{}/kNAC{}-{}-{}_{}-{}-{}.png'.format(results_dir, len(sels), protein, mol, len(states)+1, str(states[0]), str(states[-1])), dpi=600)


def Nmol_plot(results_dir, prot, mol, Nmol, fit, langmuir_df, ps, Nmol_range, Nmol_integral):
	"""Plots the Nmol as a function of feature with a series for each bin of Nmol_range"""

#	color_list=(plt.rcParams['axes.prop_cycle'].by_key()['color'])*2
#	fit.dropna(inplace=True)
	fit.replace(np.inf, np.nan, inplace=True)
	fit.fillna(value=0, inplace=True)
	print(fit)
	#Plot Langmuit fitting
	x_nmol=Nmol[Nmol.columns.levels[0].values[-2]].values
	labels=['{} $\AA$'.format(k) for k in (Nmol.columns.levels[0].values[:-2])]


	for k, nmol_init in enumerate(Nmol.columns.levels[0].values[:-2]):
		y_nmol=Nmol[(nmol_init, 'average')].values
		y_nmol_err=Nmol[(nmol_init, 'sem')].values
		plt.scatter(x_nmol, y_nmol, zorder=100, s=3)
#		color=color_list[k]
		plt.errorbar(x_nmol, y_nmol, yerr=y_nmol_err, zorder=0, linestyle='None', capsize=5, elinewidth=1, markeredgewidth=1)
#		plt.errorbar(x_nmol, y_nmol, yerr=y_nmol_err, zorder=0, linestyle='None', ecolor=color, capsize=5, elinewidth=1, markeredgewidth=1) 
		plt.plot(langmuir_df.index.values, langmuir_df[nmol_init], color='grey', linewidth=1)
#	plt.legend(labels)
	plt.xlabel(Nmol.columns.levels[0].values[-2])
	plt.ylabel(r'$N_{mol}$')
	plt.yscale('log')
	plt.title(r'{} within {} < NAC $\leq$ {} $\AA$'.format(mol, Nmol_range[0], Nmol_range[-1]))

	plt.savefig('{}/langmuir_{}-{}_{}-{}.png'.format(results_dir, prot, mol, Nmol_range[0], Nmol_range[-1]), dpi=600, bbox_inches='tight')
	plt.clf()

#	fit['labels']=labels
#	fit.plot.bar(x='labels', y=[('Kd', 'value'), ('Nmax', 'value')], yerr=[fit[('Kd', 'std')].values, fit[('Nmax', 'std')].values], color=color_list[0:len(fit)], error_kw=dict(elinewidth=1, markeredgewidth=1,capzise=5), legend=False, subplots=True, logy=True)
#	plt.savefig('{}/langmuir_parameters_bar-{}-{}.png'.format(results_dir, prot, mol), dpi=600, bbox_inches='tight')		


	#Plot langmuir parameters
	fig, ax=plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0.05})


	ax[0].bar(fit.index.values, fit['Kd', 'value'].values, width=Nmol_integral, alpha=0.7) #Change width value if bins are different
	ax[0].errorbar(fit.index.values, fit['Kd', 'value'].values, yerr=fit['Kd', 'std'].values, zorder=0, linestyle='None', capsize=5, elinewidth=1, markeredgewidth=1, lolims=False)
#	ax[0].fill_between(fit.index.values,  fit['Kd', 'value'].values - fit['Kd', 'std'].values, fit['Kd', 'value'].values + fit['Kd', 'std'].values, alpha=0.5)
	ax[0].set_yscale('log')
	ax[0].set_ylabel(r'$K_{D}$ (M)')
	ax[0].set_xlim(1, fit.index.values[-1])

	ax[1].bar(fit.index.values, fit['Nmax', 'value'].values, width=Nmol_integral, alpha=0.7)
#	ax[1].fill_between(fit.index.values,  fit['Nmax', 'value'].values - fit['Nmax', 'std'].values, fit['Nmax', 'value'].values + fit['Nmax', 'std'].values, alpha=0.5)
#	ax[1].scatter(fit.index.values, fit['Nmax', 'value'].values)
	ax[1].errorbar(fit.index.values, fit['Nmax', 'value'].values, yerr=fit['Nmax', 'std'].values, zorder=0, linestyle='None', capsize=5, elinewidth=1, markeredgewidth=1, lolims=False)
	ax[1].set_yscale('log')
	ax[1].set_xscale('log')
	ax[1].set_ylabel(r'$N_{max}$')
	ax[1].set_xlabel('$d_{NAC}$ ($\AA$)')

	plt.savefig('{}/langmuir_parameters-{}-{}_{}-{}.png'.format(results_dir, prot, mol, Nmol_range[0], Nmol_range[-1]), dpi=600, bbox_inches='tight')


