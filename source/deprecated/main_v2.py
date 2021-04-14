#!/bin/python
#Function usage is described with the -h flag
import os
import numpy as np
#import h5py
#import cupy as cp
import pandas as pd
import argparse
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time as time_clock
import pickle
from multiprocessing import Process, Manager
import itertools
import re

import sys
sys.path.insert(0, '/home/media/dataHog/hca/proLig')
#from proLig import *
from subAnalysis import *
from plot_functions import *


manager=Manager()

#np=cp.get_array_module()
#print(np)
#plt.show(block=True)
plt.ion()


path=(os.getcwd())
results_dir=path + '/results/'
temp_dir='/tmp'

#Definition of input options from arguments
parser = argparse.ArgumentParser(
        description='Main script to execute multiprocessor tasks', fromfile_prefix_chars='@')
parser.add_argument("-prot", metavar='Molecule', type=str, help='protein name')
parser.add_argument('-mol', metavar='Molecule', type=str, help='substrate name')
parser.add_argument('-conc', metavar='Molar', type=str, nargs='+', help='Substrate concentration(s)')
parser.add_argument('--concentrations', metavar='Molar', type=float, nargs='+', help='Bulk substrate concentration(s) (M)')

parser.add_argument('-feature', metavar='MACRO', type=str, nargs='+', help='Feature(s) to be calculated')
parser.add_argument('--calcNAC', action="store_true", help='Calculate NAC. Must specify --sel and --bulk')
parser.add_argument('--sel', metavar='Selection', type=str, nargs='+', action='append', help='selection or @selections file of distances')
parser.add_argument('--bulk', metavar='Angstrom', type=float, nargs=2, help='Lower, upper boundary of region for NAC fit to bulk')

parser.add_argument('--NACstates', action="store_true", help='Discretize NAC values into state. Must specify --states')
parser.add_argument('--states', metavar='Angstrom', type=float, nargs='+', help='Selection(s): state upper bounds')
parser.add_argument('--stateLabels', metavar='L', type=str, nargs='+', help='State labels (states + 1)')

parser.add_argument('--dist_NAC', action="store_true", help='Distances vs NAC calculation . Must specify --dist')
parser.add_argument('--dist', metavar='Selection', type=str, nargs='+', action='append', help='selection or @selections file of distances')

parser.add_argument('--angle_NAC', action="store_true", help='Angle vs NAC calculation. Must specify --angle.')
parser.add_argument('--angle', metavar='Selection', type=str, nargs='+', action='append', help='selection or @selections file of atoms')

parser.add_argument('-ps', metavar='ps', type=int, help="Physical time of each timestep (ps). Must be specified")
parser.add_argument('--dt', metavar='i', type=int, default=1, help='Read each i th frame. Default 1')
parser.add_argument('--tstart', metavar='ps', type=int, default=0, help='Start time to read trajectory. Default 0')
parser.add_argument('--tstop', metavar='ps', type=int, default=-1, help='Stop time to read trajectory. Default -1')
parser.add_argument('--reps', metavar='N', type=int, default=10, help='Number N of simulation replicates. Default 10')
parser.add_argument('--init', metavar='N', type=int, default=1, help='Initial N value. Default 1')

parser.add_argument('--extractFrames', action="store_true", help='Extract frames with specified NACs. Must specify --cutoff(s)')
parser.add_argument('--cutoff', metavar='Angstrom', type=float, nargs='+', help='NAC cutoff value(s) in angstrom.')
parser.add_argument('--densityCalc', action="store_true", help='Calculate density map of extracted frames. Must specify --cutoff(s)')
parser.add_argument('--density_sel', metavar='Selection', type=str, nargs=1, default=None, help='Selection for density map')

parser.add_argument('--nmol', action="store_true", help='Langmuir fitting of molecules within shell of "Nmol_integral" = 1 angstrom of NAC. Must specify --nmol_range')
parser.add_argument('--nmol_range', metavar='Angstrom', nargs=2, type=float, help='Lower, upper boundary of NAC values for Langmuir fitting.')

parser.add_argument('--ToF', action="store_true", help='Wether to calculate and store feature (water)')

args = parser.parse_args()

#Define inputs
calcNAC=args.calcNAC

NACstates=args.NACstates
dist_NAC=args.dist_NAC
angle_NAC=args.angle_NAC
nmol=args.nmol
extractFrames=args.extractFrames
prot=args.prot
mol=args.mol
conc=args.conc
concentrations=args.concentrations
ps=args.ps
dt=args.dt
start=args.tstart / ps
tstop=args.tstop
densityCalc=args.densityCalc

feature=args.feature
ToF=args.ToF


cutoff=args.cutoff
reps=args.reps + 1 
rep=args.init
bulk=args.bulk

sels=args.sel

angles=args.angle


if tstop == -1:
	stop= tstop
else:
	stop= int(tstop / ps)

if concentrations == None:
	concentrations = conc

#Check inputs and create directories
print("\nFeature calculation for {} and {} at concentration(s) {} in {} replicates (initial {})".format(prot, mol, conc, reps - 1 , rep))
print("Start at {} ps (frame {}) and end at {} ps (frame {}) with stride of {}. Timestep of {} ps.".format(args.tstart, start, tstop, stop, dt, ps))


if not os.path.exists(results_dir):
	os.makedirs(results_dir)

if not os.path.exists(results_dir + 'structures/'):
	os.makedirs(results_dir + 'structures/')

print("path: " + path)
print("results: " + results_dir)

# Populate features list
# =============================================================================

#List of features to be calculated. (Fix length)
features=[]

featureBuilder={'test':1, 'test2':2}

# =============================================================================
# macro={}

# =============================================================================


#kwargs are the arguments required by most functions
kwargs={'results_dir':results_dir,
        'temp_dir':temp_dir,
        'dt':dt,
        'mol':mol,
        'prot':prot,
        'start':start,
        'stop':stop,
        'ps':ps}


#Tricks to skip NAC and angle calculation routines if files already present.
if sels == None:
    sels_string=str(glob.glob(results_dir + 'dG_nac*-{}-{}*.csv'.format(prot, mol)))
    try:
        #find dG_nac file and retrieve length of selections by creating a dummy sels array
        sels=np.arange(0, int(re.split('-{}'.format(prot), re.split('dG_nac', sels_string)[1])[0]))
        print("Warning: NAC selections not defined. Found length of {} for selections in stored dG_nac file".format(len(sels)))
    except:
        print("Error: NAC selections not found. Define selections")
kwargs.update({'sels':sels})

angleCalc=True
if angles == None:
    angles_string=str(glob.glob(results_dir + 'ANGLES*-{}-{}*.csv'.format(prot, mol)))
    try:
        #find ANGLES file and retrieve length of angles by creating a dummy angles array
        angles=np.arange(0, int(re.split('-{}'.format(prot), re.split('ANGLES', angles_string)[1])[0]))
        print("Warning: Angles not defined. Found length of {} for angles in stored ANGLES file".format(len(angles)))
        angleCalc=False
    except:
        print("Error: Angle not found. Define angle")

kwargs.update({'angles':angles})


# Populate features list
# =============================================================================

#Macros for NAC calculations)
resolution=0.5
NAC_range=np.arange(0, 180.5, resolution)
NAC_midRange=NAC_range[:-1] + (resolution / 2)

try:
    NAC_bulkRange=np.arange(bulk[0], bulk[1], resolution)
    print('Bulk regions for NAC calculation between {} and {}'.format( bulk[0], bulk[-1]))
except:
    print('Warning: bulk region not defined')


if extractFrames == True:
    features.append('extractFrames')
    kwargs.update({'cutoff':cutoff})



if calcNAC == True:
    print("NAC calculation within range {} to {} in bin of {} angstrom".format(NAC_range[0], NAC_range[-1], resolution))
  
    for idx, sel in enumerate(sels, 1):
        print("\tNAC distance {}: {}".format(idx, str(sel)))

    features.append('nac_rep')
    features.append('length')
    dG_c=pd.DataFrame()
    nac_c=pd.DataFrame()
    bulk_fit_c=pd.DataFrame()
    kwargs.update({'NAC_range':NAC_range, 'NAC_midRange':NAC_midRange, 'NAC_bulkRange':NAC_bulkRange, 'resolution':resolution})


if angle_NAC == True:

    print("\tAngle: {}".format(str(angles)))
    angleNAC_c=pd.DataFrame()
    features.extend(['angles_dist_array'])

if dist_NAC == True:
    dists=args.dist
    features.extend(['distance_dist_array'])
    for dist in dists:
        print("\tDistance {}: {}".format(dists.index(dist) + 1, str(dist)))
    distanceNAC_c=pd.DataFrame()
    kwargs.update({'dists':dists})


if NACstates == True:
    states=args.states
    state_labels=args.stateLabels
    features.append('nac_states')
    
    if state_labels == None:
        state_labels=['A', 'P', 'S', 'B'] #Optional
        if len(state_labels) == len(states) + 1:
            print('Warning: No state labels were defined. Using {}'.format(state_labels))
        else:
            print('Error: No state labels were defined. Failed to use {} since {} states not = {} labels'.format(state_labels, 
                  len(states), len(state_labels)))
            
    #Create a mapping of state index to combination of labels.
    state_encodings = map(list, itertools.product([0, 1], repeat=len(states)+1))
    state_names=[[] for i in range(len(state_encodings))]
    for i, encoding in enumerate(state_encodings):
        for f, value in enumerate(encoding):
            if value == 1:
                state_names[i].append(state_labels[f])
    for i, name in enumerate(state_names):
        state_names[i]=''.join(map(str,name))
        
    kwargs.update({'state_encodings':state_encodings, 'states':states})
    kNAC_c=pd.DataFrame()


if nmol == True:
	#Bin values in 1A 
	Nmol_integral=1.0 #width of bin for Nmol count
	Nmol_range=np.arange(args.nmol_range[0], args.nmol_range[1], Nmol_integral)
	features.append('nmol')
	print("Number of molecules at NAC range {} to {} with {} angstrom integral.".format(Nmol_range[0], Nmol_range[-1], Nmol_integral))
	kwargs.update({'Nmol_range':Nmol_range, 'Nmol_integral':Nmol_integral})
	Nmol_c=pd.DataFrame()
    

if ToF == True:
    mol='water-' + mol
    results_dir_tof=results_dir + '/tof/'
    
    if not os.path.exists(results_dir_tof):
	    os.makedirs(results_dir_tof)
    
    kwargs.update({'results_dir_tof':results_dir_tof})

#print('kwargs:', kwargs)
    
print('Features:{}'.format(features))
# =============================================================================


clock_i = time_clock.time()

#Features are calculated for each replicate as defined here
# =============================================================================
def replicate_iterator(features_multiproc, i):
    """ Core function of the workflow: Loops through each replicate to obtain the defined features.
    Each core will perform all instructions given here.
    Requires the replicate index 'i' to set the working directory and the list of features 'features_multiproc'."""

    replicate=("{}/{}-{}-{}/sim{}/".format(path, prot, kwargs['mol'], c, str(i)))
    if not os.path.exists(replicate + 'results/'): #Create a results folder for each replicate
        os.makedirs(replicate + 'results/')

    if mol == 'pNPB':
        TPR=prot + '_pr.gro'
        XTC=prot + '_pr.xtc'
    elif c == '50mM_cont':
        TPR=prot + '.gro'
        XTC=prot + '.xtc'
    else:
        TPR=prot + '-eq.gro'
        XTC=prot + '.xtc' 

    kwargs.update({'replicate':replicate, 'c':c, 'i':i, 'TPR':TPR, 'XTC':XTC})
    
    if os.path.isfile(replicate + TPR) and os.path.isfile(replicate + XTC):

        if ToF == True:
            features_multiproc['nac_rep'][i], features_multiproc['nac_states'][i], features_multiproc['length'][i], features_multiproc['nmol'][i]=NAC_calculation_water(**kwargs)
        
        else:
            if calcNAC == True:
                features_multiproc['nac_rep'][i], features_multiproc['length'][i]=NAC_calculation(**kwargs)
    
            if extractFrames == True:
                extract_frames(**kwargs)			
    
            if angle_NAC == True:
                features_multiproc['angles_dist_array'][i]=nac_angles(**kwargs)
    
            if dist_NAC == True:
                features_multiproc['distance_dist_array'][i]=nac_distances(**kwargs)      
    
            if NACstates == True:
                features_multiproc['nac_states'][i]=NAC_discretization_byFrame(**kwargs)
    
            if nmol== True:
                features_multiproc['nmol'][i]=nac_Nmol(**kwargs)

    else:
        print("Warning: Could not load trajectory files of {} replicate {}. Skipping calculations".format(c, i))
# =============================================================================


for c, c_fitted in zip(conc, concentrations):
    print("Working on concentration {}".format(c))
    #if len(features) != 0:
    #Create empty dataframes and multiprocessing dictionary of features
    index=pd.MultiIndex.from_product([[c], []], names=['[{}]'.format(mol), 'feature'])
    features_df={name: pd.DataFrame(index=index) for name in features}
    features_multiproc={name: manager.dict() for name in features}
	
	#Use multiprocessing to launch jobs, one core per replicate. 
    job = [Process(target=replicate_iterator, args=(features_multiproc, i)) for i in range(rep, reps)]
    _=[p.start() for p in job]
    _=[p.join() for p in job]
	
	#Collect the results of each job for and features

    #Zip the multiprocessing dictionary 'feat_job' and the dataframe 'feat'
    for feat_job, feat in zip(features_multiproc,features_df): 
        print("Merging results from {}.".format(feat_job))
        #Collect results from each replicate (feat_job), as given by the number of k (pickle) or v (df) items.
        for k,v in features_multiproc[feat_job].items():
            try:
                infile=open('{}/{}_pickle-{}'.format(temp_dir, feat_job, k), 'rb')
                f=pickle.load(infile)
                infile.close()
                
                #Results from jobs are concatenated in different ways: 0 rows, 1 columns
                if feat_job == 'angles_dist_array' or feat_job == 'nac_states' or feat_job == 'distance_dist_array':
                    features_df[feat]=pd.concat([features_df[feat], f], axis=0)
                else:	
                    features_df[feat]=pd.concat([features_df[feat], f], axis=1)
            except:
                features_df[feat]=pd.concat([features_df[feat], v], axis=1)	

            #Save feature dataframes in HDF format. 
            #features_df[feat].to_hdf('{}/HDF-{}-{}.h5'.format(temp_dir, prot, mol), key='{}_{}'.format(feat,c))


	#Processing of features is made here. Once done, results are appended to concentration dataframes

	#Calculate NAC histograms and obtain dG profiles
    if calcNAC == True:
        
        #NAC the averages and sem are calculated with histogram_nac()
        nac_hist=histogram_nac(features_df['nac_rep'], c)
        nac_c=pd.concat([nac_c, nac_hist], axis=1)
        
	    #dG values and bulk concentration obtained from NAC averages are calculated with NAC_normalization()
        dG_df, c_bulk=NAC_normalization(nac_hist=nac_hist, length=features_df['length'], c=c, mol=mol, NAC_bulkRange=NAC_bulkRange, resolution=resolution)
        bulk_fit_c=pd.concat([bulk_fit_c, pd.DataFrame({'c_opt':c_bulk.keys(), 'c_std':c_bulk.values()})], axis=0)
        dG_c=pd.concat([dG_c, dG_df], axis=1)

	#Obtain angle distributions as a function of NAC values. Different ways to represent the data can be made.
    if angle_NAC == True:
        angleNAC=features_df['angles_dist_array']
        angleNAC_dist=angleNAC['angle'].groupby(pd.cut(angleNAC['nac'], NAC_range)).agg(['mean', 'std']) #Mean and s.e.m.
        angleNAC_dist.columns= ['Mean-'+c, 'std-'+c]
	
	    #angleNAC_dist=angleNAC['angle'].groupby(pd.cut(angleNAC['nac'], ranges)).quantile([0.25, 0.5, 0.75]).unstack() #Q1, Median and Q3
	    #angleNAC_dist.columns= ['Q1-'+c, 'Median-'+c, 'Q3-'+c]

	    #angleNAC_dist=angleNAC['angle'].groupby(pd.cut(angleNAC['nac'], ranges)).agg(['mean', 'min', 'max']) #Mean, min and max
	    #angleNAC_dist.columns= ['Mean-'+c, 'min-'+c, 'max-'+c]
	
        angleNAC_dist.set_index(NAC_midRange, inplace=True)
        angleNAC_c=pd.concat([angleNAC_c, angleNAC_dist], axis=1)

    #Calculates the distribution of distances. Legacy: used for measurement of distances in NAC isolated frames (cutoff)
    if dist_NAC == True:
        distanceNAC=features_df['distance_dist_array']
        #distanceNAC_dist=distanceNAC['distance'].groupby(pd.cut(distanceNAC['nac'], NAC_range)).agg(['mean', 'std']) #Mean and s.e.m.
        #distanceNAC_dist.columns= ['Mean-'+c, 'std-'+c]
        
        distanceNAC_dist=distanceNAC['distance'].groupby(pd.cut(distanceNAC['nac'], NAC_range)).quantile([0.25, 0.5, 0.75]).unstack()
        distanceNAC_dist.columns= ['Q1-'+c, 'Median-'+c, 'Q3-'+c]
        distanceNAC_dist.set_index(NAC_midRange, inplace=True)
        distanceNAC_c=pd.concat([distanceNAC_c, distanceNAC_dist], axis=1)
        
        #legacy: Use to generate histograms of distances.
        '''
	    #Inelegant way to combine all replicates of each dist into one column and then rejoin the dist dataframes
	    df=pd.DataFrame()
	    for dist in dists:
		    d=features_df['distances'].filter(regex='d{}'.format(dists.index(dist)+1))
		    d=d.values.flatten()
		    df=pd.concat([df, pd.DataFrame(d, columns=['d{}@{}'.format(dists.index(dist)+1, c)])], axis=1)
	    ###
	    distance_NAC=distance_calculation(dists=dists, prot=prot, mol=mol, c=c, cutoff=cutoff, results_dir=results_dir)
        distance_NAC_c=pd.concat([distance_NAC_c, distance_NAC], axis=1)
        distance_NAC_c=pd.concat([distance_NAC_c, df], axis=1)
        '''


	#Pools the NAC values binned into states ('nac_states') and saves the .npy object for MSM calculation
    if NACstates == True:
	    filename='kNAC-{}-{}-{}-i{}-o{}-s{}.npy'.format(prot, mol, c, start*ps, stop, dt)

            print(features_df['nac_states'])
	    features_df['nac_states'].columns=[c_fitted]
	    np.save(results_dir + filename, features_df['nac_states'].values.flatten())		
	
	    features_df['nac_states'].reset_index(inplace=True, drop=True)
	    kNAC_c=pd.concat([kNAC_c, features_df['nac_states'].astype(int)], axis=1)
#		bMSM=bMSM_calculation(prot=prot, mol=mol, c=c, start=start, stop=stop, dt=dt, ps=ps, results_dir=results_dir)
	#Pools the number of counts and calculates the statistics of number of molecules
    if nmol == True:
	    Nmol_c=pd.concat([Nmol_c, features_df['nmol']], axis=0)

#####
#Save and plot features


if calcNAC == True:
	nac_c.to_csv('{}/NAC{}-{}-{}.csv'.format(results_dir, len(sels), prot, mol))
	dG_c.to_csv('{}/dG_nac{}-{}-{}.csv'.format(results_dir, len(sels), prot, mol))
	bulk_fit_c.set_index([conc], inplace=True)
	bulk_fit_c.to_csv('{}/nac_concentrations-{}-{}.csv'.format(results_dir, prot, mol))
	NAC_fitting_plot(dG_c, NAC_bulkRange, results_dir, sels, prot, mol)
	NAC_dG_plot(dG_c, NAC_bulkRange, results_dir, sels, prot, mol)

if angle_NAC == True:
	angleNAC_c.to_csv('{}/angle_nac_dist{}-{}-{}.csv'.format(results_dir, len(sels), prot, mol))
#	angle_dist_c.to_csv('{}/angle_nac{}-{}-{}.csv'.format(results_dir, len(sels), prot, mol))
#	angles_c.to_csv('{}/ANGLES{}-{}-{}.csv'.format(results_dir, len(angles), prot, mol))

	NAC_angle_plot_mean(angleNAC_c, NAC_bulkRange, conc, results_dir, sels, prot, mol)
#       NAC_angle_plot_quartiles() or NAC_angle_plot_lims()


if dist_NAC == True:
    distanceNAC_c.to_csv('{}/distancesNAC_df{}-{}-{}.csv'.format(results_dir, len(dists), prot, mol))
    NAC_distance_plot_mean(distanceNAC_c, NAC_bulkRange, conc, results_dir, sels, prot, mol)

    #Legacy: Use to generate histogram plots
    '''
    if cutoff != None:
        for co in cutoff:
            distances_hist_wNAC(distance_NAC_c, results_dir, dists, prot, mol, conc, co)
    else:
        distances_hist(distance_NAC_c, results_dir, dists, prot, mol, conc)
    '''


if NACstates == True:
	kNAC_c.to_csv('{}/kNAC{}-{}-{}_{}-{}-{}.csv'.format(results_dir, len(sels), prot, mol, len(states)+1, str(states[0]), str(states[-1])))
	
	kNAC_hist=pd.DataFrame(index=np.arange(0, len(state_names)))

	kNAC_hist=kNAC_hist.join(kNAC_c.apply(pd.value_counts)/len(kNAC_c))
	kNAC_hist.index.names=['state']
	kNAC_hist=kNAC_hist.join(pd.DataFrame(state_names, columns=['Label']))
	kNAC_hist.set_index('Label', inplace=True)
	kNAC_hist.to_csv('{}/kNAC_hist{}-{}-{}_{}-{}-{}.csv'.format(results_dir, len(sels), prot, mol, len(states)+1, str(states[0]), str(states[-1])))
	kNAC_plot(kNAC_hist, results_dir, prot, mol, sels, states)

if nmol == True:
	Nmol, fit, langmuir_df=Nmol_stats(Nmol_c, results_dir, prot, mol)

	Nmol_plot=Nmol_plot(results_dir, prot, mol, Nmol, fit, langmuir_df, ps, Nmol_range, Nmol_integral)
	fit.to_csv('{}/langmuir_parameters-{}-{}_{}-{}.csv'.format(results_dir, prot, mol, Nmol_range[0], Nmol_range[-1]))
	Nmol.to_csv('{}/number_molecules_nac-{}-{}_{}-{}.csv'.format(results_dir, prot, mol, Nmol_range[0], Nmol_range[-1]))
#####

[os.remove(f) for f in glob.glob(temp_dir + '/*pickle-*')]

clock_o=time_clock.time()
duration= clock_o - clock_i
print("time (s): {}".format(np.round(duration, decimals=2)))
