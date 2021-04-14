#!/bin/python
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, ticker
import MDAnalysis as mda
import MDAnalysis.analysis.distances as D

mol='meoh'
def NAC_calculation(wd, name,sels,c, i):
	""""Calculates distances between pairs of atom groups (sel) for each set of selections "sels" using MDAnalysis D.distance_array method.
	Calculates the histogram (0, 140, 0.5 Angstrom) of NAC values and outputs it as a dataframe (NAC) with bins defined in 'ranges' and label 'i'.
	Stores the distances and NAC objects as .npy for each combination of start, stop and dt"""
	
	#######wd= pass wd from main ("/home/esc/calb-meoh-1M_test/structures/")
	results_dir = wd + 'results/'
	start=0
	stop=-1
	dt=1
	ranges=np.arange(0,140,0.5)
	ranges_bulk=np.arange(30,40.5,0.5)	
	wd=wd + 'structures/'
	results_dir=wd + 'results/'	
	
	xtc= wd +'{}.xtc'.format(name)
	gro= wd +'{}.gro'.format(name)
	filename_NAC='NAC_{}.npy'.format(name)
	

	u=mda.Universe(gro, xtc)
	distances={} # This is the dictionary of d distances.
	for sel in sels: #iterate through the list of sels. For each sel, define atomgroup1 (sel1) and atomgroup2 (sel2)
		d=sels.index(sel) + 1
		sel1, sel2 =u.select_atoms(sel[0]), u.select_atoms(sel[1])
		distances[d]=np.array(np.around([D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory[start:stop:dt]], decimals=3))
	
	for k in distances:
			distances[k]=np.power(distances[k], 2)
	nac=np.around(np.sqrt(sum(distances.values())/len(sels)), decimals=3) #NAC will be calculated for the len(sels) in Angstrom
	shape=np.shape(nac)
	length=shape[0]
	print("NAC from replicate {}: length {}, #sel1 {}, #sel2 {} ".format(i, shape[0], shape[1], shape[2]))
	hist, ranges=np.histogram(nac, bins=ranges)
	NAC=pd.DataFrame(zip(hist, ranges), columns=[i, 'bin'])
	NAC.set_index('bin', inplace=True)
	NAC.to_csv('{}/NAC{}-{}M.csv'.format(results_dir, len(sels), i))

	
	'''Function to normalize NAC values according to theoretic distribution in a shpere with bulk concentration "c". 
	Takes as inputs "c", average NAC histogram with standard errors and number of frames where NAC was calculated "length".
	Calculations are made with the  N_theo() function. 
	The probability of NAC for each bin is P_nac and P_nac_err. The theoretical distribution of NAC at "c" is P_t.
	The value of bulk "c" is adjusted by fitting the P_nac to the bulk region "ranges_bulk" using curve_fit(). 
	P_t_optimized uses the "c_opt" bulk concentration.
	dG is calculated from P_t_optimized.
	NOTE: The ranges_bulk should be carefully chosen for each system.'''

	ranges=NAC.index.values
	bulk_min, bulk_max=ranges_bulk[0], ranges_bulk[-1]
	
	try:
		bulk=float(str(c).split('M')[0]) #get the value (in M) of the concentration from the label of c.
		factor=0.0001
		unit='M'
	except:
		bulk=float(str(c).split('mM')[0])
		factor=0.0000001
		unit='mM'

	#Calculates the theoretical NAC values in "bulk" in a spherical shell of size given by "ranges". 
	#The output is N - 1 compared to N values in ranges, so an additional value of "0.5" is added for equivalence.
	N_theo= lambda ranges, bulk: np.log(((4.0/3.0)*np.pi)*np.diff(np.power(np.append(ranges, ranges[-1] + 0.5), 3.0))*6.022*bulk*factor)

	P_nac=np.log(NAC[i]) - np.log(length)
	P_nac.rename(columns={i:'p_nac-{}'.format(i)})
	P_t=N_theo(ranges=ranges, bulk=bulk) #original theoretical distribution with predefined bulk

	#Optimize the value of bulk with curve_fit of nac vs P_theo. Initial guess is "bulk" value. bounds are  +/- 50% of bulk value. 
	c_opt, c_cov = curve_fit(N_theo, ranges_bulk, P_nac[bulk_min:bulk_max], p0= bulk) #bounds=(bulk - bulk*0.5, bulk + bulk*0.5))

	stdev=np.sqrt(np.diag(c_cov))[0]
	c_bulk={c_opt[0]: stdev}
	
	P_opt=N_theo(ranges=ranges, bulk=c_opt)

	dG=P_opt - P_nac

	dG=pd.concat([dG, P_nac, pd.Series(P_t, index=dG.index.values), pd.Series(P_opt, index=dG.index.values)], axis=1)
	dG.columns=['dG-{}'.format(i), 'state-{}'.format(i), r'$c_{}${}={}'.format('{initial}', i, c), r'$c_{}{}={}\pm${}M'.format('{adjusted}',i, np.round(c_opt[0],decimals=3), np.round(stdev, decimals=3), unit)]

	return dG

def distance_calculation(wd, name, i):
        #######wd= pass wd from main ("/home/esc/calb-meoh-1M_test/structures/")
	start=0
	stop=-1
	dt=1
	wd=wd + 'structures/'
	results_dir=wd + 'results/'

	xtc= wd +'{}.xtc'.format(name)
	gro= wd +'{}.gro'.format(name)
	filename_dist='dist_{}.npy'.format(name)
	atom1='resname LEU and name CA and resid 144'
	atom2='(resname LEU or resname ILE) and name CA and (resid 277)'
	atom3='(resname LEU or resname ILE) and name CA and (resid 285)'
	u=mda.Universe(gro, xtc)
	a1, a2,a3=u.select_atoms(atom1), u.select_atoms(atom2), u.select_atoms(atom3) 
	d=np.array([D.distance_array(a1.positions, a2.positions, box=u.dimensions) for ts in u.trajectory[start:stop:dt]])
	dist=pd.DataFrame(d.flatten())
	d2=np.array([D.distance_array(a1.positions, a3.positions, box=u.dimensions) for ts in u.trajectory[start:stop:dt]])
	dist2=pd.DataFrame(d2.flatten())
	dist=pd.concat([dist, dist2], axis='columns')
	dist.columns=['state-{}-dist1'.format(i), 'state-{}-dist2'.format(i)]
	return dist
