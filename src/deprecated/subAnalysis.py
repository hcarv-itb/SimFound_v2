#!/bin/python
#Function usage is described with the -h flag
import os
import glob
import numpy as np
try:
    import cupy as cp
except:
    pass
try:
    import fast_histogram
except:
   pass
from scipy.optimize import curve_fit
import pandas as pd
import MDAnalysis as mda
import MDAnalysis.analysis.distances as D
from MDAnalysis.analysis.density import density_from_Universe
from MDAnalysis.analysis import align
from six import itervalues
import pickle
import itertools
from itertools import permutations
import time

def extract_frames(**kwargs):
	"""Extract frames with NAC within 0.05 A from cutoff value(s) for any substrate column and convert to timestep (ps).
        Requires the NAC array to be loaded from NAC_calculation().
        Writes the corresponding frames as .xtc and the corresponding .gro. Legacy: option to save also the waters controlled with --noWater.
        Adapted from Gudrun's script"""

	for k, v in kwargs.items():
		exec(k+'=v')


	
	filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sels), mol, c, start*ps, stop, dt, i)
	NAC_array=np.load(replicate + 'results/' + filename_NAC)
	shape=np.shape(NAC_array)	

        ###Rebuild NAC array (frames, prot_ref, subs) into a frames, prot_ref*subs dataframe. NOTE: Requires test n > 1 prot refs
	#TODO: Create a multiindex with prot_ref and subs. Test if sel_frames still works for this.

	NAC=pd.DataFrame(NAC_array.reshape(shape[0], shape[1]*shape[2]))
	NAC.columns=NAC.columns + 1
	stop=start + shape[0]
	NAC.set_index(np.arange(start, stop, dt), inplace=True)

	'''Legacy of rebuild NAC array. New version seems to be faster but looses information on the subs-prot_ref pair.
	shape=np.shape(NAC_array)

	stop=start + shape[0]

	NAC=pd.DataFrame()
	for prot_ref in range(0, shape[1]):
		for sub in range(0, shape[2]):
			sub_series=pd.Series(NAC_array[:,0,sub], name=sub+1) #This is a column of all frames from a substrate
			NAC=pd.concat([NAC, sub_series], axis=1)	
	NAC.set_index(np.arange(start, stop, dt), inplace=True)
	'''
	u=mda.Universe(replicate + TPR, replicate + XTC)
	for co in cutoff:
		nac_low, nac_high =co - 0.05, co + 0.05
		sel_frames=NAC.loc[(NAC[NAC.columns] > nac_low).any(axis='columns') & (NAC[NAC.columns] <= nac_high).any(axis='columns')]
		frames=np.array(list(sel_frames.index.values))
		
		filename='{}structures/NAC_frames-{}-{}-{}-{}-{}'.format(results_dir, prot, mol, c, co, i)
		
		if len(frames) == 0:
			print("\tNo frames found in replicate {} for cutoff={} +/- 0.05 Angstrom".format(i, co))
		else:
			print("\t{} frame(s) found in replicate {} for cutoff={} +/- 0.05 Angstrom".format(len(frames), i, co))
			
			#Deprecated. This will generate large files which cannot be concatenated due to different number of waters.
			'''if noWater == True:
				selection=u.select_atoms('not resname SOL')
			else:
				selection=u.select_atoms('all')'''
			selection=u.select_atoms('not resname SOL')
			with mda.Writer(filename+'.xtc', selection) as W:
				for frame in frames:
					u.trajectory[frame]
					W.write(selection)
			with mda.Writer(filename+'.gro', selection) as W:
					u.trajectory[frames[0]]
					W.write(selection)

def calcDensity(cutoff, **kwargs):	
	"""Retrive the density in Molar of density_sel"""

	for k, v in kwargs.items():
		exec(k+'=v')

	for co in cutoff:
		filenames=str(glob.glob(results_dir + 'structures/NAC_frames-{}-{}-{}-{}-*.gro'.format(results_dir, prot, mol, c, co)))
		for f in filenames:
			xtc=f[:-4] + '.xtc'
			fit=f + '-fit.xtc'

			ref=mda.Universe(f)
			mobile=mda.Universe(f, xtc)
			alignment=align.AlignTraj(mobile, ref, filename=fit)				
			alignment.run()
		
			fit_u=mda.Universe(filename + '.gro', fit)
			density=density_from_Universe(fit_u, delta=0.5) #, atomselection=density_sel, update_selection=True)
			density.convert_density('Molar')
			density.export(filename + '-density_Molar.dx', type=double)
					

#===================================================================
#Functions to combine NAC data with another feature
#These are helper functions which load the data and turn into a dataframe for further manipulation


def nac_distances(**kwargs):
    """Function to obtain orientation of molecules (as given by difference between distances) for each NAC bin. 
    Takes as inputs the nac and distance .npy objects calculated in NAC_calculation() and distances_calculation_all(). 
    Needs to convert the arrays into dataframes. 
    NOTE: The combined dataframe of NAC and DIST_O "merge" can be built with less number of frames, e.g. skipping each 2,  with the ".iloc[::2]" option.
    This may be required due to too much data being passed from the replicate_iterator()."""

    for k, v in kwargs.items():
        exec(k+'=v')

	
    filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sels), mol, c, start*ps, stop, dt, i)
    NAC_array=np.load(replicate + 'results/' + filename_NAC)
	
    
    filename_DIST_O='DIST_O-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(mol, c, start*ps, stop, dt, i)


    #Make the calculation of the difference between distB and distA
    if not os.path.exists(replicate + 'results/' +  filename_DIST_O):
        dist_A_file='distance1-i{}-o{}-s{}-{}.npy'.format(start*ps, stop, dt, i)
        dist_B_file='distance2-i{}-o{}-s{}-{}.npy'.format(start*ps, stop, dt, i)

        if os.path.exists(replicate + 'results/' + dist_A_file) or os.path.exists(replicate + 'results/' + dist_B_file): 
            dist_A=np.load(replicate + 'results/' + dist_A_file)
            dist_B=np.load(replicate + 'results/' + dist_B_file)
        else:
            distance_df=distance_calculation_all(**kwargs) 
            dist_A, dist_B=distance_df.iloc[:,0], distance_df.iloc[:,1]
        DIST_O_array= dist_B.ravel() - dist_A.ravel()
        np.save(replicate + 'results/' + filename_DIST_O, DIST_O_array)

    else:
        DIST_O_array=np.load(replicate + 'results/' + filename_DIST_O)
    print(np.shape(DIST_O_array) )
    nac_array_df=pd.DataFrame(np.round(NAC_array.ravel(), decimals=1), columns=['nac'])
    distO_array_df=pd.DataFrame(np.round(DIST_O_array.ravel(), decimals=1), columns=['distance'])
    
    try:
        nac_array_df.equals(distO_array_df)
    except ValueError as length_error:
        print("NAC and DIST_O do not have the same length", length_error)
	
    merge=pd.concat([nac_array_df, distO_array_df], axis=1).iloc[::2]
    #with h5py.File('{}/distance_dist_array_hdf-{}.hdf5'.format(temp_dir, i), 'w') as filename:
     
    #save the pickle in the results_dir. main.py contains the option to remove it afterwards
    with open('{}/distance_dist_array_pickle-{}'.format(temp_dir, i), 'wb') as filename:
        pickle.dump(merge, filename)
    
    return filename

def nac_angles(**kwargs):
    """Function to obtain angles for each NAC bin.
    Takes as inputs the nac and angle .npy objects calculated in NAC_calculation() and Angle_calculation().
    Needs to convert the arrays into dataframes.
    NOTE: The combined dataframe of NAC and ANGLE "merge" can be built with less number of frames, e.g. skipping each 2,  with the ".iloc[::2]" option.
    This may be required due to too much data being passed from the replicate_iterator().
    The current solution is to save "merge" as a pickle."""

    for k, v in kwargs.items():
        exec(k+'=v')


    filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sels), mol, c, start*ps, stop, dt, i)
    NAC_array=np.load(replicate + 'results/' + filename_NAC)

    filename_ANGLE='ANGLE{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(angles), mol, c, start*ps, stop, dt, i)

    #Make the calculation of angle if not found. Otherwise load
    if not os.path.exists(replicate + 'results/' +  filename_ANGLE):
        ANGLE_array=angle_calculation(**kwargs)
    else:
        ANGLE_array=np.load(replicate + 'results/' + filename_ANGLE)

    nac_array_df=pd.DataFrame(np.round(NAC_array.ravel(), decimals=1), columns=['nac'])
    angle_array_df=pd.DataFrame(np.round(ANGLE_array.ravel(), decimals=1), columns=['angle'])
    try:
        nac_array_df.equals(angle_array_df)
    except ValueError as length_error:
        print("NAC and ANGLE do not have the same length", length_error)

    merge=pd.concat([nac_array_df, angle_array_df], axis=1) #.iloc[::2]
    
    #save the pickle in the results_dir. main.py contains the option to remove it afterwards
    with open('{}/angles_dist_array_pickle-{}'.format(temp_dir, i), 'wb') as filename:
        pickle.dump(merge, filename)

    #Legacy: Obtain distributions for individual replicates. Choose between quartile or mean description
    '''
    #Quartile
    dist_angle=merge['angle'].groupby(pd.cut(merge['nac'], ranges)).quantile([0.25, 0.5, 0.75])
    #Mean
    dist_angle=merge['angle'].groupby(pd.cut(merge['nac'], ranges)).agg(['mean','std'])

    filename2=open('{}/angles_dist_rep_pickle-{}'.format(results_dir, i), 'wb')
    pickle.dump(dist_angle, filename2) #save the pickle in the results_dir. main.py contains the option to remove it afterwards
    filename2.close()
    '''

    return filename


def nac_Nmol(**kwargs):
    """Calculates the number of molecules within each bin of Nmol_range of NAC values"""

    for k, v in kwargs.items():
        exec(k+'=v')


    #rebuild NAC
    filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sels), mol, c, start*ps, stop, dt, i)

    NAC_array=np.load(replicate + 'results/' + filename_NAC) 
    length=NAC_array.shape[0]

    #Make calculations
    Nmol_range_index=Nmol_range + (Nmol_integral/2) #Center value of bins
    index=pd.MultiIndex.from_product([[c], Nmol_range_index], names=['[{}]'.format(mol), 'NAC'])
    nmol_df=pd.DataFrame(columns=['nmol', 'total'], index=index)

    for lower, upper, value in zip(Nmol_range, Nmol_range[1:], Nmol_range_index):
        nmol_df.xs(c).at[value ,'nmol']= ((NAC_array > lower) & (NAC_array <= upper)).sum()
    
    nmol_df['total']= NAC_array.shape[0]
    nmol_df.dropna(inplace=True)
     
    return nmol_df

#===============================================================================


def NAC_normalization(nac_hist, c, length, mol, NAC_bulkRange, resolution):
    """Function to normalize NAC values according to theoretic distribution in a shpere with bulk concentration "c". 
    Takes as inputs "c", average NAC histogram with standard errors and number of frames where NAC was calculated "length".
    Calculations are made with the  N_theo() function. 
    The probability of NAC for each bin is P_nac and P_nac_err. The theoretical distribution of NAC at "c" is P_t.
    The value of bulk "c" is adjusted by fitting the P_nac to the bulk region "ranges_bulk" using curve_fit(). 
    P_t_optimized uses the "c_opt" bulk concentration.
    dG is calculated from P_t_optimized.
    NOTE: The "ranges_bulk" should be carefully chosen for each system."""
	
    ranges=nac_hist.index.values
    print(nac_hist)
    #Find the element of nac_hist that is closest to the min and max of NAC_bulkRange
    bulk_min, bulk_max=ranges[np.abs(ranges - NAC_bulkRange[0]).argmin()], ranges[np.abs(ranges - NAC_bulkRange[-1]).argmin()]


    length=length.loc['traj_length'].mean(axis=0)*length.loc['ref'].mean(axis=0)

    try:
        bulk_value=float(str(c).split('M')[0]) #get the value (in M) of the concentration from the label of c.
        factor=0.0001
        unit='M'
    except:
        bulk_value=float(str(c).split('mM')[0])
        factor=0.0000001
        unit='mM'
	
    #Calculates the theoretical NAC values in "bulk" in a spherical shell of size given by "ranges". 
    #The output is N - 1 compared to N values in ranges, so an additional value of "resolution" is added for equivalence.
    N_ref= lambda ranges, bulk_value: np.log(((4.0/3.0)*np.pi)*np.diff(np.power(np.append(ranges, ranges[-1] + resolution), 3.0))*6.022*bulk_value*factor)
    

    P_nac=np.log(nac_hist[c]) - np.log(length)
    P_nac_err=np.log(nac_hist[c] - nac_hist['sem']) - np.log(length)
  
    #Region of P_nac values to be used for fitting. Based on defined values for bulk 
    y_fit=P_nac.loc[bulk_min:bulk_max].values


    '''v2
    P_nac=np.log(nac_hist_quartile['Q2']) - np.log(length)
    P_nac_err1=np.log(nac_hist_quartile[nac_hist_quartile[c] - nac_hist_quartile['Q1'] - np.log(length)
    P_nac_err2=np.log(nac_hist_quartile['Q1'] nac_hist_quartile[nac_hist_quartile[c]) - np.log(length)
    '''


    P_t=N_ref(ranges=ranges, bulk_value=bulk_value)   #original theoretical distribution with predefined bulk
    #Optimize the value of bulk with curve_fit of nac vs N_ref. 
    #Initial guess is "bulk" value. Bounds can be set as  +/- 50% of bulk value. 
    c_opt, c_cov = curve_fit(N_ref, NAC_bulkRange, y_fit, p0= bulk_value) #bounds=(bulk - bulk*0.5, bulk + bulk*0.5))


    stdev=np.sqrt(np.diag(c_cov))[0]
    c_bulk={c_opt[0]: stdev}
	
    #Recalculate N_ref with adjusted bulk concentration.
    P_opt=N_ref(ranges=ranges, bulk_value=c_opt)

    #dG(kT) calculation
    dG=P_opt - P_nac
    dG_err=dG-(P_opt - P_nac_err)

    #Replace NaN in errors for arbitrary value of 30.
    dG_err.fillna(30, inplace=True)
	
    theoretic_df=pd.DataFrame({r'$c_{}$={}'.format('{initial}', c):P_t, r'$[bulk]= {}\pm${}{}'.format(np.round(c_opt[0], decimals=3), np.round(stdev, decimals=3), unit):P_opt}, index=dG.index.values)
    dG=pd.concat([dG, dG_err.abs()], axis=1)
    dG.rename(columns={c:r'[{}] = {} {}'.format(mol, np.round(c_opt[0], decimals=1), unit), 0:'err-{}'.format(c)}, inplace=True)
    dG=pd.concat([dG, P_nac, theoretic_df], axis=1)
    dG.rename(columns={c:r'$P_{}$ {}'.format('{enzyme}', c)}, inplace=True)
    
    return dG, c_bulk

def NAC_calculation(sels, **kwargs):
    """"Calculates distances between pairs of atom groups (sel) for each set of selections "sels" using MDAnalysis D.distance_array method.
    Calculates the histogram (0, 140, 0.5 Angstrom) of NAC values and outputs it as a dataframe (NAC) with bins defined in 'ranges' and label 'i'.
    Stores the distances and NAC objects as .npy for each combination of tstart, tstop and dt"""

    for k, v in kwargs.items():
        exec(k+'=v')

    filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sels), mol, c, start*ps, stop, dt, i)
    if not os.path.exists(replicate + 'results/' +  filename_NAC):

        print("\tNAC file of replicate {} [{} - {}] ps not found.".format(i, start*ps, stop))
        print("\tReading trajectory of replicate {}".format(i))
        u=mda.Universe(replicate + TPR, replicate + XTC)


        distances=[] #List of distances arrays to be populated
        for d, sel in enumerate(sels, 1): #iterate through the list of sels. For each sel, define atomgroup1 (sel1) and atomgroup2 (sel2)
            sel1, sel2 =u.select_atoms(sel[0]), u.select_atoms(sel[1])
            print(sel1)
            #print(sel2)
            #print(d, sel1, sel2)
            filename='distance{}-i{}-o{}-s{}.npy'.format(d, start*ps, stop, dt)
            if not os.path.exists(replicate + 'results/' + filename):
                print("\tDistance {} file of replicate {} not found.".format(d, i))
                dist_d=np.around([D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory[start:stop:dt]], decimals=3) 
                np.save(replicate + 'results/' + filename, dist_d)
            else:
                dist_d=np.load(replicate + 'results/' + filename)
                print("\tDistance {} file of replicate {} found. Shape: {}".format(d, i, np.shape(dist_d)))
            distances.append(dist_d)
            
	    #NAC calculation. 
        print('\tCalculating NAC of replicate {} from distance pairs'.format(i))
        nac=np.around(np.sqrt(np.power(np.asarray(distances), 2).sum(axis=0)/len(distances)), 3) #SQRT(SUM(di^2)/#i)   
    
        np.save(replicate + 'results/' + filename_NAC, nac)
    else:
        print("\tNAC file of replicate {} [{} - {}] ps found.".format(i, start*ps, stop))
        nac=np.load(replicate + 'results/' + filename_NAC)

    shape=np.asarray(np.shape(nac))
    print("\tNAC from replicate {}: length {}, #sel1 {}, #sel2 {} ".format(i, shape[0], shape[1], shape[2]))
    index=['traj_length', 'ref', 'sel']
    length=pd.DataFrame(data=shape, index=index, columns=[i])
    hist, ranges=np.histogram(nac, bins=NAC_range)
    NAC=pd.DataFrame(zip(hist, NAC_midRange), columns=[i, 'bin'])
    NAC.set_index('bin', inplace=True)


    with open('{}/nac_rep_pickle-{}'.format(temp_dir, i), 'wb') as filename:
        pickle.dump(NAC, filename)
       
    return filename, length


def NAC_calculation_water(sels, **kwargs):
    """"Time of Flight method for NAC, kNAC and Nmol calculations. Useful for large number of distance pairs, where 
    storing the NAC is not viable. For each frame calculates the NAC value,
    discretizes the values into states to assign a state to the frame and updates de NAChistogram. 
    Nmol is calculated differently from normal function - integrals are taken from the histogram and not from the NAC object"""

    for k, v in kwargs.items():
        exec(k+'=v')

    filename_NAC='{}/NAC{}-water_{}-{}-i{}-o{}-s{}-rep{}.csv'.format(results_dir_tof, len(sels), mol, c, start*ps, stop, dt, i)
    filename_Nmol='{}/Nmol-water_{}-{}-i{}-o{}-s{}-rep{}.csv'.format(results_dir_tof, mol, c, start*ps, stop, dt, i)
    filename_kNAC='{}/kNAC{}-water_{}-{}-i{}-o{}-s{}-rep{}.csv'.format(results_dir_tof, len(sels), mol, c, start*ps, stop, dt, i)
    
    #Nmol_df
    Nmol_range_index=Nmol_range + (Nmol_integral/2) #Center value of bins
    index=pd.MultiIndex.from_product([[c], Nmol_range_index], names=['[{}]'.format(mol), 'NAC'])
    nmol_df=pd.DataFrame(columns=['nmol', 'total'], index=index)
 

    #print(os.path.exists(filename_NAC))
    
    
    u=mda.Universe(replicate + TPR, replicate + XTC)

    dict_distances={}
    for d, sel in enumerate(sels, 1):
        dict_distances[d]=u.select_atoms(sel[0]), u.select_atoms(sel[1])
    #print(dict_distances)

    NAC_states=pd.DataFrame()
    nac_hist_trj=np.zeros_like(NAC_range[:-1])
    traj_length=0
    for ts in u.trajectory[start:stop:dt]:
        distances=[]
        t=time.time()
        
        for d, sel in dict_distances.items():
            sel1, sel2 =sel[0], sel[1]
            #print(d, sel1, sel2)
            dist_d=D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) 
            distances.append(dist_d)
        d_H=np.minimum(np.asarray(distances)[1], np.asarray(distances)[2])
        del distances[1:]
        distances.append(d_H)
        
        #Warning. Use only for water, when d1 is Oxigen, d2 and d3 are hydrogen and d4 is the min(d2,d3)
        nac=np.around(np.sqrt(np.power(np.asarray(distances), 2).sum(axis=0)/len(distances)), 3) #SQRT(SUM(di^2)/#i)
        #nac_hist, _=np.histogram(nac, bins=NAC_range) #use if not slow. chop last NAC_range[:-1] for nac_hist_trj
        nac_hist=fast_histogram.histogram1d(nac, bins=(len(NAC_range)-1), range=[NAC_range[0], NAC_range[-2]])
        nac_hist_trj += nac_hist
        NAC_states=pd.concat([NAC_states, state_mapper(nac, states, state_encodings)], axis=1)
        
        traj_length +=1
        if traj_length*dt % 10000 == 0:
            print(traj_length*dt)
    print('Finished ToF calculation for replicate {}'.format(i))
		
      
    shape=[len(NAC_states.columns), np.shape(nac)[0], np.shape(nac)[1]]
    index=['traj_length', 'ref', 'sel']
    length=pd.DataFrame(data=shape, index=index, columns=[i])
    NAC=pd.DataFrame(data=nac_hist_trj, index=NAC_midRange, columns=[i])
    NAC.index.name='bin'

    #Nmol Calculations

    for lower, upper, value in zip(Nmol_range, Nmol_range[1:], Nmol_range_index):
        lower_b = NAC.index <= lower
        upper_b = NAC.index <= upper
        nmol_df.xs(c).at[value ,'nmol']= int(NAC[upper_b].sum().values - NAC[lower_b].sum().values)
        nmol_df['total']= NAC.values.sum()

    nmol_df.dropna(inplace=True)
    NAC_states_flatten=pd.DataFrame(NAC_states.values.flatten())
    
    nmol_df.to_csv(filename_Nmol)
    NAC.to_csv(filename_NAC)
    NAC_states_flatten.to_csv(filename_kNAC)
        

    
    
    with open('{}/nac_rep_pickle-{}'.format(temp_dir, i), 'wb') as filename:
        pickle.dump(NAC, filename)
        
    with open('{}/nac_states_pickle-{}'.format(temp_dir, i), 'wb') as filename2:
        pickle.dump(NAC_states_flatten, filename2)    
     
        
    return filename, filename2, length, nmol_df



def histogram_nac(histogram_df, c):
    """Average and standard error of set of NAC histograms per bin. Takes as input a histogram_df, which contains the count value per NAC bin."""
    
    mean=histogram_df.mean(axis=1)
    sem=histogram_df.sem(axis=1)
    histogram=pd.concat({c:mean, 'sem':sem}, axis=1)

    return histogram

def histogram_nac_quartile(histogram_df, c, ranges):
	'''Median and quartiles of NAC values per bin ranges. Takes as input a histogram_df, which contains all NAC histograms from each replicate.'''
	histogram=histogram_df.quantile([0.25,0.5,0.75], axis=1).transpose()
	histogram.columns= ['Q1-'+c, 'Median-'+c, 'Q3-'+c]
	print(histogram)
	return histogram



def NAC_discretization_byFrame(**kwargs):
    """Discretizes the NAC values based on state discretization per frame (one frame: state x 1) 
    Not sensitive to which state a given molecule is at each frame. Produces equal-sized strings for all concentrations
    OR based on state discretization per #substrates (one frame = state x #substrates).
    OR based on combinatorial state discretization (one frame = state x 1), but with states^2
    Sensitive to which state a given molecule is at each frame.
    Requires the NAC array to be loaded from NAC_calculation().
    Values of state boundaries are given by states array."""

    for k, v in kwargs.items():
        exec(k+'=v') in locals()
    stop=kwargs['stop']
    

    filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sels), mol, c, start*ps, stop, dt, i)
    NAC_array=np.load(replicate + 'results/' + filename_NAC)
    shape=np.shape(NAC_array)

    print("\tDiscretizing NAC arrays of replicate {}. length {}, #ref {}, #sel {} ".format(i, shape[0], shape[1], shape[2]))

    NAC_reshape=NAC_array.reshape(shape[0], shape[1]*shape[2]) #Make a 2D array of time rows and ref*sel columns
    stop=start + shape[0]



    if shape[1] == 1: 
        NAC=pd.DataFrame(NAC_reshape)
        NAC.columns=NAC.columns + 1
        NAC.set_index(np.arange(start, stop, dt), inplace=True)

    else:
        NAC=pd.DataFrame()
        NAC_split=np.split(NAC_reshape, shape[1], axis=1)
        for ref in NAC_split:
            NAC_ref=pd.DataFrame(ref)
            NAC_ref.columns=NAC_ref.columns + 1
            NAC_ref.set_index(np.arange(start, stop, dt), inplace=True)
            NAC=pd.concat([NAC, NAC_ref], axis=0)


    #####NOTE CHANGE TO FLATTEN IF REF MORE THAN 1. PUT STATE_DF IN IF CONDITION

    state_df=state_mapper(NAC.values, states, state_encodings)
        
    with open('{}/nac_states_pickle-{}'.format(temp_dir, i), 'wb') as filename:
        pickle.dump(state_df, filename)

    return filename


	###

	###One state per frame. Transforms the data from 'one frame = state x #substrates' into 'one frame = state x 1' 
	#state_df=np.amin(state_map, axis=1)	

	#One state per molecule x frame
	#state_df=state_map.values.ravel()
#	print(state_df)

#Helper function to map NAC values into state indexes
def state_mapper(array, states, state_encodings): 
    
    state_map=np.digitize(array, states, right=True) #Discretize NAC values in state bins.
    #state_map=NAC.applymap(lambda x:np.digitize(x, states, right=True)) #Deprecated. Much slower.

    #Combinatorial encoding of states 
    def state_evaluator(x):
        for c,v in enumerate(state_encodings):
            if np.array_equal(v, x.values):
                return c


    state_comb=pd.DataFrame()
    for s in range(0, len(states)+1):
        state_comb[s]=(state_map == s).any(1).astype(int)
    state_df=state_comb.apply(state_evaluator, axis=1)
    return state_df


#===============================================================
#Functions to calculate properties. TODO: Consider calculating NAC distances with these methods instead of a dedicated function

def distance_calculation(dists, prot, mol, c, cutoff, results_dir):
	'''Calculates distances between pairs of atom groups (dist) for each set of selections "dists" using MDAnalysis D.distance_array method.
	Stores the distances in results_dir. Returns the distances as dataframes.
	Uses the NAC_frames files obtained in extract_frames(). 
	NOTE: Can only work for multiple trajectories if the number of atoms is the same (--noWater)'''


	distances_df=pd.DataFrame() #Dataframe that stores all the distances for all cuttoffs
	for co in cutoff:
		frames='NAC_frames-{}-{}-{}-{}'.format(prot, mol, c, co)
		GRO=sorted(glob.glob('{}structures/{}-*.gro'.format(results_dir, frames)))
		XTC=sorted(glob.glob('{}structures/{}-*.xtc'.format(results_dir, frames)))
		try: 
			u=mda.Universe(GRO[0], XTC)	
			print("Reading frames of {} at {} NAC cutoff {}".format(prot, c, co))
		except FileNotFoundError:
			print('No files found for cutoff {}. Check if file {} exist.'.format(co, frames))
		distances={} # This is the dictionary of d distances.
		for dist in dists: #iterate through the list of dists. For each dist, define atomgroup1 (sel1) and atomgroup2 (sel2)
			d=dists.index(dist) + 1
			sel1, sel2 =u.select_atoms(dist[0]), u.select_atoms(dist[1])
			filename='distance{}-{}-{}-{}-co{}.npy'.format(d, prot, mol, c, co)
			
			if not os.path.exists(results_dir + filename):
				print("\tDistance{} file of cutoff {} not found.".format(d, co))
				distances[d]=np.around([D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory], decimals=3)
				np.save(results_dir + filename, distances[d])
			else:
				distances[d]=np.load(results_dir + filename)
				print("\tDistance {} file of cutoff {} found. Loading...".format(d, co))
			print("\tShape of distance {}: {}".format(d, np.shape(distances[d])[0]))
			distances_df=pd.concat([distances_df, pd.DataFrame(distances[d].ravel(), columns=[r'd{}@{} {} $\AA$'.format(d,c,co)])], axis=1)
	distances_df.index.name='frame'
	
	#Optional. Store df as pickle
	#filename=open('{}/distances_pickle-{}'.format(results_dir, c), 'wb')
	#pickle.dump(distances_df, filename)	
	#filename.close()
	#return filename
			
	return distances_df

def distance_calculation_all(dists, **kwargs):
    """Calculates distances between pairs of atom groups (dist) for each set of selections "dists" using MDAnalysis D.distance_array method.
    Stores the distances in replicate as .npy. Returns the dataframe of distances or stores it as pickle."""
	

    for k, v in kwargs.items():
        exec(k+'=v')

    distances_df=pd.DataFrame() #Dataframe that stores all the distances for all cuttoffs
    distances={} # This is the dictionary of d distances.
    for d, dist in enumerate(dists, 1): #iterate through the list of dists. For each dist, define atomgroup1 (sel1) and atomgroup2 (sel2)
        u=mda.Universe(replicate + TPR, replicate + XTC)
        sel1, sel2 =u.select_atoms(dist[0]), u.select_atoms(dist[1])
        #print(sel1)
        #print(sel2)
        filename='distance{}-i{}-o{}-s{}-{}.npy'.format(d, start*ps, stop, dt, i)
        if not os.path.exists(replicate + 'results/' + filename):
            print("\tDistance{} file of replicate {} not found.".format(d, i))
            distances[d]=np.around([D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory[start:stop:dt]], decimals=3)
            np.save(replicate + 'results/' + filename, distances[d])
        else:
            distances[d]=np.load(replicate + 'results/' + filename)
            print("\tDistance {} file of replicate {} found. Shape: {}".format(d, i, np.shape(distances[d])[0]))
        distances_df=pd.concat([distances_df, pd.DataFrame(distances[d].ravel(), columns=[r'd{}@{}-{}'.format(d,c,i)])], axis=1)
        distances_df.index.name='frame'

    return distances_df

    '''
    #Optional. Store df as pickle for further analysis of distances
    filename=open('{}/distances_rep_pickle-{}'.format(temp_dir, i), 'wb')
    pickle.dump(distances_df, filename)	
    filename.close()
	
    return filename
    '''


def angle_calculation(**kwargs):
    """Calculates angles between atom groups (sel) for one selection (angles) using MDAnalysis "calc_angles" method.
    Calculates the histogram (0, 180, 1 degree) of angle values and outputs it as a dataframe (ANGLE) with bins defined in 'ranges' and label 'i'.
    Stores the ANGLE object as .npy for each combination of tstart, tstop and dt

    NOTE: Currently, the method works for only one angle selection (angles) so it does not iterate through the list of sels.
    For each sel, define atom1 (sel1) and atom2 (sel2) atom3 (sel3)"""


    for k, v in kwargs.items():
        exec(k+'=v') in locals()

    angles=kwargs['angles']

    calc_angles=mda.lib.distances.calc_angles


    #Core function to define angles. Sel1 is the reference in the protein, which can be > 1 if multiple sel1 occur (e.g. octamer)
    def ABC(sel1, sel2, sel3):
        angle_all=[]
        for j,v in enumerate(sel1):
            angle_j=[calc_angles(sel1.atoms.positions[j], sel2.atoms.positions[k], sel3.atoms.positions[l], box=u.dimensions) for k,l in zip(range(0, len(sel2.atoms.positions)), range(0, len(sel3.atoms.positions)))]
            angle_all.append(angle_j)

        return angle_all

    filename_ANGLE='ANGLE{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(angles), mol, c, start*ps, stop, dt, i)
    if not os.path.exists(replicate +  'results/' + filename_ANGLE):
        print("\tReading trajectory of replicate {}".format(i))
        u=mda.Universe(replicate + TPR, replicate + XTC)
       
        angles_={} # This is the dictionary of d angles.
        print("\tANGLE file of replicate {} [{} - {}] ps not found.".format(i, start*ps, stop))

        #for sel in angles:
        sel=angles[0]
        d=angles.index(sel) + 1
        sel1, sel2, sel3 =u.select_atoms(sel[0]), u.select_atoms(sel[1]), u.select_atoms(sel[2])
        print(sel1, sel2, sel3)
        angle=[ABC(sel1, sel2, sel3) for ts in u.trajectory[start:stop:dt]] #loop through t steps

        print(np.shape(angle))
        angle_array=np.rad2deg(angle)  # This is angle x t array stored as values of key d in angles
        np.save(replicate + 'results/' + filename_ANGLE, angle_array)
    else:
        print("\tANGLE file of replicate {} [{} - {}] ps found.".format(i, start*ps, stop))
        angle_array=np.load(replicate + 'results/' + filename_ANGLE)
    
    shape=np.shape(angle_array)
    print("\tANGLE from replicate {}: length {}, #sel1 {}, #sel2 {} ".format(i, shape[0], shape[1], shape[2]))
    

    #Legacy: Calculate histogram of angles for the replicate
    '''
    ranges_angles=np.arange(1,181,1)
    hist, ranges_angles=np.histogram(angles, bins=ranges_angles)
    angle_df=pd.DataFrame(zip(hist, ranges_angles), columns=[i, 'angle'])
    angle_df.set_index('angle', inplace=True)

    filename=open('{}/angles_rep_pickle-{}'.format(temp_dir, i), 'wb')
    pickle.dump(ANGLE, filename) #save the merge as pickle in the results_dir. main.py contains the option to remove it afterwards
    filename.close()
    '''

    return angle_array

#===========================================================================	




#===========================================================================
#Functions for Langmuir calculation
def Nmol_NAC(**kwargs):
    """Calculates the number of molecules within each bin of Nmol_range of NAC values"""

    for k, v in kwargs.items():
        exec(k+'=v')


    #rebuild NAC 
    filename_NAC='NAC{}-{}-{}-i{}-o{}-s{}-rep{}.npy'.format(len(sels), mol, c, start*ps, stop, dt, i)

    NAC_array=np.load(replicate + 'results/' + filename_NAC)
    length=NAC_array.shape[0]
		
    #Make calculations
    Nmol_range_index=Nmol_range + (Nmol_integral/2) #Center value of bins
    index=pd.MultiIndex.from_product([[c], Nmol_range_index], names=['[{}]'.format(mol), 'NAC'])
    nmol_df=pd.DataFrame(columns=['nmol', 'total'], index=index)

    for lower, upper, value in zip(Nmol_range, Nmol_range[1:], Nmol_range_index): 
        nmol_df.xs(c).at[value ,'nmol']= ((NAC_array > lower) & (NAC_array <= upper)).sum()
        nmol_df['total']= NAC_array.shape[0]

    nmol_df.dropna(inplace=True)
  
    return nmol_df


def Nmol_stats(Nmol_c, results_dir, prot, mol):
	"""Calculates the statistics of number of molecules for each radius and concentration
	Performs fitting of data to Langmuir model"""

	def Langmuir(c, Nmax, Kd):
		return Nmax*c / (Kd + c)

	nmol_list=Nmol_c.index.unique(level=1).tolist()
	index=pd.MultiIndex.from_product([nmol_list, ['average', 'sem']], names=['NAC', 'stats'])
	stats_df=pd.DataFrame(columns=index, index=Nmol_c.index.unique(level=0))



	for r, r_df in Nmol_c.groupby(level=1):
		mean=r_df['nmol'].mean(axis=1) / r_df['total'].mean(axis=1)
		stats_df.loc[:, (r,'average')]=mean.values 
		sem=r_df['nmol'].sem(axis=1) / r_df['total'].mean(axis=1)
		stats_df.loc[:, (r, 'sem')]=sem.values


	try: #Check if there is already a correction of bulk concentrations (NAC method)
		conc_table=pd.read_csv('{}nac_concentrations-{}-{}.csv'.format(results_dir, prot, mol), index_col=0)
	except: #If not, use the defaul -conc values
		print("No bulk concentration(s) from NAC normalization found. Using inputs -conc")
		conc_table=stats_df.index

	#Convert all concentration values to M (if mM)
	for c in conc_table.index:
		try:
			float(str(c).split('mM')[0])
			conc_table.loc[c,'c_opt']=conc_table.loc[c,'c_opt']/1000
			conc_table.loc[c,'c_std']=conc_table.loc[c,'c_std']/1000
		except:
			pass

	#Calculate Nmax and Kd for each nmol_range
	index_fit=pd.MultiIndex.from_product([['Nmax', 'Kd'], ['value', 'std']])	
	fit=pd.DataFrame(columns=index_fit, index=nmol_list)

	x_points=np.linspace(conc_table['c_opt'][0], conc_table['c_opt'][-1], 100)
	langmuir_df=pd.DataFrame(index=x_points, columns=nmol_list)

	for NAC in stats_df.columns.levels[0].values:
		#Optional: Use the standard deviation as sigma. For Calb-MeOH errors in Kd got worse. From aprox. 30% to >40%
		popt, pcov=curve_fit(Langmuir, conc_table['c_opt'].values[:len(stats_df)], stats_df[NAC, 'average'].values, p0=[stats_df.at[stats_df.index[-1], (NAC, 'average')], conc_table['c_opt'].values[1]]) #, sigma=stats_df[nmol_init, 'sem'].values, absolute_sigma=True)

		#Store the Nmax and Kd values with corresponding standard deviations if Kd stdev is < 0.5
		if np.sqrt(np.diag(pcov))[1] / popt[1] < 0.5:

			fit.loc[NAC, 'Nmax']=popt[0], np.sqrt(np.diag(pcov))[0]
			fit.loc[NAC, 'Kd']=popt[1], np.sqrt(np.diag(pcov))[1]

			#Generate Langmuir curve with fitted values
			langmuir_df[NAC]=Langmuir(x_points, popt[0], popt[1])

	#Append concentrations values in M to the final list
	stats_df['[{}] (M)'.format(mol)]=conc_table['c_opt'].values[:len(stats_df)]
	stats_df['+/- [{}] (M)'.format(mol)]=conc_table['c_std'].values[:len(stats_df)]

	return stats_df, fit, langmuir_df

