# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:24:37 2021

@author: hcarv
"""

try:
    #SFv2
    import tools
except:
    print('Could not load SFv2 modules')

#legacy
from simtk.openmm.app import *
import simtk.openmm as omm

used_units=('picoseconds', 'picosecond', 'nanoseconds', 'nanosecond',
            'kelvin', 
            'nanometers', 'angstroms',
            'molar', 
            'atmospheres',  
            'kilojoules_per_mole')

from simtk.unit import * 
from simtk.openmm import app
from simtk import unit
from sys import stdout
#import parmed as pmd
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as md
import os
import re
import numpy as np
from pdbfixer.pdbfixer import PDBFixer

# =============================================================================
# peptide = 'H3K36'
# sim_time = '100ns'
# 
# traj_folder = 'trajectories_SETD2_{}_{}'.format(peptide, sim_time)
# 
# 
# 
# =============================================================================




class Protocols:

    def __init__(self, workdir=os.getcwd()):
        """
        

        Parameters
        ----------
        workdir : TYPE, optional
            DESCRIPTION. The default is os.getcwd().

        Returns
        -------
        None.

        """


        self.workdir=os.path.abspath(workdir)


        # Simulation Options

        self.Simulate_Steps = 5e7  # 100ns

        self.SAM_restr_eq_Steps = 5e6          
        self.SAM_free_eq_Steps = 5e6   




    def pdb2omm(self, 
              input_pdbs=None,
              solvate=True,
              protonate=True,
              fix_pdb=True,
              inspect=False,
              extra_input_pdb=[],
              ff_files=[],
              extra_ff_files=[],
              extra_names=[],
              other_ff_instance=False,
              pH_protein = 7.0,
              protonation_dictionary={}):
        """
        Method to prepare an openMM system from PDB and XML/other force field definitions.
        Returns self, so that other methods can act on it.
        Requires input PDB file(s) handled by "input_pdbs". 
        Uses default AMBER force fields if none are provide by "ff_files".
        Includes to provided force fields (or defaults) additional XML/other definitions with "extra_ff_files".
        TODO: include "extra_input_pdb" methods to build boxes on the fly.
        

        Parameters
        ----------
        input_pdb : TYPE, optional
            DESCRIPTION. The default is None.
        solvate : TYPE, optional
            DESCRIPTION. The default is True.
        protonate : TYPE, optional
            DESCRIPTION. The default is True.
        fix_pdb : TYPE, optional
            DESCRIPTION. The default is True.
        extra_input_pdb : TYPE, optional
            DESCRIPTION. The default is [].
        ff_files : TYPE, optional
            DESCRIPTION. The default is [].
        extra_ff_files : TYPE, optional
            DESCRIPTION. The default is [].
        extra_names : TYPE, optional
            DESCRIPTION. The default is [].
        other_ff_instance : TYPE, optional
            DESCRIPTION. The default is False.
        pH_protein : TYPE, optional
            DESCRIPTION. The default is 7.0.

        Returns
        -------
        None.

        """

# =============================================================================
#         
#                       extra_input_pdb=[], #['SAM_H3K36.pdb', 'ZNB_H3K36.pdb']
#               ff_files=[], #['amber14-all.xml', 'amber14/tip4pew.xml', 'gaff.xml'],
#               extra_ff_files=[], #['SAM.xml', 'ZNB.xml']
#               extra_names=[], #['SAM', 'ZNB'],
#         
# =============================================================================
        tools.Functions.fileHandler(self.workdir)
       
        input_pdb=f'{self.workdir}/{input_pdbs}'
        
        if fix_pdb:
            
            pdb=PDBFixer(input_pdb)
            
            pdb.findMissingResidues()
            pdb.findMissingAtoms()
            pdb.addMissingAtoms()
        
        else:
            pdb = app.PDBFile(input_pdb)
        
        pre_system = app.Modeller(pdb.topology, pdb.positions)
    
        forcefield=self.setForceFields(ff_files=ff_files, 
                                         extra_ff_files=extra_ff_files,
                                         omm_ff=False)
    
        if protonate:
            pre_system.addHydrogens(forcefield, 
                             pH = pH_protein, 
                             variants = self.setProtonationState(pre_system.topology.chains(), 
                                                                 protonation_dict=protonation_dictionary) )

        
        # add ligand structures to the model
        for extra_pdb_file in extra_input_pdb:
            extra_pdb = app.PDBFile(extra_pdb_file)
            pre_system.add(extra_pdb.topology, extra_pdb.positions)


        #Call to static solvate
        if solvate:
            pre_system=self.solvate(pre_system, forcefield)
    
        #Create a openMM topology instance
        system = forcefield.createSystem(pre_system.topology, 
                                         nonbondedMethod=app.PME, 
                                         nonbondedCutoff=1.0*nanometers,
                                         ewaldErrorTolerance=0.0005, 
                                         constraints='HBonds', 
                                         rigidWater=True)
        
        #Update attributes
        self.input_pdb=input_pdb
        self.system=system
        self.topology=pre_system.topology
        self.positions=pre_system.positions
        
        
            
        #TODO: A lot. Link to Visualization
        self.system_pdb=self.writePDB(pre_system.topology, pre_system.positions, name='system')
        
        return self
    
    def setSimulations(self, 
                      dt = 0.002*picoseconds, 
                      temperature = 300*kelvin, 
                      friction = 1/picosecond,
                      pressure=1*atmospheres,
                      pH=7.0,
                      equilibrations=[{'ensemble': 'NPT', 
                                       'step': 1*nanoseconds, 
                                       'report': 1000, 
                                       'restrained_sets': {'selections': ['protein and backbone'], 
                                                           'forces': [100*kilojoules_per_mole/angstroms]}}]):
        """
        Function to setup simulation protocols.
        Returns a simulation object.
        Updates openMM instance with simulation and some self attributes to make simulation hyperparameters more acessible (HPCsim)

        Parameters
        ----------
        dt : TYPE, optional
            DESCRIPTION. The default is 0.002*picoseconds.
        temperature : TYPE, optional
            DESCRIPTION. The default is 300*kelvin.
        friction : TYPE, optional
            DESCRIPTION. The default is 1/picosecond.
        equilibrations : TYPE, optional
            DESCRIPTION. The default is [('NPT', 1*nanoseconds, 100)].
        pressure : TYPE, optional
            DESCRIPTION. The default is 1*atmospheres.
        pH : TYPE, optional
            DESCRIPTION. The default is 7.0.

        Returns
        -------
        simulation : TYPE
            DESCRIPTION.

        """
  

        self.dt=dt
        self.temperature=temperature
        self.friction=friction
        self.pressure=pressure
        self.pH=pH
        #TODO: make pressure, temperature protocol specific

        #TODO: Decide on whether hardware specs are stored under simulation or on the system. 
        #later is better for different machines (update function)

        platform = omm.Platform.getPlatformByName('CUDA')
        gpu_index = '0'
        platformProperties = {'Precision': 'single','DeviceIndex': gpu_index}
        
        #print(platform)
        
        #TODO: make it classmethod maybe
        #TODO: Set integrator types
        integrator = omm.LangevinIntegrator(temperature, 
                                            friction, 
                                            dt)
        
        simulation = app.Simulation(self.topology, 
                                    self.system, 
                                    integrator, 
                                    platform, 
                                    platformProperties)
        
        simulation.context.setPositions(self.positions)
        
        selection_reference_topology = md.Topology().from_openmm(self.topology)
       
        self.trj_write={}
        self.steps={}       
        self.trj_subset='all'
        self.trj_indices = selection_reference_topology.select(self.trj_subset)
        self.eq_protocols=[]
 
        #trajectory_out_atoms = 'protein or resname SAM or resname ZNB'
        
        for eq in equilibrations:
                        
            print('Setting equilibration protocol:')
            
            for k,v in eq.items():
                print(f'\t{k}: {v}')
                        
            if type(eq['step']) != int:
                
                print(f"Steps is not unitless: {eq['step']}")
                eq['step']=int(eq['step']/self.dt) 
                
                print(f"Converted to unitless using integration time of {self.dt}: {eq['step']}")
            
            self.eq_protocols.append(eq)
            
            #Legacy, passed as single global.
            #ensemble= steps, save_steps, restrained_sets=eq
            #self.steps[eq['ensemble']]=steps
            #self.trj_write[eq['ensemble']] = save_steps #10000
            #self.eq_protocols.append((, restrained_sets))
            
        self.simulation=simulation
        
        return simulation


    def run_energyMinimization(self,
                     *args):
        """
        

        Parameters
        ----------
        simulation : TYPE
            DESCRIPTION.
        topology : TYPE
            DESCRIPTION.

        Returns
        -------
        simulation : TYPE
            DESCRIPTION.

        """
       
        #if not provided, use stored from instance (default)
        #TODO: make it pretty.
        

        
        if not args:
            simulation=self.simulation
            topology=self.topology

        Ei=simulation.context.getState(getEnergy=True).getPotentialEnergy()

        print(f'Performing energy minimization: {Ei}')
        
        
        #TODO: pass Etol, iterations etc.
        simulation.minimizeEnergy()
        Eo=simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(f'System is now minimized: {Eo}')

        positions_new = simulation.context.getState(getPositions=True).getPositions()
        self.Emin=self.writePDB(topology, positions_new, name='Emin')
        
        self.positions=positions_new
        
        self.simulation=simulation

        return simulation


    def run_equilibrations(self, *args):
        
        
        protocols=self.eq_protocols
        
        for p in protocols:
            
            if p['ensemble'] == 'NPT':
                
                print(f"Run: {p['ensemble']}")
                run=self.equilibration_NPT(p)
            
            else:
                print('TODO')
                
                #TODO: expand.
        
        return run
                
        
        
    def equilibration_NPT(self, protocol):      
        

        if protocol['restrained_sets']:

            #call to setRestraints, returns updated system. Check protocol['restrained_sets'] definitions on setSimulation.            
            self.system=self.setRestraints(protocol['restrained_sets'])

        
        # add MC barostat for NPT     
        self.system.addForce(omm.MonteCarloBarostat(self.pressure, 
                                                    self.temperature, 
                                                    25)) 
        #TODO: force is hardcoded, make it go away. Check units throughout!
        
        
        self.simulation.context.setPositions(self.positions)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        
        # Define reporters
        self.simulation.reporters.append(app.StateDataReporter(f'{self.workdir}/report.csv', 
                                                          protocol['report'], 
                                                          step=True, 
                                                          potentialEnergy=True, 
                                                          temperature=True, 
                                                          progress=True, 
                                                          remainingTime=True, 
                                                          speed=True, 
                                                          totalSteps=protocol['step'], 
                                                          separator='\t'))
        #TODO: Decide on wether same extent as steps for reporter
        #TODO: Link to streamz
        
        self.simulation.reporters.append(HDF5Reporter(f'{self.workdir}/equilibration_NPT.h5', 
                                                 protocol['report'], 
                                                 atomSubset=self.trj_indices))
        
        trj_time=protocol['step']*self.dt
        
        
        #############
        ## WARNING ##
        #############
        print(f"Restrained NPT equilibration ({trj_time})...")
        
        self.simulation.step(protocol['step'])
        
        
        #state_npt_EQ = self.simulation.context.getState(getPositions=True, getVelocities=True)
        
        positions_new = self.simulation.context.getState(getPositions=True, getVelocities=True).getPositions()
        
        print('NPT equilibration finished.')
        
        self.EQ_NPT=self.writePDB(self.simulation.topology, positions_new, name='EQ_NPT')
        Eo=self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        
        print(f'System is now equilibrated (?): {Eo}')
        
        
        self.positions=positions_new
    
        self.simulation=simulation
        
# =============================================================================
#         
#         import subprocess
#         cmd="mdconvert equilibration_NPT.h5 -o equilibration_NPT.xtc"
#         
#         process=subprocess.Popen(cmd.split(), stdout=self.workdir, cwd=self.workdir)
#         
#         output, error = process.comunicate()
#         
#         print(output, error)
#     
# =============================================================================
    
        return self.simulation

    def setRestraints(self, restrained_sets):
                      
                      
                      #=['protein and backbone'],
                      #forces=[100*kilojoules_per_mole/angstroms]):
        """
        Set position restraints on atom set(s).

        Parameters
        ----------
        restrained_sets : list of str, optional
            Selection(s) (MDTraj). The default is ['protein and backbone'].
        forces : list of int, optional
            The force applied in kilojoules_per_mole/angstroms. The default is 100.

        Returns
        -------
        system : TYPE
            DESCRIPTION.

        """
        
        
        
        #('protein and name CA and not chainid 1', 'chainid 3 and resid 0')
        #trajectory_out_atoms = 'protein or resname SAM or resname ZNB'
        

        topology = md.Topology().from_openmm(self.topology)   
                     
        
        if len(restrained_sets['selections']) == len(restrained_sets['forces']):
        
            for restrained_set, force_value in zip(restrained_sets['selections'], restrained_sets['forces']):
        
                force = omm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
                force.addGlobalParameter("k", force_value*kilojoules_per_mole/angstroms**2)
                force.addPerParticleParameter("x0")
                force.addPerParticleParameter("y0")
                force.addPerParticleParameter("z0")
            
                for res_atom_index in topology.select(restrained_set):
                    
                    force.addParticle(int(res_atom_index), self.positions[int(res_atom_index)].value_in_unit(unit.nanometers))
            
            #TODO: Decide wether to spawn system reps. Streamlined now. Legacy removes later.    
            self.system.addForce(force)

            return self.system
        
        else:
            print('Inputs for restrained sets are not the same length.')
    
    
        #      for res_atom_index, force in zip(restrained_indices_list, forces_list):
                       
        # 
        # 
        # # Free Equilibration
        # # forces: 0->HarmonicBondForce, 1->HarmonicAngleForce, 2->PeriodicTorsionForce, 3->NonbondedForce, 4->CMMotionRemover, 5->CustomExternalForce, 6->CustomExternalForce, 7->MonteCarloBarostat
        # n_forces = len(system.getForces())
        # system.removeForce(n_forces-2)
        # print('force removed')
        # 
        # # optional ligand restraint to force slight conformational changes
        # if restrained_ligands:
        #     
        #     integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        #     simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
        #     simulation.context.setState(state_npt_EQ)
        #     simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_restr_eq_Steps, separator='\t'))
        #     simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'free_BB_restrained_SAM_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
        #     print('free BB NPT equilibration of protein with restrained SAM...')
        #     simulation.step(SAM_restr_eq_Steps)
        #     state_free_EQP = simulation.context.getState(getPositions=True, getVelocities=True)
        #     positions = state_free_EQP.getPositions()
        #     app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_BB_restrained_SAM_NPT_EQ.pdb', 'w'), keepIds=True)
        #     print('Successful free BB, SAM restrained equilibration!')
        #   
        #     # equilibration with free ligand   
        #     n_forces = len(system.getForces())
        #     system.removeForce(n_forces-2)
        #     integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        #     simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
        #     simulation.context.setState(state_free_EQP)
        #     simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_free_eq_Steps, separator='\t'))
        #     simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'SAM_free_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
        #     print('SAM free NPT equilibration...')
        #     simulation.step(SAM_free_eq_Steps)
        #     state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
        #     positions = state_free_EQ.getPositions()
        #     app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'SAM_free_NPT_EQ.pdb', 'w'), keepIds=True)
        #     print('Successful SAM free equilibration!')
        #     
        # else:
        #     
        #     # remove ligand restraints for free equilibration (remove the second last force object, as the last one was the barostat)
        #     n_forces = len(system.getForces())
        #     system.removeForce(n_forces-2)
        #     simulation.context.setState(state_npt_EQ)
        #     simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_free_eq_Steps, separator='\t'))
        #     simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT_free.h5', 10000, atomSubset=trajectory_out_indices))
        #     print('free NPT equilibration...')
        #     simulation.step(SAM_free_eq_Steps)
        #     state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
        #     positions = state_free_EQ.getPositions()
        #     app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_NPT_EQ.pdb', 'w'), keepIds=True)
        #     print('Successful free equilibration!')
        # 
        # number_replicates = 5
        # count = 0
        # while (count < number_replicates):
        #  count = count+1 
    

    def setForceFields(self,
                         ff_files=[], 
                         extra_ff_files=[],
                         omm_ff=None,
                         ff_path=None,
                         defaults=('amber14-all.xml', 'amber14/tip4pew.xml'), 
                         add_residue_file=None):

        

        #add_residue_file='add_residue_def.xml'
        
                        
        #TODO: make more checks             
        #definition of additional residues (for ligands or novel residues); bonds extracted from ligand xml files
    
        if ff_path == None:
            
            ff_path=self.workdir
    
        if len(extra_ff_files) > 0:
                           
            for ff in extra_ff_files:
                ff_files.append(ff)
            
            #TODO: probably remove this
            if add_residue_file != None:
                ff_files.append(add_residue_file)
            
   
        for idx, ff in enumerate(ff_files):
            
            print(f'Extra force field file {idx+1}: {ff_path}/{ff}')
            ff_files[idx]=os.path.abspath(f'{ff_path}/{ff}') 
 
        #Wrap up what was defined
        if len(ff_files) > 0:
            
            print(f'Using default force fields: {defaults}')
            ff_files=defaults   
            forcefield = app.ForceField(*defaults)
        
        elif omm_ff:
            print(f'Other openMM force field instance has been passed: {omm_ff}')
            pass
            #TODO: get this from somewhere else    
            #forcefield=self.pre-setup('InputsFromGroup2')
        
        else:
            forcefield = app.ForceField(*ff_files)
        
        
        return forcefield    
    
    
    
    def writePDB(self,
                 topology, 
                 positions,
                 name='test',
                 keepIds=True):
        """
        Generates the PDB of provided OpenMM system with given "name" on class's "workdir".
        

        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        positions : TYPE
            DESCRIPTION.
        name : TYPE, optional
            DESCRIPTION. The default is 'test'.

        Returns
        -------
        None.

        """
        
        
        #TODO: allow other extensions
        
        out_pdb=f'{self.workdir}/{name}.pdb'
        app.PDBFile.writeFile(topology, positions, open(out_pdb, 'w'))
            
        print(f'PDB file generated: {out_pdb}')
        
        return out_pdb



 

    @staticmethod
    def solvate(system, 
                forcefield,
                box_size=9.0,
                boxtype='cubic',
                padding=None,
                ionicStrength=0*molar):
        """
        
        Generation of box parameters and solvation.

        Parameters
        ----------
        system : TYPE
            DESCRIPTION.
        boxtype : str, optional
            Box type, either 'cubic' or 'rectangular'. The default is 'cubic'.
        box_padding : float, optional
            Box padding. The default is 1.0.

        Returns
        -------
        None.

        """

        # add box vectors and solvate
        #TODO: Allow other box definitions. Use method padding to get prot sized+padding distance
        system.addSolvent(forcefield, 
                                     model='tip4pew', 
                                     neutralize=True, 
                                     ionicStrength=ionicStrength, #0.1 M for SETD2
                                     padding=None, 
                                     boxSize=omm.Vec3(box_size, box_size, box_size))
            
        return system


        
    @staticmethod
    def setProtonationState(system, protonation_dict=()):
        """
    
        Method to get a dictionary of residue types:
            (chain_name, residue number): restype

        Residue definitions for titratable residues:
    
            Histidine: HIE, HID, HIP, HIN; 
            Glutamate: GLU, GLH; 
            Aspartate: ASP, ASH; 
            Lysine: LYN, LYS; 
            Cysteine: CYS, CYX
    
    
        Parameters
        ----------
    
        system: object
            Modeller instance
    

        Returns
        -------
    
        protonation_list: list
            list of modified residues, used as "variants" by addHydrogens (openMM).

        Example
        -------
    
        state Histidine: 
     
            protonation_dict = {('A',84): 'HIP', ('A',86): 'HID'}

        """
    
        #only for manual protonation
        #TODO: Outsource residue protonation states.
    
        if not bool(protonation_dict):
            
            print('No protonation dictionary provided. Using default.')
    
            protonation_dict = {('A',1499): 'CYX', 
                    ('A',1501): 'CYX', 
                    ('A',1516): 'CYX', 
                    ('A',1520): 'CYX', 
                    ('A',1529): 'CYX', 
                    ('A',1533):'CYX', 
                    ('A',1539):'CYX', 
                    ('A',1631):'CYX', 
                    ('A',1678):'CYX', 
                    ('A',1680):'CYX', 
                    ('A',1685):'CYX', 
                    ('B',36): 'LYN'} 

        protonation_list = []
        key_list=[]


        for chain in system:
        
            chain_id = chain.id
            protonations_in_chain_dict = {}
            
            for protonation_tuple in protonation_dict:
                
                if chain_id == protonation_tuple[0]:
				
                    residue_number = protonation_tuple[1]			
                    protonations_in_chain_dict[int(residue_number)] = protonation_dict[protonation_tuple]
                    key_list.append(int(residue_number))
        
            for residue in chain.residues():
                with open("log_protonation.txt", "a") as myfile:
                
                    residue_id = residue.id
                    myfile.write(residue_id)
                
                    if int(residue_id) in key_list:
                        myfile.write(': Protonated')
                        myfile.write(residue_id)
                        protonation_list.append(protonations_in_chain_dict[int(residue_id)])
			
                    else:
                        protonation_list.append(None)
                        myfile.write('-')

        return protonation_list



    @classmethod
    def box_padding(system, box_padding=1.0):
        """
        

        Parameters
        ----------
        system : TYPE
            DESCRIPTION.
        box_padding : TYPE, optional
            DESCRIPTION. The default is 1.0.

        Returns
        -------
        system_shifted : TYPE
            DESCRIPTION.
        d : TYPE
            DESCRIPTION.
        d_x : TYPE
            DESCRIPTION.

        """
        
        x_list, y_list, z_list = [], [], []

        # get atom indices for protein plus ligands
        for index in range(len(system.positions)):
            x_list.append(system.positions[index][0]._value)
            y_list.append(system.positions[index][1]._value)
            z_list.append(system.positions[index][2]._value)
        x_span = (max(x_list) - min(x_list))
        y_span = (max(y_list) - min(y_list))
        z_span = (max(z_list) - min(z_list))

        # build box and add solvent
        d =  max(x_span, y_span, z_span) + (2 * box_padding)

        d_x = x_span + (2 * box_padding)
        d_y = y_span + (2 * box_padding)
        d_z = z_span + (2 * box_padding)

        prot_x_mid = min(x_list) + (0.5 * x_span)
        prot_y_mid = min(y_list) + (0.5 * y_span)
        prot_z_mid = min(z_list) + (0.5 * z_span)

        box_x_mid = d_x * 0.5
        box_y_mid = d_y * 0.5
        box_z_mid = d_z * 0.5

        shift_x = box_x_mid - prot_x_mid
        shift_y = box_y_mid - prot_y_mid
        shift_z = box_z_mid - prot_z_mid
            
            
        system_shifted=app.Modeller(system.topology, system.positions)
        
        # shift coordinates to the middle of the box
        for index in range(len(system_shifted.positions)):
            
            system_shifted.positions[index] = (system_shifted.positions[index][0]._value + shift_x,
                                               system_shifted.positions[index][1]._value + shift_y,
                                               system_shifted.positions[index][2]._value + shift_z)*nanometers
            
        return system_shifted, d, d_x
    
    
    
    
# =============================================================================
# ###########################################
# #####################LEGACY################






# 
#  # Simulate
#  
#  print('Simulating...')
#  
#  # create new simulation object for production run with new integrator
#  integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
#  simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
#  simulation.context.setState(state_free_EQ)
#  simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=Simulate_Steps, separator='\t'))
#  simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'production_5v21_100ns{}.h5'.format(count), 10000, atomSubset=trajectory_out_indices))
#  print('production run of replicate {}...'.format(count))
#  simulation.step(Simulate_Steps)
#  state_production = simulation.context.getState(getPositions=True, getVelocities=True)
#  state_production = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
#  final_pos = state_production.getPositions()
#  app.PDBFile.writeFile(simulation.topology, final_pos, open(traj_folder + '/' + 'production_5v21_100ns{}.pdb'.format(count), 'w'), keepIds=True)
#  print('Successful production of replicate {}...'.format(count))
#  del(simulation)
# =============================================================================

# ============================================================================= DES GOLD
# 
# # Create OpenFF topology with 1 ethanol and 2 benzenes.
# ethanol = Molecule.from_smiles('CCO')
# benzene = Molecule.from_smiles('c1ccccc1')
# off_topology = Topology.from_molecules(molecules=[ethanol, benzene, benzene])
# 
# # Convert to OpenMM Topology.
# omm_topology = off_topology.to_openmm()
# 
# # Convert back to OpenFF Topology.
# off_topology_copy = Topology.from_openmm(omm_topology, unique_molecules=[ethanol, benzene])
# 
# 
# =============================================================================

