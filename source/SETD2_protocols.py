# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:24:37 2021

@author: hcarv
"""


#SFv2
import tools

#legacy
from simtk.openmm.app import *
import simtk.openmm as mm
from simtk.unit import picoseconds, kelvin, nanometers, molar
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

peptide = 'H3K36'
sim_time = '100ns'

traj_folder = 'trajectories_SETD2_{}_{}'.format(peptide, sim_time)







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

        # Integration Options

        self.dt = 0.002*picoseconds
        self.temperature = 300*kelvin
        self.friction = 1/picoseconds
        self.sim_ph = 7.0

        # Simulation Options

        self.Simulate_Steps = 5e7  # 100ns
        self.npt_eq_Steps = 5e6    # 10ns
        self.SAM_restr_eq_Steps = 5e6          
        self.SAM_free_eq_Steps = 5e6   


        #Hardware specs
        self.workdir=os.path.abspath(workdir)
        self.platform = Platform.getPlatformByName('CUDA')
        self.gpu_index = '0'
        self.platformProperties = {'Precision': 'single','DeviceIndex': self.gpu_index}

    def setSystem(self, 
              input_pdb=None,
              solvate=True,
              protonate=True,
              fix_pdb=True,
              extra_input_pdb=[], #['SAM_H3K36.pdb', 'ZNB_H3K36.pdb']
              ff_files=[], #['amber14-all.xml', 'amber14/tip4pew.xml', 'gaff.xml'],
              extra_ff_files=[], #['SAM.xml', 'ZNB.xml']
              extra_names=[], #['SAM', 'ZNB'],
              other_ff_instance=False):
        """
        

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
        #['SAM_H3K36.pdb', 'ZNB_H3K36.pdb']              ff_files : TYPE, optional
            DESCRIPTION. The default is [].
        #['amber14-all.xml', 'amber14/tip4pew.xml', 'gaff.xml'] : TYPE
            DESCRIPTION.
        extra_ff_files : TYPE, optional
            DESCRIPTION. The default is [].
        #['SAM.xml', 'ZNB.xml']              extra_names : TYPE, optional
            DESCRIPTION. The default is [].
        #['SAM', 'ZNB'] : TYPE
            DESCRIPTION.
        other_ff_instance : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        system : TYPE
            DESCRIPTION.

        """
        
        tools.Functions.fileHandler(self.workdir)
       
        input_pdb=f'{self.workdir}/{input_pdb}'
        
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
                             pH = self.sim_ph, 
                             variants = self.setProtonationState(pre_system.topology.chains(), 
                                                                 protonation_dict={('A',187): 'ASP', ('A',224): 'HID'}) )

        
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
        
        #Update attribute
        self.system=system
        self.topology=pre_system.topology
        self.positions=pre_system.positions
    
        
        return system
    
    def setSimulation(self):
        """
        

        Returns
        -------
        simulation : TYPE
            DESCRIPTION.

        """
        
        #TODO: make it classmethod maybe
        #TODO: Set integrator types
        integrator = mm.LangevinIntegrator(self.temperature, 
                                            self.friction, 
                                            self.dt)
        
        simulation = app.Simulation(self.topology, 
                                    self.system, 
                                    integrator, 
                                    self.platform, 
                                    self.platformProperties)
        
        simulation.context.setPositions(self.positions)
        
        return simulation


    def pre_setup(self, input_omm_ff='InputsFromGroup2'):
        """
        
        Do your thing here.
        
        #TODO: Spit out the topology, chains, segId of the sytem.
        
        """

        pass


    def minimization(self,
                     positions_simulation):
        """
        

        Parameters
        ----------
        positions_simulation : TYPE
            DESCRIPTION.

        Returns
        -------
        min_pos : TYPE
            DESCRIPTION.
        simulation : TYPE
            DESCRIPTION.

        """

        (positions, simulation)=positions_simulation        

        print('Performing energy minimization...')
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        
        app.PDBFile.writeFile(simulation.topology, 
                              positions, 
                              open(self.workdir+'preMin.pdb', 'w'), 
                              keepIds=True)
        
        simulation.minimizeEnergy()
        
        min_pos = simulation.context.getState(getPositions=True).getPositions()
        
        app.PDBFile.writeFile(simulation.topology, 
                              positions, 
                              open(self.workdir+'postMin.pdb', 'w'), 
                              keepIds=True)
        
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        print('System is now minimized')


        return (min_pos, simulation)

    def setForceFields(self,
                         ff_files=[], 
                         extra_ff_files=[],
                         omm_ff=None,
                         ff_path=None,
                         defaults=('amber14-all.xml', 'amber14/tip4pew.xml'), 
                         add_residue_file=None):
        """
        
        
        
        
        
        
        
        """
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
    
    



    @classmethod
    def setRestraints(positions_system,
                      restrained_sets=('protein and name CA and not chainid 1', 'chainid 3 and resid 0'),
                      forces=[100, 150],
                      restrained_set=True):
        """
        
        Set position restraints on atom set(s).

        Parameters
        ----------
        positions_system : tuple
            A tuple of positions and system object
        restrained_sets : list, optional
            Selection(s) (MDTraj). The default 'protein and name CA and not chainid 1' and 'chainid 3 and resid 0'.
            Backbone and Restrained peptide, cofactor, and metal-dummy atoms
        forces : list, optional
            The force applied in kilojoules_per_mole/angstroms. Has to be same size as the restrained_sets. The default is [100,150].    


        Returns
        -------
        System: object
            A modified version of the system with costumized forces.

        """
        
        #TODO: Fix difference of positions_system as inputs and only system as out.
        
        (positions, system)=positions_system
        selection_reference_topology = md.Topology().from_openmm(system.topology)   
                     
        
        restrained_indices_list=[]
        for restrained_set in restrained_sets:

            restrained_indices = selection_reference_topology.select(restrained_set)
            restrained_indices_list.append(restrained_indices)

        forces_list=[]
        for idx, f in enumerate(forces):
        
            force = mm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
            force.addGlobalParameter("k", f*unit.kilojoules_per_mole/unit.angstroms**2)
        
            force.addPerParticleParameter("x0")
            force.addPerParticleParameter("y0")
            force.addPerParticleParameter("z0")
            
            forces_list.append(force)

        #TODO: Fixed for the last (second) set on SETD2.
        chain=0
        
        if restrained_set:
            chain=-1
            
        for res_atom_index in restrained_indices_list[chain]:   
            forces_list[chain].addParticle(int(res_atom_index), positions[int(res_atom_index)].value_in_unit(unit.nanometers))    
        system.addForce(forces_list[chain])
    
        return system

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
                                     boxSize=mm.Vec3(box_size, box_size, box_size))
            
        return system


        
    @staticmethod
    def setProtonationState(system, 
                            protonation_dict={}):
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
        else:
            
            protonation_dict= {('A',187): 'ASP', 
                    ('A',224): 'HID'} 


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


#trajectory_out_atoms = 'protein or resname SAM or resname ZNB'
#trajectory_out_interval = 10000
#trajectory_out_indices = selection_reference_topology.select(trajectory_out_atoms)


# # NPT Equilibration
# 
# # add barostat for NPT
# system.addForce(mm.MonteCarloBarostat(1*unit.atmospheres, temperature*unit.kelvin, 25))
# simulation.context.setPositions(min_pos)
# simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)
# simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=npt_eq_Steps, separator='\t'))
# simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT.h5', 10000, atomSubset=trajectory_out_indices))
# print('restrained NPT equilibration...')
# simulation.step(npt_eq_Steps)
# state_npt_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
# positions = state_npt_EQ.getPositions()
# app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'post_NPT_EQ.pdb', 'w'), keepIds=True)
# print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
# print('Successful NPT equilibration!')
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






