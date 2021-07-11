# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:24:37 2021

@author: hcarv
"""
#SFv2
try:
    import tools
    from Decorators import MLflow
    
except:
    pass

#legacy
from simtk.openmm.app import *
import simtk.openmm as omm

used_units=('picoseconds', 'picosecond', 'nanoseconds', 'nanosecond', 'hour',
            'kelvin', 
            'nanometers', 'angstroms',
            'molar', 
            'atmospheres',  
            'kilojoules_per_mole')

from simtk.unit import * 
from simtk.openmm import app
from simtk import unit
from sys import stdout
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as md
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import datetime
from pdbfixer.pdbfixer import PDBFixer







class Protocols:

    def __init__(self, 
                 workdir,
                 project_dir,
                 def_input_ff='/inputs/forcefields',
                 def_input_struct='/inputs/structures',
                 overwrite_mode=False):
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
        self.project_dir=os.path.abspath(project_dir)
        
        print(f'Setting openMM simulation protocols in {self.workdir}')
        
        #TODO: make robust
        self.def_input_struct=os.path.abspath(def_input_struct)
        self.def_input_ff=os.path.abspath(def_input_ff)
        
        
        stored_omm=glob.glob(f'{self.workdir}/omm_*.pkl')
        
        if len(stored_omm) > 0:
            
            print(f'\tWarning! Found conserved openMM system(s). {stored_omm[0]}')
            
            idx=int(stored_omm[0].split('omm_')[1].split('.pkl')[0])
            
            if overwrite_mode:
                self.omm_current=(idx, f'{self.workdir}/omm_{idx}.pkl')
                self.omm_previous = (0, f'{self.workdir}/omm_{idx}.pkl')
            else:
                self.omm_current=(idx+1, f'{self.workdir}/omm_{idx+1}.pkl')
                self.omm_previous=(idx, f'{self.workdir}/omm_{idx}.pkl')
                
                print(f'\tOverwrite {overwrite_mode}: {self.omm_current[1]} ({self.omm_previous[0]})')
                
        
        else:
            
            self.omm_current=(1, f'{self.workdir}/omm_1.pkl')
            self.omm_previous=(0, f'{self.workdir}/omm_1.pkl')
            

        
        self.ow_mode=overwrite_mode
            
        #print(self.def_input_struct)
        #print(self.def_input_ff)

    def pdb2omm(self, 
              input_pdb=None,
              solvate=True,
              protonate=True,
              fix_pdb=True,
              insert_molecules=False,
              extra_input_pdb=[],
              ff_files=[],
              extra_ff_files=[],
              extra_names=[],
              other_ff_instance=False,
              pH_protein = 7.0,
              residue_variants={},
              other_omm=False,
              input_sdf_file=None,
              box_size=9.0,
              name='NoName',
              insert_smiles=None):
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
        
        from openmmforcefields.generators import SystemGenerator, GAFFTemplateGenerator
        from openff.toolkit.topology import Molecule
        
        
        self.structures={}
        
        out_pdb=f'{self.workdir}/system.pdb'
        
        if self.omm_previous[0] >= 1 :
            
            #print('\tWarning! Found conserved openMM system: ', self.omm_previous[1])
                                              
            pkl_file=open(self.omm_previous[1], 'rb')
            system = pickle.load(pkl_file)
            
            pdb = app.PDBFile(out_pdb)
            pre_system = app.Modeller(pdb.topology, pdb.positions)
            
            self.system=system
            self.structures['system']=out_pdb
            self.input_pdb=out_pdb
            self.structures['input_pdb']=out_pdb
            
            self.topology=pdb.topology
            self.positions=pdb.positions
            
            print(f"\tLoaded openMM system: \n\tFile: {self.structures['system']}, \n\tTopology: {self.topology}")
            
        elif self.omm_previous[0] == 0:
            print('\tGenerating new openMM system.')
        
        
            self.input_pdb=input_pdb
            self.structures['input_pdb']=input_pdb
            
            #Fix the input_pdb with PDBFixer
            if fix_pdb:
                
                pdb=PDBFixer(self.input_pdb)
                
                pdb.findMissingResidues()
                pdb.findMissingAtoms()
                pdb.addMissingAtoms()
            
            else:
                pdb = app.PDBFile(self.input_pdb)
            
            
            #Generate a Modeller instance of the fixed pdb
            #It will be used to populate system
            pre_system = app.Modeller(pdb.topology, pdb.positions)
        
        
            #Add ligand structures to the model with addExtraMolecules_PDB
            if len(extra_input_pdb) > 0:
    
                pre_system, self.extra_molecules=self.addExtraMolecules_PDB(pre_system, extra_input_pdb)
    
    
        
            #Create a ForceField instance with provided XMLs with setForceFields()
            forcefield, ff_paths=self.setForceFields(ff_files=ff_files)
        
            if insert_molecules:
                molecules =  Molecule.from_smiles(insert_smiles, allow_undefined_stereo=True)
                gaff = GAFFTemplateGenerator(molecules=molecules, forcefield='gaff-2.11')
                gaff.add_molecules(molecules)
                forcefield.registerTemplateGenerator(gaff.generator)
    
        
            #Call to setProtonationState()
            print('\tProtonating.')
            if protonate:
                
                if residue_variants:
                
                    pre_system.addHydrogens(forcefield, pH = pH_protein, 
                                            variants = self.setProtonationState(pre_system.topology.chains(), 
                                                                     protonation_dict=residue_variants))
    
                else:
                    pre_system.addHydrogens(forcefield, pH = pH_protein)
            
    
            #Call to solvate()
            #TODO: For empty box, add waters, remove then
            print('\tSolvating.')
            if solvate:
                pre_system=self.solvate(pre_system, forcefield, box_size=box_size)
        
            self.topology=pre_system.topology
            self.positions=pre_system.positions
        
            #Define system. Either by provided pre_system, or other_omm system instance.
            
            if other_omm:
                
                system = self.omm_system(input_sdf_file, 
                                                         pre_system,
                                                         forcefield,
                                                         self.def_input_struct,
                                                         ff_files=ff_paths, 
                                                         template_ff='gaff-2.11')
                            
                #forcefield not needed?? 
        
            else:
                
                #Create a openMM topology instance
                system = forcefield.createSystem(pre_system.topology, 
                                             nonbondedMethod=app.PME, 
                                             nonbondedCutoff=1.0*nanometers,
                                             ewaldErrorTolerance=0.0005, 
                                             constraints='HBonds', 
                                             rigidWater=True)
                
            #Update attributes
            self.system=system

            system_pkl=open(self.omm_current[1], 'wb')
            pickle.dump(system, system_pkl)
            system_pkl.close()
    
            #TODO: A lot. Link to Visualization
            self.structures['system']=self.writePDB(pre_system.topology, pre_system.positions, name='system')
        
        
            print(f"\tSystem is now converted to openMM type: \n\tFile: {self.structures['system']}, \n\tTopology: {self.topology}")
            

        return self
    
  


  

    def setSimulations(self, 
                       dt = 0.002*picoseconds, 
                       temperature = 300*kelvin, 
                       friction = 1/picosecond,
                       pressure=1*atmospheres,
                       pH=7.0,
                       gpu_index='0',
                       productions=[],
                       equilibrations=[]):
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

        #Call to sniffMachine
        self.platform, self.platformProperties = self.sniffMachine(gpu_index)
        
        print(f"Using platform {self.platform.getName()}, ID: {self.platformProperties['DeviceIndex']}")
        

             
        selection_reference_topology = md.Topology().from_openmm(self.topology)
      
        self.trj_write={}
        self.steps={}       
        self.trj_subset='not water'
        self.trj_indices = selection_reference_topology.select(self.trj_subset)
        self.minimization={}
        self.eq_protocols=[]
        self.productions=[]

        for eq in equilibrations:
            
            eq=self.convert_Unit_Step(eq, self.dt)
            self.eq_protocols.append(eq)

        for prod in productions:
        
            prod=self.convert_Unit_Step(prod, self.dt)
            self.productions.append(prod)
                

    #TODO: Get platform features from setSimulations, useful for multiprocessing.
    def runSimulations(self, 
                       run_Emin=True, 
                       run_equilibrations=True, 
                       run_productions=False, 
                       overwrite_=False,
                       resume_=True,
                       minimum_effort=False,
                       compute_time=0.05,
                       reportFactor=0.1):
        """
        

        Parameters
        ----------
        run_Emin : TYPE, optional
            DESCRIPTION. The default is True.
        run_equilibrations : TYPE, optional
            DESCRIPTION. The default is True.
        run_productions : TYPE, optional
            DESCRIPTION. The default is False.
        overwrite_ : TYPE, optional
            DESCRIPTION. The default is False.
        resume_ : TYPE, optional
            DESCRIPTION. The default is True.
        minimum_effort : TYPE, optional
            DESCRIPTION. The default is False.
        compute_time : TYPE, optional
            DESCRIPTION. The default is 0.05.
        reportFactor : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        simulations : TYPE
            DESCRIPTION.

        """
        
        simulations={}
        
        print(f'Run mode: \n\tOverwrite: {overwrite_}\n\tResume: {resume_}\n\tCompute time: {compute_time*60} min\n')
        
        if run_Emin:
            
            print("\nEnergy minimization")
            simulations['minimization']=self.simulationSpawner(self.minimization,
                                                               index=1,
                                                               kind='minimization',
                                                               label='Emin',
                                                               overwrite=overwrite_,
                                                               resume=resume_,
                                                               compute_time_=compute_time)
        
        if run_equilibrations:
        
            for idx, p in enumerate(self.eq_protocols, 1):
            
                print(f"\nEQ run {idx}: {p['ensemble']}")
                simulations[f"EQ-{p['ensemble']}"]=self.simulationSpawner(p, 
                                                                          index=idx, 
                                                                          kind='equilibration', 
                                                                          label=p['ensemble'],
                                                                          overwrite=overwrite_,
                                                                          resume=resume_,
                                                                          compute_time_=compute_time,
                                                                          reportFactor_=reportFactor,
                                                                          fixed_step=True)
        
        if run_productions:
                            
            for idx, p in enumerate(self.productions, 1):
                
                
                if not run_equilibrations:
                    idx=idx-1
                
                print(f"\nProduction run {idx}: {p['ensemble']}")
                simulations[f"MD-{p['ensemble']}"]=self.simulationSpawner(p, 
                                                                          index=idx, 
                                                                          kind='production', 
                                                                          label=p['ensemble'],
                                                                          overwrite=overwrite_,
                                                                          resume=resume_,
                                                                          compute_time_=compute_time,
                                                                          reportFactor_=reportFactor,
                                                                          fixed_step=minimum_effort)
                        
        return simulations


    def simulationSpawner(self,
                          protocol, 
                          label='NPT', 
                          kind='production', 
                          index=0,
                          overwrite=False,
                          resume=True,
                          compute_time_=0.05,
                          reportFactor_=0.1,
                          fixed_step=False):
        """
        

        Parameters
        ----------
        protocol : TYPE
            DESCRIPTION.
        label : TYPE, optional
            DESCRIPTION. The default is 'NPT'.
        kind : TYPE, optional
            DESCRIPTION. The default is 'production'.
        index : TYPE, optional
            DESCRIPTION. The default is 0.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        resume : TYPE, optional
            DESCRIPTION. The default is True.
        compute_time_ : TYPE, optional
            DESCRIPTION. The default is 0.05.
        fixed_step : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        simulation : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        
        
        #TODO: Set integrator types    
        integrator = omm.LangevinIntegrator(self.temperature, self.friction, self.dt)
        
        name=f'{kind}_{label}-{index}'
        trj_file=f'{self.workdir}/{name}' 
        
        reportTime=compute_time_ * reportFactor_
        
        #Initiate simulation from stored state.
        simulation_init = app.Simulation(self.topology, 
                                        self.system, 
                                        integrator, 
                                        self.platform, 
                                        self.platformProperties)
        simulation_init.context.setPositions(self.positions)
        state_init=simulation_init.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        energy_init = state_init.getPotentialEnergy()
        positions_init = state_init.getPositions()
        
        
        #Check for checkpoints and update simulation state
        simulation, status, last_step, checkpoints=self.setSimulationState(simulation_init,
                                                                           self.dt,
                                                                           trj_file,
                                                                           resume, 
                                                                           overwrite)
        
        
        
        #Deliver final state from loaded checkpoint    
        if status != 'complete':
            
            #Write initial 
            self.structures['f{name}_init']=self.writePDB(self.topology, positions_init, name=f'{name}_init')
            print(f'\tSystem initial potential energy: {energy_init}')
            
            #Run the Eminimization
            if kind == 'minimization':
               
                print('\tEnergy minimization')
                simulation.minimizeEnergy()
                simulation.saveState(checkpoints['final'])
                state=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
                self.structures[name]=self.writePDB(self.topology, state.getPositions(), name=name)
           
            #Run the equilibrations or productions  
            else:
                
                #Fetch number of steps
                steps_, append_, total_step=self.getSteps(protocol, simulation, last_step, self.dt)
 
                #Simulation has not started. Generate new.
                if steps_ == total_step and status == 'null':
                    
                    print(f'\tPopulating file(s) {trj_file}.*')
            
                
                    #If there are user defined costum forces
                    try:
                        if protocol['restrained_sets']:
                            self.system=self.setRestraints(protocol['restrained_sets'])
    
                    except KeyError:    
                        pass
        
                    #If ensemble is NPT, add Barostat.
                    #TODO: frequency is hardcoded to 25, make it go away. Check units throughout!
                    #TODO: Set other barostats.
                    if label == 'NPT':
                        print(f'\tAdding MC barostat for {label} ensemble')      
                        self.system.addForce(omm.MonteCarloBarostat(self.pressure, self.temperature, 25)) 
        
                    #TODO: Emulate temperature increase
                    #Set velocities if continuation run
                    if index == 1 and kind == 'equilibration':
                        print(f'\tFirst equilibration run. Assigning velocities to {self.temperature}')
                        simulation.context.setVelocitiesToTemperature(self.temperature)
                    
                    elif index == 0 and kind == 'production':
                        print(f'\tWarning! First production run with no previous equilibration. Assigning velocities to {self.temperature}')
                        simulation.context.setVelocitiesToTemperature(self.temperature)
            
                    #Set velocities if continuation run    
                    else:
                        print(f'\tWarning! Continuation run. Assigning velocities from stored state.')
                        simulation.context.setVelocities(self.velocities)
                    
                    
                
                simulation.reporters.append(HDF5Reporter(f'{trj_file}.h5', protocol['report'], atomSubset=self.trj_indices))
                simulation.reporters.append(DCDReporter(f'{trj_file}.dcd', protocol['report'], append=append_))
                simulation.reporters.append(app.StateDataReporter(f'{trj_file}.csv', protocol['report'], 
                                                              step=True, totalEnergy=True, temperature=True, density=True,
                                                              progress=True, remainingTime=True, speed=True, 
                                                              totalSteps=steps_, separator=','))
                #Execute the two types of simulation run
                if fixed_step:
                    
                    simulation.reporters.append(CheckpointReporter(f'{trj_file}.chk', protocol['report']))
                    print('\tSimulating...')
                    
                    try:
                        simulation.step(steps_)
                        simulation.saveState(checkpoints['final'])
                        last_step = simulation.currentStep
                    
                    except:
                        
                        last_step = simulation.currentStep
                        print(f'\t Warning! Could not run all steps: {last_step}/{total_step}')
                        simulation.saveState(checkpoints['timeout'][1])
                        simulation.saveCheckpoint(checkpoints['timeout'][0])
                
                else:
                    print(f'\tSimulating for {str(datetime.timedelta(hours=compute_time_))} ({str(datetime.timedelta(hours= reportTime))} checkpoints)...')
                    simulation.runForClockTime(compute_time_, 
                                               checkpointFile=checkpoints['checkpoint'][0],
                                               stateFile=checkpoints['checkpoint'][1], 
                                               checkpointInterval=reportTime)
                    simulation.saveState(checkpoints['timeout'][1])
                    simulation.saveCheckpoint(checkpoints['timeout'][0])

                    last_step = simulation.currentStep
                
                remaining_step_end = total_step - last_step
                state=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
                
                
                #Evaluate status of current execution
                if remaining_step_end > 0:
                        
                    print(f"\tWarning! Simulation terminated by timeout: {last_step}/{total_step} = {remaining_step_end} steps to complete.")
                    self.structures['f{name}_chk']=self.writePDB(self.topology, state.getPositions(), name=f'{name}_chk')
                    
                else:
                    
                    print(f"\tSimulation completed or extended: {last_step}/{total_step} (+{np.negative(remaining_step_end)}) steps.")
                    simulation.saveState(checkpoints['final'])
                    self.structures[name]=self.writePDB(self.topology, state.getPositions(), name=name)
                    
                    #Remove forces is simulation is over.
                    if label == 'NPT':
                        self.system=self.forceHandler(self.system, kinds=['CustomExternalForce', 'MonteCarloBarostat'])
                    elif label == 'NVT':
                        self.system=self.forceHandler(self.system, kinds=['CustomExternalForce'])
        
        
        else:
            state=simulation_init.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        
        #Update attributes
        self.state=state
        self.positions = state.getPositions()
        self.velocities = state.getVelocities()
        self.energy=state.getPotentialEnergy()     
        self.simulation = simulation    
            
        print(f'\tSystem final potential energy: {self.energy}')

        return (simulation, self.state)
        
    @staticmethod
    def getSteps(protocol, simulation, last_step, dt):
    
        total_step = protocol['step']
                
        if last_step < total_step:
            
            steps_= total_step - last_step
            if last_step == 0:
                append_=False
            else:
                append_=True
                    
        elif last_step >= total_step:
                    
            steps_=0
            append_=False
                    
        trj_time=steps_*dt
        total_trj_time=total_step*dt
        print(f"\tSimulation time: {trj_time} out of {total_trj_time} = {steps_}/{total_step} steps")
            
            
        return steps_, append_, total_step
                


    @staticmethod
    def setSimulationState(simulation_i, dt, trj_file, resume, overwrite):

        from xml.etree import cElementTree as ET 
                
        checkpoints={'final' : (f'{trj_file}_final.xml'),
                     'checkpoint' : (f'{trj_file}.chk', f'{trj_file}.xml'),
                     'timeout' : (f'{trj_file}_time.chk', f'{trj_file}_time.xml')}
        
        def chkResume(simulation_i, chk):
            try:
                simulation_i.loadCheckpoint(chk[0])
            
            except:
                simulation_i.loadState(chk[1])
                
            root=ET.parse(chk[1]).getroot()
            last_ps = float(root.attrib['time'])
            dt_time=float(dt.__str__().split(' ')[0])    
            last_step=int(last_ps/ dt_time)
                
                            
                
            return simulation_i, last_step
        
        
        if resume:
            
            if os.path.exists(checkpoints['final']):
                simulation, last_step = simulation_i.loadState(checkpoints['final']), -1
                print('\tA complete checkpoint was found.')
                
                status = 'complete'
            
            else:
                
                if os.path.exists(checkpoints['timeout'][1]):
                    simulation, last_step=chkResume(simulation_i, checkpoints['timeout'])
                    print('\tA timeout checkpoint was found.')
                    status = 'clocked'
                
                elif os.path.exists(checkpoints['checkpoint'][1]):     
                    simulation, last_step=chkResume(simulation_i, checkpoints['checkpoint'])
                    print('\tA checkpoint file was found.')
                    status = 'incomplete'
                else:
                    print('\tNo checkpoint found.')
                    simulation ,last_step = simulation_i, 0
                    status = 'null'
        
        if overwrite:
            print('\tWarning! Overwrite mode set to True.')
            simulation, last_step = simulation_i, 0
            status = 'null'
            

            
        #state=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        
        return simulation, status, last_step, checkpoints
        

    #TODO: Merge with forceHandler.
    def setRestraints(self, restrained_sets):
                      
                      
                      #=['protein and backbone'],
                      #forces=[100*kilojoules_per_mole/angstroms]):
        """
        Set position restraints on atom set(s).

        Parameters
        ----------
        restrained_sets : dictionary
            Selection(s) (MDTraj) and forces (in kilojoules_per_mole/angstroms).

        Returns
        -------
        system : TYPE
            DESCRIPTION.

        """
        
        
        
        #('protein and name CA and not chainid 1', 'chainid 3 and resid 0')
        #trajectory_out_atoms = 'protein or resname SAM or resname ZNB'
        

        topology = md.Topology().from_openmm(self.topology)  
        
        equation="\t(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2"
        
        print('\tApplying custom external force: ', equation)
                     
        
        if len(restrained_sets['selections']) == len(restrained_sets['forces']):
        
            for restrained_set, force_value in zip(restrained_sets['selections'], restrained_sets['forces']):
        
                force = omm.CustomExternalForce(equation)
                force.addGlobalParameter("k", force_value*kilojoules_per_mole/angstroms**2)
                force.addPerParticleParameter("x0")
                force.addPerParticleParameter("y0")
                force.addPerParticleParameter("z0")
                
                print(f'\tRestraining set ({force_value*kilojoules_per_mole/angstroms**2}): {len(topology.select(restrained_set))} atoms')
            
                for res_atom_index in topology.select(restrained_set):
                    
                    force.addParticle(int(res_atom_index), self.positions[int(res_atom_index)].value_in_unit(unit.nanometers))
            
            #   TODO: Decide wether to spawn system reps. Streamlined now. Legacy removes later.    
                self.system.addForce(force)

            return self.system
        
        else:
            print('\tInputs for restrained sets are not the same length.')
    

    def setForceFields(self,
                         ff_files=[],
                         input_path=None,
                         defaults=[]):
        """
        Method to include force field files provided as [ff_files]. 
        If None is defined, it will include as default:'amber14/protein.ff14SB.xml', 'amber14/tip4pew.xml'.
        The "defaults" argument can be passed to override it.

        Parameters
        ----------
        ff_files : list, optional
            The force field files to include (under the default project *input path*). The default is empty.
        
        defaults : list, optional
            Default force field file(s) to use. The default is ['amber14/protein.ff14SB.xml', 'amber14/tip4pew.xml'].
        
        input_path : str, optional
            The input path of the force field files.

        Returns
        -------
        None.

        """
            
        ff_list=[]
        
        if len(ff_files) > 0:
                    
            for idx, ff in enumerate(ff_files, 1):
            
                path_ff=f'{self.def_input_ff}/{ff}'

                if not os.path.exists(path_ff):
                
                    print(f'\tFile {path_ff} not found!')
            
                else:
                    
                    print(f'\tAdding extra FF file {idx}: {path_ff}')
                    ff_list.append(path_ff)
        

        if len(defaults) == 0:      
                
            for d in defaults:
                ff_list.append(d)
            print('\tAdded defaults: ', defaults)
        
        
        forcefield = app.ForceField(*ff_list)
                
        return forcefield, ff_list  
    
    
    
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
        app.PDBFile.writeFile(topology, positions, open(out_pdb, 'w'), keepIds=keepIds)
            
        print(f'\tPDB file generated: {out_pdb}')
        
        return out_pdb


    def addExtraMolecules_PDB(self, system, extra_input_pdb, input_path=None):
        
        if input_path == None:
            
            input_path=self.def_input_struct
            
        total_mols=[]
        
        for idx, pdb_e in enumerate(extra_input_pdb, 1):
            
            
            path_pdb=f'{self.def_input_struct}/{pdb_e}'

            if not os.path.exists(path_pdb):
                
                print(f'\tFile {path_pdb} not found!')
            else:
                print(f'\tAdding extra PDB file {idx} to pre-system: {path_pdb}')
                extra_pdb = app.PDBFile(path_pdb)
                       
            try:
                system.add(extra_pdb.topology, extra_pdb.positions)
                print(f'\t{extra_pdb.topology}')
            
                total_mols.append((idx, path_pdb))  
            
            except:
                
                print(f'\tMolecule {idx} ({extra_pdb}) could not be added by openMM')

        for mol in total_mols:
            
            print(f'\tAdded {mol[0]}: {mol[1]}')  
  
        return system, total_mols

    

    @staticmethod
    def forceHandler(system, kinds=['CustomExternalForce']):
        
        for kind in kinds:
            
            forces=system.getForces()
            #print([f.__class__.__name__ for f in forces])
                
            to_remove=[]
        
            for idx, force in enumerate(forces):
            
                if force.__class__.__name__ == kind:
            
                    print(f'\tForce of kind {kind} ({idx}) will be removed.')                
                    to_remove.append(idx)
        
            for remove in reversed(to_remove):       
        
                system.removeForce(remove)
                
            #print([f.__class__.__name__ for f in system.getForces()])
                
        return system

    def omm_system(self, input_sdf,
                   input_pdb=None,
                   ff_files=['protein.ff14SB.xml', 'tip4pew.xml'],
                   template_ff='gaff-2.11'):
        """
        Method to include molecules from GAFF Generator methods.
        Supports inclusion with SDF files.
        TODO: Include SMILES.
        Takes as "input_system" a Modeller instance.

        Parameters
        ----------
        input_sdf : TYPE
            DESCRIPTION.
        input_system : TYPE
            DESCRIPTION.
        forcefield : TYPE
            DESCRIPTION.
        input_path : TYPE
            DESCRIPTION.
        ff_files : TYPE, optional
            DESCRIPTION. The default is [].
        template_ff : TYPE, optional
            DESCRIPTION. The default is 'gaff-2.11'.

        Returns
        -------
        system : TYPE
            DESCRIPTION.
        forcefield : TYPE
            DESCRIPTION.

        """
        

        
        from openmmforcefields.generators import SystemGenerator, GAFFTemplateGenerator
        from openff.toolkit.topology import Molecule
        
        
        pdb = app.PDBFile(input_pdb)
        
        ff_list=[]
        
        for idx, ff in enumerate(ff_files, 1):
            
                path_ff=f'{self.def_input_ff}/{ff}'

                if not os.path.exists(path_ff):
                
                    print(f'\tFile {path_ff} not found!')
            
                else:
                    
                    print(f'\tAdding extra FF file {idx}: {path_ff}')
                    ff_list.append(path_ff)
        
        
        # maybe possible to add same parameters that u give forcefield.createSystem() function
        forcefield_kwargs ={'constraints' : app.HBonds,
                            'rigidWater' : True,
                            'removeCMMotion' : False,
                            'hydrogenMass' : 4*amu }
		
        system_generator = SystemGenerator(forcefields=ff_list, 
                                           small_molecule_forcefield=template_ff,
                                           forcefield_kwargs=forcefield_kwargs, 
                                           cache='db.json')
        

        input_sdfs=[]
        for idx, sdf in enumerate(input_sdf, 1):
        
            
            path_sdf=f'{self.def_input_struct}/{sdf}'

            if not os.path.exists(path_sdf):
                
                print(f'\tFile {path_sdf} not found!')
            else:
                print(f'\tAdding extra SDF file {idx} to pre-system: {path_sdf}')
                input_sdfs.append(path_sdf)
                       
        molecules = Molecule.from_file()
# =============================================================================
#         
#         molecules = Molecule.from_file(*input_sdfs, file_format='sdf')
# 
#         
#         print(molecules)
#         
#         #pre_system = system_generator.create_system(topology=input_system.topology)#, molecules=molecules)
#         
#         
#         gaff = GAFFTemplateGenerator(molecules=molecules, forcefield=template_ff)
#         gaff.add_molecules(molecules)
# =============================================================================
        gaff.aff_molecules(pdb.topology)
        
        forcefield = ForceField('protein.ff14SB.xml', 'tip3pew.xml')
        
        forcefield.registerTemplateGenerator(gaff.generator)
        
        system = forcefield.createSystem(input_system.topology, 
                                         nonbondedMethod=app.PME, 
                                         nonbondedCutoff=1.0*nanometers,
                                         ewaldErrorTolerance=0.0005, 
                                         constraints='HBonds', 
                                         rigidWater=True)
        
        #forcefield.registerResidueTemplate(template)
        
        self.system=system
        
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
        forcefield : TYPE
            DESCRIPTION.
        box_size : TYPE, optional
            DESCRIPTION. The default is 9.0.
        boxtype : TYPE, optional
            DESCRIPTION. The default is 'cubic'.
        padding : TYPE, optional
            DESCRIPTION. The default is None.
        ionicStrength : TYPE, optional
            DESCRIPTION. The default is 0*molar.

        Returns
        -------
        system : TYPE
            DESCRIPTION.

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
    def sniffMachine(gpu_index='0'):
        """
        Method to get the fastest available platform. 
        Uses [openmmtools] to query platform properties.
        Sets the GPU index with "gpu_index".

        Parameters
        ----------
        gpu_index : TYPE
            DESCRIPTION.

        Returns
        -------
        platform : TYPE
            DESCRIPTION.
        platformProperties : TYPE
            DESCRIPTION.

        """
        
        import openmmtools
        
        #TODO: Use a test to check the number of available gpu (2 Pixar,3 Snake/Packman, 4 or 8 HPC) 
        
        #avail=openmmtools.utils.get_available_platforms()
        #print(avail)
        fastest=openmmtools.utils.get_fastest_platform()
        
        #platform=omm.Platform.getPlatformByName('CUDA')
        platform=fastest

        platformProperties = {'Precision': 'mixed',
                              'DeviceIndex': gpu_index}
        

        print(f'The fastest platform is {fastest.getName()}')
        print(f'Selected GPU ID: {gpu_index}')
        
        return platform, platformProperties
    
    
        
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

    @staticmethod
    def convert_Unit_Step(sim_dict, dt):
            
        
        convert=['step', 'report']
            
        for c in convert:
            
            value=sim_dict[c]
            
            if type(value) != int:
                
                #print(f'Value of {c} is not step: {value}')
                sim_dict[c]=int(value/dt) 
                
            #print(f"Converted to step using integration time of {dt}: {sim_dict[c]}")
                
        return sim_dict



    @classmethod
    def box_padding(cls, system, box_padding=1.0):
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
