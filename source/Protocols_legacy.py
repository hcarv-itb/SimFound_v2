# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:24:37 2021

@author: hcarv
"""

try:
    #SFv2
    import tools
    from Decorators import MLflow
    
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
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as md
import os
import re
import numpy as np
from pdbfixer.pdbfixer import PDBFixer







class Protocols:

    def __init__(self, 
                 workdir,
                 project_dir,
                 def_input_ff='/inputs/forcefields',
                 def_input_struct='/inputs/structures'):
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
        
        print(f'Setting openMM simulation protocols in {self.workdir}.')
        
        #TODO: make robust
        self.def_input_struct=os.path.abspath(def_input_struct)
        self.def_input_ff=os.path.abspath(def_input_ff)
        
        #print(self.def_input_struct)
        #print(self.def_input_ff)

    def pdb2omm(self, 
              input_pdb=None,
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
              residue_variants={},
              other_omm=False,
              input_sdf_file=None,
              box_size=9.0,
              name='NoName'):
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
        
        self.structures={}
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
    
        #Call to setProtonationState()
        if protonate:
            
            if residue_variants:
            
                pre_system.addHydrogens(forcefield, pH = pH_protein, 
                                        variants = self.setProtonationState(pre_system.topology.chains(), 
                                                                 protonation_dict=residue_variants))

            else:
                pre_system.addHydrogens(forcefield, pH = pH_protein)
        

        #Call to solvate()
        #TODO: For empty box, add waters, remove then
        if solvate:
            pre_system=self.solvate(pre_system, forcefield, box_size=box_size)
    
        self.topology=pre_system.topology
        self.positions=pre_system.positions
    
        #Define system. Either by provided pre_system, or other_omm system instance.
        
        if other_omm:
            
            system, forcefield_other=self.omm_system(input_sdf_file, 
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

        #TODO: A lot. Link to Visualization
        self.structures['system']=self.writePDB(pre_system.topology, pre_system.positions, name='system')
        
        
        print(f"System is now converted to openMM type: \n\tFile: {self.structures['system']}, \n\tTopology: {self.topology}")
        
        return self
    
  

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
        platform, platformProperties = self.sniffMachine(gpu_index)
        
        print(f"Using platform {platform.getName()}, ID: {platformProperties['DeviceIndex']}")
        
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
        self.trj_subset='not water'
        self.trj_indices = selection_reference_topology.select(self.trj_subset)
        self.eq_protocols=[]
        self.productions=[]
 
        #trajectory_out_atoms = 'protein or resname SAM or resname ZNB'

        for eq in equilibrations:
            
            eq=self.convert_Unit_Step(eq, self.dt)
            self.eq_protocols.append(eq)

        for prod in productions:
        
            prod=self.convert_Unit_Step(prod, self.dt)
            self.productions.append(prod)
                
        #TODO: Checkpoints for continuation. Crucial for production.
        #simulation.saveCheckpoint(os.path.abspath(f'{self.workidr}/checkpoint.chk'))
            

        self.simulation=simulation
        
        return simulation

    #TODO: Get platform features from setSimulations, useful for multiprocessing.
    def runSimulations(self, run_Emin=True, run_equilibrations=True, run_productions=False):
        
        if run_Emin:
            
            run=self.simulationSpawner(self.simulation, kind='minimization')
        
        if run_equilibrations:
        
            for idx, p in enumerate(self.eq_protocols, 1):
            
                print(f"EQ run {idx}: {p['ensemble']}")
                run=self.simulationSpawner(p, index=idx, kind='equilibration', label=p['ensemble'])
        
        if run_productions:
                            
            for idx, p in enumerate(self.productions, 1):
                
                print(f"Production run {idx}: {p['ensemble']}")
                run=self.simulationSpawner(p, index=idx, kind='production', label=p['ensemble'])
                        
        return run


    def simulationSpawner(self, protocol, label='NPT', kind='equilibration', index=0):
        
        
        self.energy=self.simulation.context.getState(getEnergy=True).getTotalEnergy()

        print(f'Spwaning new simulation:')
        print(f'\tSystem total energy: {self.energy}')
        
        
        if kind == 'minimization':
            
            self.simulation.minimizeEnergy()
            self.structures['minimization']=self.writePDB(self.topology, self.positions, name='minimization')
            print(f'\tPerforming energy minimization: {Ei}')
        
        else:
            
            name=f'{kind}_{label}'
            trj_file=f'{self.workdir}/{kind}_{label}'
        
            #If there are user defined costum forces
            try:
                if protocol['restrained_sets']:
                    self.system=self.setRestraints(protocol['restrained_sets'])

            except KeyError:    
                pass

            if label == 'NPT':
            
                #TODO: frequency is hardcoded to 25, make it go away. Check units throughout!
                print(f'\tAdding MC barostat for {name} ensemble')      
                self.system.addForce(omm.MonteCarloBarostat(self.pressure, self.temperature, 25)) 
        
            self.simulation.context.setPositions(self.positions)
        
            if index == 1 and kind == 'equilibration':
                print(f'\tFirst equilibration run {name}. Assigning velocities to {self.temperature}')
                self.simulation.context.setVelocitiesToTemperature(self.temperature)
        
            # Define reporters
            self.simulation.reporters.append(DCDReporter(f'{trj_file}.dcd', protocol['report']))           
            self.simulation.reporters.append(HDF5Reporter(f'{trj_file}.h5', protocol['report'], atomSubset=self.trj_indices))
            self.simulation.reporters.append(app.StateDataReporter(f'{trj_file}.csv', 
                                                          protocol['report'], 
                                                          step=True, 
                                                          totalEnergy=True, 
                                                          temperature=True,
                                                          density=True,
                                                          progress=True, 
                                                          remainingTime=True, 
                                                          speed=True, 
                                                          totalSteps=protocol['step'], 
                                                          separator='\t'))
        
            positions_first = self.simulation.context.getState(getPositions=True).getPositions()
        
            self.structures['f{name}_ref']=self.writePDB(self.simulation.topology, positions_first, name=f'{name}_0')
        
            #############
            ## WARNIN ##
            #############
        
            trj_time=protocol['step']*self.dt
            print(f'\t{kind} simulation in {label} ensemble ({trj_time})...')
            self.simulation.step(protocol['step'])
        
            
            #############
            ## WARNOUT ##
            #############

            print(f'\t{kind} simulation in {label} ensemble ({trj_time}) completed.')
            self.structures[f'{name}']=self.writePDB(self.simulation.topology, self.positions, name='{name}')

        state=self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        self.positions = state.getPositions()
        self.velocities = state.getVelocities()
        self.energy=state.getPotentialEnergy()


        

        
        #Remove Costum forces (always initialized in next run)
        if label == 'NPT':
            self.system=self.forceHandler(self.system, kinds=['CustomExternalForce', 'MonteCarloBarostat'])
        elif label == 'NVT':
            self.system=self.forceHandler(self.system, kinds=['CustomExternalForce'])
            
        return self.simulation


    def run_energyMinimization(self):
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
        

        Ei=self.simulation.context.getState(getEnergy=True).getPotentialEnergy()

        print(f'Performing energy minimization: {Ei}')
        
        
        #TODO: pass Etol, iterations etc.
        self.simulation.minimizeEnergy()
        Eo=self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(f'System is now minimized: {Eo}')

        self.positions = self.simulation.context.getState(getPositions=True).getPositions()        
        self.structures['minimization']=self.writePDB(self.topology, self.positions, name='minimization')        
        
        return self.simulation



                

    def production_NPT(self, protocol):      
        

        prod_trj=f'{self.workdir}/production_NPT'

        self.simulation.context.setPositions(self.positions)
        
        # =============================================================================
        # NO need for now since NPT eq protocols are not removing the barostat
        #         # add MC barostat for NPT     
        #         self.system.addForce(omm.MonteCarloBarostat(self.pressure, 
        #                                                     self.temperature, 
        #                                                     25)) 
        #         #TODO: force is hardcoded, make it go away. Check units throughout!
        # =============================================================================
                
                
        # Define reporters

        #TODO: Decide on wether same extent as steps for reporter
        #TODO: Link to mlflow or streamz
        
        
        #TODO: Use append on DCD for control of continuation
        self.simulation.reporters.append(DCDReporter(f'{prod_trj}.dcd', protocol['report']))           
        self.simulation.reporters.append(HDF5Reporter(f'{prod_trj}.h5', protocol['report'], atomSubset=self.trj_indices))
        self.simulation.reporters.append(app.StateDataReporter(f'{prod_trj}.csv', 
                                                          protocol['report'], 
                                                          step=True, 
                                                          totalEnergy=True, 
                                                          temperature=True,
                                                          density=True,
                                                          progress=True, 
                                                          remainingTime=True, 
                                                          speed=True, 
                                                          totalSteps=protocol['step'], 
                                                          separator='\t'))
        
        
        positions_first = self.simulation.context.getState(getPositions=True, getVelocities=True).getPositions()
        self.writePDB(self.simulation.topology, positions_first, name='production_NPT_0')
        
        #############
        ## WARNIN ##
        #############
        trj_time=protocol['step']*self.dt
        print(f"NPT production ({trj_time})...")
        
        self.simulation.step(protocol['step'])

        #############
        ## WARNOUT ##
        #############

        state=self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        self.positions = state.getPositions()
        self.velocities = state.getVelocities()
        self.energy=state.getPotentialEnergy()


        print('NPT production finished.')
        
        self.EQ_NPT_PDB=self.writePDB(self.simulation.topology, self.positions, name='production_NPT')
        Eo=self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        
        print(f'System energy: {Eo}')
        
            
        return self.simulation        

   
    def equilibration_NPT(self, protocol, index=0):      
            
        eq_trj=f'{self.workdir}/equilibration_NPT'

        if protocol['restrained_sets']:

            #call to setRestraints, returns updated system. Check protocol['restrained_sets'] definitions on setSimulation.            
            self.system=self.setRestraints(protocol['restrained_sets'])

        # add MC barostat for NPT     
        self.system.addForce(omm.MonteCarloBarostat(self.pressure, 
                                                    self.temperature, 
                                                    25)) 
        #TODO: force is hardcoded, make it go away. Check units throughout!
        
        
        self.simulation.context.setPositions(self.positions)
        
        if index == 0:
        
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
        
        # Define reporters
        self.simulation.reporters.append(DCDReporter(f'{eq_trj}.dcd', protocol['report']))           
        self.simulation.reporters.append(HDF5Reporter(f'{eq_trj}.h5', protocol['report'], atomSubset=self.trj_indices))
        self.simulation.reporters.append(app.StateDataReporter(f'{eq_trj}.csv', 
                                                          protocol['report'], 
                                                          step=True, 
                                                          totalEnergy=True, 
                                                          temperature=True,
                                                          density=True,
                                                          progress=True, 
                                                          remainingTime=True, 
                                                          speed=True, 
                                                          totalSteps=protocol['step'], 
                                                          separator='\t'))
        
        positions_first = self.simulation.context.getState(getPositions=True).getPositions()
        
        self.writePDB(self.simulation.topology, positions_first, name='equilibration_NPT_0')
        
        #############
        ## WARNIN ##
        #############
        
        trj_time=protocol['step']*self.dt
        print(f"Restrained NPT equilibration ({trj_time})...")
        
        self.simulation.step(protocol['step'])
        
        #############
        ## WARNOUT ##
        #############

        state=self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        self.positions = state.getPositions()
        self.velocities = state.getVelocities()
        self.energy=state.getPotentialEnergy()

        print('NPT equilibration finished.')
        
        self.EQ_NPT_PDB=self.writePDB(self.simulation.topology, self.positions, name='equilibration_NPT')
         
        self.system=self.forceHandler(self.system, kinds=['CustomExternalForce', 'MonteCarloBarostat'])         
            
        return self.simulation
 
    
 
 
    
 
    @staticmethod
    def forceHandler(system, kinds=['CustomExternalForce']):
        
        for kind in kinds:
            
            forces=system.getForces()
                
            to_remove=[]
        
            for idx, force in enumerate(forces):
            
                if force.__class__.__name__ == 'CustomExternalForce':
            
                    print(f'Force of kind {kind} ({idx}) will be removed.')                
                    to_remove.append(idx)
        
            for remove in reversed(to_remove):       
        
                system.removeForce(remove)
                
        return system
            
# =============================================================================
#             print('updated:')        
#             for force in forces:
#                 print(force.__class__.__name__)
# =============================================================================


    def equilibration_NVT(self, protocol, index=0):      
            
        eq_trj=f'{self.workdir}/equilibration_NVT'

        if protocol['restrained_sets']:

            #call to setRestraints, returns updated system. Check protocol['restrained_sets'] definitions on setSimulation.            
            self.system=self.setRestraints(protocol['restrained_sets'])

# =============================================================================    
#         self.system.addForce(omm.MonteCarloBarostat(self.pressure, 
#                                                     self.temperature, 
#                                                     25)) 
#         #TODO: force is hardcoded, make it go away. Check units throughout!
# =============================================================================
        
        
        self.simulation.context.setPositions(self.positions)
        
        if index == 0:
        
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
        
        # Define reporters
        self.simulation.reporters.append(DCDReporter(f'{eq_trj}.dcd', protocol['report']))           
        self.simulation.reporters.append(HDF5Reporter(f'{eq_trj}.h5', protocol['report'], atomSubset=self.trj_indices))
        self.simulation.reporters.append(app.StateDataReporter(f'{eq_trj}.csv', 
                                                          protocol['report'], 
                                                          step=True, 
                                                          totalEnergy=True, 
                                                          temperature=True,
                                                          density=True,
                                                          progress=True, 
                                                          remainingTime=True, 
                                                          speed=True, 
                                                          totalSteps=protocol['step'], 
                                                          separator='\t'))
        
        positions_first = self.simulation.context.getState(getPositions=True).getPositions()
        
        self.writePDB(self.simulation.topology, positions_first, name='equilibration_NVT_0')
        
        #############
        ## WARNIN ##
        #############
        
        trj_time=protocol['step']*self.dt
        print(f"Restrained NVT equilibration ({trj_time})...")
        
        self.simulation.step(protocol['step'])
        
        #############
        ## WARNOUT ##
        #############

        state=self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        self.positions = state.getPositions()
        self.velocities = state.getVelocities()
        self.energy=state.getPotentialEnergy()

        print('NVT equilibration finished.')
        
        self.EQ_NPT_PDB=self.writePDB(self.simulation.topology, self.positions, name='equilibration_NVT')

        self.system=self.forceHandler(self.system, kinds=['CustomExternalForce'])
        #Remove implemented forces (only costum)
        #TODO: remove MC for fresh delivery
            
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
        
        equation="(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2"
        
        print('Applying potential: ', equation)
                     
        
        if len(restrained_sets['selections']) == len(restrained_sets['forces']):
        
            for restrained_set, force_value in zip(restrained_sets['selections'], restrained_sets['forces']):
        
                force = omm.CustomExternalForce(equation)
                force.addGlobalParameter("k", force_value*kilojoules_per_mole/angstroms**2)
                force.addPerParticleParameter("x0")
                force.addPerParticleParameter("y0")
                force.addPerParticleParameter("z0")
                
                print(f'Restraining set ({force_value*kilojoules_per_mole/angstroms**2}): {len(topology.select(restrained_set))}')
            
                for res_atom_index in topology.select(restrained_set):
                    
                    force.addParticle(int(res_atom_index), self.positions[int(res_atom_index)].value_in_unit(unit.nanometers))
            
            #   TODO: Decide wether to spawn system reps. Streamlined now. Legacy removes later.    
                self.system.addForce(force)

            return self.system
        
        else:
            print('Inputs for restrained sets are not the same length.')
    
 
    
    

    def setForceFields(self,
                         ff_files=[],
                         input_path=None,
                         defaults=[']):

        

        #add_residue_file='add_residue_def.xml'
        
                        
        #TODO: make more checks             
        #definition of additional residues (for ligands or novel residues); bonds extracted from ligand xml files
    

            
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
            
            defaults=['amber14/protein.ff14SB.xml', 'amber14/tip4pew.xml']
        
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
        app.PDBFile.writeFile(topology, positions, open(out_pdb, 'w'))
            
        print(f'PDB file generated: {out_pdb}')
        
        return out_pdb


    @staticmethod
    def omm_system(input_sdf,
                   input_system,
                   forcefield,
                   input_path,
                   ff_files=[],
                   template_ff='gaff-2.11'):
        

        
        from openmmforcefields.generators import SystemGenerator, GAFFTemplateGenerator
        from openff.toolkit.topology import Molecule
        
        
        # maybe possible to add same parameters that u give forcefield.createSystem() function
        forcefield_kwargs ={'constraints' : app.HBonds,
                            'rigidWater' : True,
                            'removeCMMotion' : False,
                            'hydrogenMass' : 4*amu }
		
        system_generator = SystemGenerator(forcefields=ff_files, 
                                           small_molecule_forcefield=template_ff,
                                           forcefield_kwargs=forcefield_kwargs, 
                                           cache='db.json')
        

        input_sdfs=[]
        for idx, sdf in enumerate(input_sdf, 1):
        
            
            path_sdf=f'{input_path}/{sdf}'

            if not os.path.exists(path_sdf):
                
                print(f'\tFile {path_sdf} not found!')
            else:
                print(f'\tAdding extra SDF file {idx} to pre-system: {path_sdf}')
                input_sdfs.append(path_sdf)
                       

        molecules = Molecule.from_file(*input_sdfs, file_format='sdf')
        
        print(molecules)
        
        system = system_generator.create_system(topology=input_system.topology)#, molecules=molecules)
        
        
        gaff = GAFFTemplateGenerator(molecules=molecules, forcefield=template_ff)
        gaff.add_molecules(molecules)
        print(gaff)
        forcefield.registerTemplateGenerator(gaff.generator)
        
        #forcefield.registerResidueTemplate(template)
        
        print(system)
        print(forcefield)
        
        
        return system, forcefield
 

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
    def sniffMachine(gpu_index='0'):
        """
        

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
    
    
    
    
# =============================================================================
# ###########################################
# #####################LEGACY################



# =============================================================================
#             print('No protonation dictionary provided. Using default.')
#     
#             protonation_dict = {('A',1499): 'CYX', 
#                     ('A',1501): 'CYX', 
#                     ('A',1516): 'CYX', 
#                     ('A',1520): 'CYX', 
#                     ('A',1529): 'CYX', 
#                     ('A',1533):'CYX', 
#                     ('A',1539):'CYX', 
#                     ('A',1631):'CYX', 
#                     ('A',1678):'CYX', 
#                     ('A',1680):'CYX', 
#                     ('A',1685):'CYX', 
#                     ('B',36): 'LYN'} 
# 
# 
# =============================================================================
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

  # =============================================================================
#         
#                       extra_input_pdb=[], #['SAM_H3K36.pdb', 'ZNB_H3K36.pdb']
#               ff_files=[], #['amber14-all.xml', 'amber14/tip4pew.xml', 'gaff.xml'],
#               extra_ff_files=[], #['SAM.xml', 'ZNB.xml']
#               extra_names=[], #['SAM', 'ZNB'],
#         
# ============================================================================= 
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
# =============================================================================
#         
#         
#                 # Simulation Options
# 
#         self.Simulate_Steps = 5e7  # 100ns
# 
#         self.SAM_restr_eq_Steps = 5e6          
#         self.SAM_free_eq_Steps = 5e6   
# =============================================================================
