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
 
            #TODO: Call to GAFF Generator SMILES
        # =============================================================================
#        smiles='C[S+](CC[C@@H](C(=O)[O-])N)C[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)O'
#        molecules =  Molecule.from_smiles(,allow_undefined_stereo=True)
# 
#         
#         print(molecules)
#         
#
#         
#         
#         gaff = GAFFTemplateGenerator(molecules=molecules, forcefield=template_ff)
#         gaff.add_molecules(molecules)
#         forcefield.registerTemplateGenerator(gaff.generator)
# =============================================================================
    
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
    def runSimulations(self, run_Emin=True, run_equilibrations=True, run_productions=False):
        
        simulations={}
        
        if run_Emin:
            
            simulations['minimization']=self.simulationSpawner(self.minimization, kind='minimization')
        
        if run_equilibrations:
        
            for idx, p in enumerate(self.eq_protocols, 1):
            
                print(f"EQ run {idx}: {p['ensemble']}")
                simulations[f"EQ-{p['ensemble']}"]=self.simulationSpawner(p, 
                                                       index=idx, 
                                                       kind='equilibration', 
                                                       label=p['ensemble'])
        
        if run_productions:
                            
            for idx, p in enumerate(self.productions, 1):
                
                
                if not run_equilibrations:
                    idx=idx-1
                
                print(f"Production run {idx}: {p['ensemble']}")
                simulations[f"MD-{p['ensemble']}"]=self.simulationSpawner(p, 
                                           index=idx, 
                                           kind='production', 
                                           label=p['ensemble'])
                        
        return simulations


    def simulationSpawner(self,
                          protocol, 
                          label='NPT', 
                          kind='production', 
                          index=0):
        
        
        #TODO: Set integrator types    
        integrator = omm.LangevinIntegrator(self.temperature, self.friction, self.dt)
        simulation = app.Simulation(self.topology, 
                                    self.system, 
                                    integrator, 
                                    self.platform, 
                                    self.platformProperties)
        
        simulation.context.setPositions(self.positions)
                
        state_initial=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        energy_initial = state_initial.getPotentialEnergy()
        positions_initial = state_initial.getPositions()

        print('Spwaning new simulation:')
        
        if kind == 'minimization':
            
            print('\tEnergy minimization:')
            print(f'\tSystem potential energy: {energy_initial}')
            
            simulation.minimizeEnergy()
            state=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            
            self.structures['minimization']=self.writePDB(self.topology, state.getPositions(), name='minimization')
            
            
            #TODO: initialize simulation with flag "state=" from previous state of previous simulations
            
        else:
            
            name=f'{kind}_{label}-{index}'
            trj_file=f'{self.workdir}/{kind}_{label}-{index}'
            
            print(f'\t{name}: populating file {trj_file}.')
            
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
        
        
            #Set velocities if continuation run
            if index == 1 and kind == 'equilibration':
                print(f'\tFirst run {name}. Assigning velocities to {self.temperature}')
                simulation.context.setVelocitiesToTemperature(self.temperature)
                

            
            elif index == 0 and kind == 'production':
                print(f'\tWarning: First run {name} with no previous equilibration. Assigning velocities to {self.temperature}')
                simulation.context.setVelocitiesToTemperature(self.temperature)
            
            else:
                print(f'\tWarning: continuation run  {name}. Assigning velocities from stored state.')
                simulation.context.setVelocities(self.velocities)
                
            
                #TODO: Emulate temperature increase
                #TODO: initialize simulation with flag "state=" from previous state of previous simulations
        
            # Define reporters
            
            simulation.reporters.append(CheckpointReporter(f'{self.workdir}/{name}_{index}.chk', 1000))
            simulation.reporters.append(DCDReporter(f'{trj_file}.dcd', protocol['report']))           
            simulation.reporters.append(HDF5Reporter(f'{trj_file}.h5', protocol['report'], atomSubset=self.trj_indices))
            simulation.reporters.append(app.StateDataReporter(f'{trj_file}.csv', 
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
        
            
        
            self.structures['f{name}_ref']=self.writePDB(self.topology, positions_initial, name=f'{name}_0')
        
            #############
            ## WARNIN ##
            #############
            trj_time=protocol['step']*self.dt
            print(f'\t{kind} in {label} ensemble ({trj_time})...')
            
            
            checkpoint_final=f'{self.workdir}/{name}_{index}.chk'
            checkpoint=f'{self.workdir}/{name}.chk'
            
            
            if os.path.exists(checkpoint_final):
            
                print(f'\t{checkpoint_final} file found. Simulation terminated. Loading final state')
                
                with open(checkpoint_final, 'rb') as f:
                    simulation.context.loadCheckpoint(f.read())
                
                positions = simulation.context.getState(getPositions=True).getPositions()

            elif os.path.exists(checkpoint):
                
                print(f'\t{checkpoint} file found. Incomplete simulation. Resuming from there.')
                
                with open(checkpoint, 'rb') as f:
                    simulation.context.loadCheckpoint(f.read())
                    
                positions = simulation.context.getState(getPositions=True).getPositions()
            
            else:
                
                print('\tNo checkpoint files found. Starting new.')
                simulation.step(protocol['step'])
                simulation.saveCheckpoint(f'{self.workdir}/{name}_{index}.chk')
                
                positions = simulation.context.getState(getPositions=True).getPositions()
            
            #############
            ## WARNOUT ##
            #############

            print(f'\t{kind} simulation in {label} ensemble ({trj_time}) completed.')
            self.structures[name]=self.writePDB(self.topology, positions, name=name)

        state=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        self.positions = state.getPositions()
        self.velocities = state.getVelocities()
        self.energy=state.getPotentialEnergy()
        
        self.state=state
        
        

        print(f'\tSystem potential energy: {self.energy}')
        
        #Remove Costum forces (always initialized in next run)
        if label == 'NPT':
            self.system=self.forceHandler(self.system, kinds=['CustomExternalForce', 'MonteCarloBarostat'])
        elif label == 'NVT':
            self.system=self.forceHandler(self.system, kinds=['CustomExternalForce'])
            
        return (simulation, self.state)



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
                
                print(f'\tRestraining set ({force_value*kilojoules_per_mole/angstroms**2}): {len(topology.select(restrained_set))}')
            
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
