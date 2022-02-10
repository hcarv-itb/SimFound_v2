# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:24:37 2021

@author: hcarv
"""
#SFv2
try:
    import tools
    import Visualization
    
except Exception as v:
    print(f'Warning: {v.__class__} {v}')
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
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as md
import os
import glob
import pickle
import numpy as np
import datetime
import nglview
import random
import shutil
import re



class Protocols:
    
    ref_name ='system.pdb' 

    def __init__(self,
                 project,
                 name='undefined',
                 def_input_ff='/inputs/forcefields',
                 def_input_struct=None,
                 def_job_folder='/jobs',
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
        self.project = project
        self.protocols = project.systems
        self.systems = project.systems
        self.ow_mode = overwrite_mode

# =============================================================================
#         for name, system in project.systems.items():
#             #print(f'Setting openMM simulation of {name} protocols in {system.name_folder}', end='\r')
#             self.protocols[name]=system #self.check_omm(system, self.ow_mode) #system
# =============================================================================
                                
        self.workdir =project.workdir
        
        if def_input_struct != None:
            print('Warning! ')
            self.def_input_struct=def_input_struct
        else:
            self.def_input_struct=project.def_input_struct
        print('Default input folder for structures: ', self.def_input_struct)    
        
        
        self.def_input_ff=project.def_input_ff
        self.def_job_folder=os.path.abspath(f'{project.workdir}/{def_job_folder}')
        
        tools.Functions.fileHandler([self.def_input_struct, self.def_input_ff, self.def_job_folder])
        
        #status of openMM objects.
        
        

            
        

    def sample_mixer(self, msm_model):
        
        print(self.project.def_input_struct)
        model_samples = list(set(glob.glob(f'{self.project.def_input_struct}/{msm_model}/sample_*.pdb')) 
                             - set(glob.glob(f'{self.project.def_input_struct}/{msm_model}/*delchain*.pdb')))

# =============================================================================
#         random_sample1 = random.sample(model1_samples, self.project.replicas)
#         print(len(random_sample))
#         
#         if set(random_sample1) != len(random_sample1):
#             print('Warning! Selected samples contain one or more duplicates')
# =============================================================================
        return model_samples 
        

    @tools.log
    #@Visualization.show
    def handle_pdb(self, 
                   input_pdb, 
                   ligand_pdb=[], 
                   out_folder=None, 
                   box_size=150, 
                   tolerance=5.0, 
                   edit_chains=[1, 2]):
        """
        

        Parameters
        ----------
        input_pdb : TYPE
            DESCRIPTION.
        ligand_pdb : TYPE, optional
            DESCRIPTION. The default is [].
        out_folder : TYPE, optional
            DESCRIPTION. The default is None.
        box_size : TYPE, optional
            DESCRIPTION. The default is 150.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 5.0.
        edit_chains : TYPE, optional
            DESCRIPTION. The default is [1, 2].

        Returns
        -------
        complex_pdb : TYPE
            DESCRIPTION.

        """
        
        
        def clean_previous(input_pdbs):
            for input_pdb in input_pdbs:
                file_path, file_name_ = os.path.split(input_pdb)
        
                no_chain_files = glob.glob(f'{file_path}/*delchain*.{extension}')
                if len(no_chain_files):
                    print('has delchain')
                    for file in no_chain_files:
                        os.remove(file)

        
        #TODO: make this list for inputs, this is planned along the function.
        
        file_path, file_name_ = os.path.split(input_pdb) 
        file_name, extension = file_name_.split('.')
        if out_folder != None:
            file_path = out_folder
        
        #clean_previous([input_pdb])
        #clean_previous(ligand_pdb)
        

        
        #tools.Tasks.run_bash('pdb_wc', input_pdb)

        no_counterions = tools.Tasks.run_bash('pdb_delchain', input_pdb, '-D', mode='out') #remove CL ions chainD
        
        
        #TODO: this is set for single protein, single ligand. Make flexible for multiples
        protein_pdb = [no_counterions]
        protein_name =self.project.protein[0]
        ligand_name = self.project.ligand[0]
        n_protein = len([protein_pdb])
        n_ligand = len(ligand_pdb)
        
        check_paths = protein_pdb + ligand_pdb
        #clean_previous(check_paths)
        copy_files =[]
        for file in check_paths:
            print(file)
            if re.search('@', file):
                file_path_original, file_name_mod = os.path.split(file)
                shutil.copy(file, f'{out_folder}/{file_name_mod}')
                copy_files.append(f'{out_folder}/{file_name_mod}')
            else:
                copy_files.append(file)
        protein_pdb=copy_files[:len(protein_pdb)]
        ligand_pdb=copy_files[len(protein_pdb):]
        
        base_name = f'{file_path}/{protein_name}-{ligand_name}_{n_protein}-{n_ligand}'
        
        #this is loading two times instead of passing ref to packmol since not always protein is ref
        ref = md.load_pdb(protein_pdb[0])
        #print(ref.topology)
        
        
        #Warning! for packmol, weird names cannot be used otherwise fortran errors occur.
        complex_pdb = self.build_w_packmol(base_name,  
                                           protein_pdb,  
                                           ligand_pdb, 
                                           box_size=box_size,
                                           tolerance=tolerance)

        print(f'Complex: {md.load_pdb(complex_pdb).topology}')
        file_name_complex = os.path.split(complex_pdb)[1].split('.')[0]
        chain_files_stored = glob.glob(f'{file_path}/{file_name_complex}_?.{extension}')  
        if len(chain_files_stored):
            for file in chain_files_stored:
                os.remove(file)
        #Split chains for openMM


        clean_previous([input_pdb])


        print(f'Pre-processing for openMM')
        split=tools.Tasks.run_bash('pdb_splitchain', complex_pdb, mode='create') 
        chain_files = glob.glob(f'{file_path}/{file_name_complex}_?.{extension}')
        
        

        
        for idx, pdb_file in enumerate(chain_files): #chain_files[n_chains:]):
            chain_ID = chr(ord('@')+idx+1)
            
            
            print(f'\tChain {idx} ID:{chain_ID}')
            if idx in edit_chains:
                print(f'\tchain edit: {idx}_CONNECT')
                chain_file_name, ext = os.path.split(pdb_file)[1].split('.')
                
                #TODO: extract CONNECT records from input file, or reset indexes and build anew.
                chain_edit_file=f'{self.def_input_struct}/{idx}_CONNECT'
                files = [pdb_file, chain_edit_file]
                edit_file = f'{file_path}/{chain_file_name}_edit.{ext}'
                with open(edit_file, 'w') as outfile:
                    for file in files:
                        with open(file) as infile:
                            outfile.write(infile.read())
                ref=md.load_pdb(edit_file)
            else:
                ref=md.load_pdb(pdb_file)
            print('\t', ref.topology)


        return complex_pdb



    def build_w_packmol(self,
                        base_name,
                        protein_pdb,
                        ligand_pdb,
                        box_size=150.0, 
                        n_protein=1,
                        tolerance=7.0):
        """
        

        Parameters
        ----------
        base_name : TYPE
            DESCRIPTION.
        protein_pdb : TYPE
            DESCRIPTION.
        ligand_pdb : TYPE
            DESCRIPTION.
        box_size : TYPE, optional
            DESCRIPTION. The default is 150.0.
        n_protein : TYPE, optional
            DESCRIPTION. The default is 1.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 7.0.

        Returns
        -------
        None.

        """
            

        ref = md.load_pdb(protein_pdb[0])
        print(ref.topology)
        print('Building box of size ', box_size)
        n_chains = ref.n_chains
        for chain in ref.topology.chains: #['index', 'topology', '_residues']
            
            chain_ID = chr(ord('@')+chain.index+1)
            print(f'Chain {chain.index} ID:{chain_ID}')
            if len(chain._residues) < 30:
                for res in chain._residues:
                    print('\t', res)
            else:
                print(f'\tFrom {chain._residues[0]} to {chain._residues[-1]}')
                    
        out_file = f'{base_name}.pdb'
        job_file = f'{base_name}.inp'         

        center_box = box_size/2
        #Warning! due to PBC, box should have 1 angstrom padding
        with open(job_file, 'w') as f:
            f.write(f'''
# {base_name}

tolerance {tolerance}
output {out_file}
filetype pdb
seed -1

add_box_sides 1.0




structure {protein_pdb[0]}
  number {n_protein}
  inside cube 0 0 0 {box_size} 
  center 
  fixed {center_box} {center_box} {center_box} 0 0 0
end structure
''')

        #TODO: This is for extra inputs which  are protein, has to handle chains
        #TODO: For other molecules, this can be done in one step by specifying molecule number in single call
            print('\tAdding ligand molecules:')
            for i, lig in enumerate(ligand_pdb, n_chains):
                print(f'\t{i}: {lig}')
                lig = tools.Tasks.run_bash('pdb_delchain', lig, '-B', mode='out') #remove CL ions chainB
                ref_ = md.load_pdb(lig)
                n_chain_ = ref_.n_chains

                for i_ in range(i, i+n_chain_):
                    chain_ID = chr(ord('@')+i_+1)
                    print(f'\tChain {i_} ID: {chain_ID} {ref_.topology}') 
                    f.write(f'''
structure {lig}
  number 1
  resnumbers 1
  chain {chain_ID}
  inside cube 0 0 0 {box_size}
end structure
''')

        create=tools.Tasks.run_bash('packmol', job_file, '<', mode='create') 
        print('Box generation: ', create)
        return out_file


    @staticmethod    
    def check_omm(system, ow_mode):
        
        stored_omm=glob.glob(f'{system.name_folder}/omm_*.pkl')
        system.stored_sim=(1, f'{system.name_folder}/sim.pkl')
        
        
        if len(stored_omm) > 0:
            
            print(f'\tWarning! Found conserved openMM system(s). {stored_omm[0]}')
            idx=int(stored_omm[0].split('omm_')[1].split('.pkl')[0])
            
            if ow_mode:
                system.omm_current=(idx, f'{system.name_folder}/omm_{idx}.pkl')
                system.omm_previous = (0, f'{system.name_folder}/omm_{idx}.pkl')
                system.status = 'new'
            else:
                system.omm_current=(idx+1, f'{system.name_folder}/omm_{idx+1}.pkl')
                system.omm_previous=(idx, f'{system.name_folder}/omm_{idx}.pkl')
                system.status = 'open'
 
        else:
            system.omm_current=(1, f'{system.name_folder}/omm_1.pkl')
            system.omm_previous=(0, f'{system.name_folder}/omm_1.pkl')
            system.status = 'new'
        
        print(f'\tStatus {system.status}: {system.omm_previous[0]}') 
        
        if system.status == 'open' and not ow_mode:
            print('\tLoading conserved openMM system: ', system.omm_previous[1])
            
            ref_pdb=f'{system.name_folder}/{Protocols.ref_name}'
                                              
            pkl_file=open(system.omm_previous[1], 'rb')
            system.system_omm = pickle.load(pkl_file)
            
            pdb = app.PDBFile(ref_pdb)
            #pre_system = app.Modeller(pdb.topology, pdb.positions)
                      
            system.topology_omm=pdb.topology
            system.positions=pdb.positions
        
        return system


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
              ionicStrength=0*unit.molar,
              residue_variants={},
              other_omm=False,
              input_sdf_file=None,
              box_size=9.0*unit.nanometers,
              insert_smiles=None,
              padding_=None):
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
        
        from openmmforcefields.generators import GAFFTemplateGenerator
        from openff.toolkit.topology import Molecule


        self.protocols_omm={}

        for name, system_protocol in self.systems.items():
        
            system=self.check_omm(system_protocol, self.ow_mode) 
                
            if system.status == 'new' or self.ow_mode:
                print('\tGenerating new openMM system for: ', name)
                system.structures = {}
                
                
                if input_pdb is None:
                    print('Will read input PDB from input topology of system: ', system.input_topology)
                    system.structures['input_pdb']=system.input_topology
                else:
                    system.structures['input_pdb'] = f'{system.name_folder}/{input_pdb}'
                    print(f'Will read default input PDB from system folder. Extra PDB files (if used) must be there: {system.name_folder}/{system.supra_folder}.pdb')
                    input_path_extras = system.name_folder
# =============================================================================
#                 else: 
#                     system.structures['input_pdb'] = f'{self.def_input_struct}/{input_pdb}'
#                     print(f'Will read default input PDB from default input structures folder: {self.def_input_struct}/{input_pdb}')
#                     input_path_extras = None
# =============================================================================


            
                #Fix the input_pdb with PDBFixer
                if fix_pdb:
                    pdb=PDBFixer(system.structures['input_pdb'])
                    pdb.findMissingResidues()
                    pdb.findMissingAtoms()
                    pdb.addMissingAtoms()
            
                else:
                    pdb = app.PDBFile(system.structures['input_pdb'])
            
                #Generate a Modeller instance of the fixed pdb
                #It will be used to populate system
                pre_system = app.Modeller(pdb.topology, pdb.positions)
        
        
                #Add ligand structures to the model with addExtraMolecules_PDB
                if len(extra_input_pdb) > 0:
    
                    pre_system, self.extra_molecules=self.addExtraMolecules_PDB(pre_system, extra_input_pdb, input_path=input_path_extras)
    
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
                        variants_ = self.setProtonationState(pre_system.topology.chains(), protonation_dict=residue_variants)
                        
                        pre_system.addHydrogens(forcefield, pH=pH_protein, variants=variants_ )
    
                    else:
                        pre_system.addHydrogens(forcefield, pH=pH_protein)
            
    
                #Call to solvate()
                #TODO: For empty box, add waters, remove then
                #Will neutralize and fix Ionic Strength
                print('\tSolvating.')
                if solvate:
                    pre_system=self.solvate(pre_system, forcefield, box_size=box_size, padding=padding_, ionicStrength=ionicStrength)
        
                system.topology_omm=pre_system.topology
                system.positions=pre_system.positions
        
                #Define system. Either by provided pre_system, or other_omm system instance.
                if other_omm:    
                    system_omm = self.omm_system(input_sdf_file, 
                                                         pre_system,
                                                         forcefield,
                                                         self.def_input_struct,
                                                         ff_files=ff_paths, 
                                                         template_ff='gaff-2.11')
                    #forcefield not needed?? 
                else:    
                    #Create a openMM topology instance
                    system_omm = forcefield.createSystem(pre_system.topology, 
                                             nonbondedMethod=app.PME, 
                                             nonbondedCutoff=1.0*nanometers,
                                             ewaldErrorTolerance=0.0005, 
                                             constraints='HBonds', 
                                             rigidWater=True)

                system_pkl=open(system.omm_current[1], 'wb')
                pickle.dump(system_omm, system_pkl)
                system_pkl.close()
            
                #Update attributes
                system.system_omm=system_omm
                system.status = 'open'
                system.structures['system']=self.writePDB(system.name_folder, pre_system.topology, pre_system.positions, name='system')   
                print(f"\tSystem is now converted to openMM type: \n\tFile: {system.structures['system']}, \n\tTopology: {system.topology_omm}")     
  
            self.protocols_omm[name]=system
        
        print(self.protocols_omm)

    def setSimulations(self, 
                       dt = 0.002*picoseconds, 
                       temperature = 300*kelvin, 
                       friction = 1/picosecond,
                       pressure=1*atmospheres,
                       pH=7.0,
                       productions=[],
                       equilibrations=[],
                       store_only='not water'):
        """
        Function to setup simulation protocols, openMM oriented.
        Updates a simulation object.
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
        self.steps={}
        self.trj_write={} 
        
        for name, system in self.protocols_omm.items():
            
            print('setting up: ', name)
            selection_reference_topology = md.Topology().from_openmm(system.topology_omm)
            system.trj_indices = selection_reference_topology.select(store_only)                 
            system.minimization={}
            system.equilibrations=[]
            system.productions=[]

            for eq in equilibrations:
            
                eq=self.convert_Unit_Step(eq, self.dt)
                system.equilibrations.append(eq)

            for prod in productions:
        
                prod=self.convert_Unit_Step(prod, self.dt)
                system.productions.append(prod)
                

    #TODO: Get platform features from setSimulations, useful for multiprocessing.
    def runSimulations(self,
                       where='Local',
                       run_Emin=True, 
                       run_equilibrations=True, 
                       run_productions=False, 
                       overwrite_=False,
                       resume_=True,
                       minimum_effort=False,
                       compute_time=0.05,
                       run_time_HPC = '47:30:00',
                       partition_HPC='gpu_4',
                       reportFactor=0.1,
                       gpu_index='0',
                       **kwargs):
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
        print(type(compute_time), type(reportFactor), compute_time, reportFactor)
        reportTime=float(compute_time) * reportFactor
        #Call to setMachine
        #TODO: Fetch time and specify which of the replicates to run here.
        #TODO: Make call here for generation of corresponding .sh file generattion and .sh spawner.
        
        for name, system in self.protocols_omm.items():
        
            print('System: ', name)
            
            replicate = self.systems[name].replicate
            
            if system.status != 'complete':

                if where != 'Local':
                    job=tools.Tasks(machine=where,
                                     n_gpu=len(gpu_index),
                                     run_time=run_time_HPC)
                    job.setMachine(gpu_index_= gpu_index)
                    job.generateSripts(name, self.def_job_folder, replicate, compute_time, **kwargs)
                
                    print('Script(s) generated for: ', name)
                                
                else:
                    job=tools.Tasks(machine=where,
                                     n_gpu=len(gpu_index),
                                     run_time=compute_time)
                    system.simulations = {}
                    self.platform, self.platformProperties =job.setMachine(gpu_index_= gpu_index, force_platform=True)
            
                    if run_Emin:
                        print("\nEnergy minimization")
                        system.simulations['minimization']=self.simulationSpawner(system,
                                                                                  index=1,
                                                                                  kind='minimization',
                                                                                  label='Emin',
                                                                                  overwrite=overwrite_,
                                                                                  resume=resume_,
                                                                                  compute_time_=compute_time)
        
                    if run_equilibrations:
                        for idx, protocol in enumerate(system.equilibrations, 1):
                            print(f"\nEQ run {idx}: {protocol['ensemble']}")
                            system.simulations[f"EQ-{protocol['ensemble']}"]=self.simulationSpawner(system,
                                                                                                    protocol=protocol, 
                                                                                                    index=idx, 
                                                                                                    kind='equilibration', 
                                                                                                    label=protocol['ensemble'],
                                                                                                    overwrite=overwrite_,
                                                                                                    resume=resume_,
                                                                                                    compute_time_=compute_time,
                                                                                                    reportFactor_=reportFactor,
                                                                                                    fixed_step=True)
                
                    if run_productions:
                        for idx, protocol in enumerate(system.productions, 1):
                            if not run_equilibrations:
                                idx=idx-1
                        
                            print(f"\nProduction run {idx}: {protocol['ensemble']}")
                            system.simulations[f"MD-{protocol['ensemble']}"]=self.simulationSpawner(system,
                                                                                                    protocol=protocol, 
                                                                                                    index=idx, 
                                                                                                    kind='production', 
                                                                                                    label=protocol['ensemble'],
                                                                                                    overwrite=overwrite_,
                                                                                                    resume=resume_,
                                                                                                    compute_time_=compute_time,
                                                                                                    reportFactor_=reportFactor,
                                                                                                    fixed_step=minimum_effort)
            
                    
    def simulationSpawner(self,
                          system,
                          protocol=None, 
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
        
        #print(system.status)
        #TODO: Set integrator types    
        integrator = omm.LangevinIntegrator(self.temperature, self.friction, self.dt)
        
        name=f'{kind}_{label}-{index}'
        trj_file=f'{system.name_folder}/{name}' 
        
        print(name)
        
        reportTime=float(compute_time_) * reportFactor_
        
        #Initiate simulation from stored state.
        simulation_init = app.Simulation(system.topology_omm, 
                                        system.system_omm, 
                                        integrator, 
                                        self.platform, 
                                        self.platformProperties)
        simulation_init.context.setPositions(system.positions)
        state_init=simulation_init.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        energy_init = state_init.getPotentialEnergy()
        positions_init = state_init.getPositions()
        
        
        #Check for checkpoints and update simulation state
        simulation, system.status, last_step, checkpoints=self.setSimulationState(simulation_init, self.dt, trj_file, resume, overwrite)
        
        if system.status == 'complete':
            state_sim=simulation_init.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            print('\tSystem is complete.')
            
        #Run the Eminimization
        elif kind == 'minimization' and system.status == 'new':
               
            print('\tEnergy minimization')
            simulation.minimizeEnergy()
            simulation.saveState(checkpoints['final'])
            state_sim=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            system.structures[name]=self.writePDB(system.name_folder, system.topology_omm, state_sim.getPositions(), name=name)
        
        else:
                
            #Fetch number of steps
            steps_, append_, total_step=self.getSteps(protocol, simulation, last_step, self.dt)
            
                
            if system.status == 'new':
                
                #Write initial 
                print('\tSetting new system:', system.name)
                
                system.structures['f{name}_init']=self.writePDB(system.name_folder, system.topology_omm, positions_init, name=f'{name}_init')
                print(f'\tSystem initial potential energy: {energy_init}')
               
                #Run the equilibrations or productions  
    

     
                #Simulation has not started. Generate new.
                if steps_ == total_step:
                    
                    print(f'\tPopulating file(s) {trj_file}.*')
            
                
                    #If there are user defined costum forces
                    try:
                        if protocol['restrained_sets']:
                            system.system_omm=self.setRestraints(system, protocol['restrained_sets'])
    
                    except KeyError:    
                        pass
        
                    #If ensemble is NPT, add Barostat.
                    #TODO: frequency is hardcoded to 25, make it go away. Check units throughout!
                    #TODO: Set other barostats.
                    if label == 'NPT':
                        print(f'\tAdding MC barostat for {label} ensemble')      
                        system.system_omm.addForce(omm.MonteCarloBarostat(self.pressure, self.temperature, 25)) 
        
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
                        print('\tWarning! Continuation run. Assigning velocities from stored state.')
                        simulation.context.setVelocities(system.velocities)
                    
                simulation.reporters.append(HDF5Reporter(f'{trj_file}-{last_step}.h5', protocol['report'], atomSubset=system.trj_indices))
                simulation.reporters.append(app.StateDataReporter(f'{trj_file}.csv', protocol['report'], 
                                                                  step=True, totalEnergy=True, temperature=True, density=True,
                                                                  progress=True, remainingTime=True, speed=True, 
                                                                  totalSteps=steps_, separator=',')) 
            
            simulation.reporters.append(DCDReporter(f'{trj_file}.dcd', protocol['report'], append=append_))
      


            #Execute the two types of simulation run
            if fixed_step:
                
                simulation.reporters.append(CheckpointReporter(f'{trj_file}.chk', protocol['report']))
                print('\tSimulating...')
                
                try:
                    simulation.step(steps_)
                    simulation.saveState(checkpoints['final'])
                    last_step = simulation.currentStep
                    system.status = 'complete'
                    
                except:
                     last_step = simulation.currentStep
                     print(f'\t Warning! Could not run all steps: {last_step}/{total_step}')
                     simulation.saveState(checkpoints['timeout'][1])
                     simulation.saveCheckpoint(checkpoints['timeout'][0])
                     system.status = 'open'
            
            else:
                print(f'\tSimulating for {str(datetime.timedelta(hours=compute_time_))} ({str(datetime.timedelta(hours= reportTime))} checkpoints)...')
                print(compute_time_)
                simulation.runForClockTime(compute_time_, 
                                           checkpointFile=checkpoints['checkpoint'][0],
                                           stateFile=checkpoints['checkpoint'][1], 
                                           checkpointInterval=reportTime)
                simulation.saveState(checkpoints['timeout'][1])
                simulation.saveCheckpoint(checkpoints['timeout'][0])

                last_step = simulation.currentStep
            
            remaining_step_end = total_step - last_step
            state_sim=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            
            
            #Evaluate status of current execution
            if remaining_step_end > 0:
                    
                print(f"\tWarning! Simulation terminated by timeout: {last_step}/{total_step} = {remaining_step_end} steps to complete.")
                system.structures['f{name}_chk']=self.writePDB(system.name_folder, system.topology_omm, state_sim.getPositions(), name=f'{name}_chk')
                system.status = 'open'
                
            else:
                
                print(f"\tSimulation completed or extended: {last_step}/{total_step} (+{np.negative(remaining_step_end)}) steps.")
                simulation.saveState(checkpoints['final'])
                system.structures[name]=self.writePDB(system.name_folder, system.topology_omm, state_sim.getPositions(), name=name)
                system.status = 'complete'
                
                #Remove forces if simulation is over.
                if label == 'NPT':
                    system.system_omm=self.forceHandler(system.system_omm, kinds=['CustomExternalForce', 'MonteCarloBarostat'])
                elif label == 'NVT':
                    system.system_omm=self.forceHandler(system.system_omm, kinds=['CustomExternalForce'])
        

        
        #Update attributes
        system.state=state_sim
        system.positions = state_sim.getPositions()
        system.velocities = state_sim.getVelocities()
        system.energy=state_sim.getPotentialEnergy()     
        system.simulation = simulation 
            
        print(f'\tSystem final potential energy: {system.energy} ({system.status})')
        
        return system
        
        
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
                    status = 'open'
                
                elif os.path.exists(checkpoints['checkpoint'][1]):     
                    simulation, last_step=chkResume(simulation_i, checkpoints['checkpoint'])
                    print('\tA checkpoint file was found.')
                    status = 'open'
                else:
                    print('\tNo checkpoint found.')
                    simulation ,last_step = simulation_i, 0
                    status = 'new'
        
        if overwrite:
            print('\tWarning! Overwrite mode set to True.')
            simulation, last_step = simulation_i, 0
            status = 'new'
            

            
        #state=simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        
        return simulation, status, last_step, checkpoints
        

    #TODO: Merge with forceHandler.
    @staticmethod
    def setRestraints(system, restrained_sets):
                      
                      
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
        

        topology = md.Topology().from_openmm(system.topology_omm)  
        
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
                    
                    force.addParticle(int(res_atom_index), system.positions[int(res_atom_index)].value_in_unit(unit.nanometers))
            
            #   TODO: Decide wether to spawn system reps. Streamlined now. Legacy removes later.    
                system.system_omm.addForce(force)

            return system.system_omm
        
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
    
    @staticmethod
    def writePDB(workdir,
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
        out_pdb=f'{workdir}/{name}.pdb'
        app.PDBFile.writeFile(topology, positions, open(out_pdb, 'w'), keepIds=keepIds)
            
        print(f'\tPDB file generated: {out_pdb}')
        
        return out_pdb


    def addExtraMolecules_PDB(self, system, extra_input_pdb, input_path=None):
        
        if input_path == None:
            input_path=self.def_input_struct
            
        total_mols=[]
        
        for idx, pdb_e in enumerate(extra_input_pdb, 1):
            
            
            path_pdb=f'{input_path}/{pdb_e}'

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
                
                print(f'\tMolecule {idx} ({pdb_e}) could not be added by openMM')

# =============================================================================
#         for mol in total_mols:
#             
#             print(f'\tAdded {mol[0]}: {mol[1]}')  
# =============================================================================
  
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
        if padding is None:
            print('')
            system.addSolvent(forcefield, 
                              model='tip4pew', 
                              neutralize=True, 
                              ionicStrength=ionicStrength, #0.1 M for SETD2
                              boxSize=omm.Vec3(box_size, box_size, box_size)*unit.nanometers)
            
        else:
            system.addSolvent(forcefield, 
                              model='tip4pew', 
                              neutralize=True, 
                              ionicStrength=ionicStrength,
                              padding = padding) #0.1 M for SETD2
            
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
