System Setup with openMM
========================
    
These are instructions for system setup using **openMM**.

    
Input files
-----------    

To setup simulation systems using openMM, a topology of the *System* needs to be provided, together with the selected force field input files.

Force field files are dependent on the selected type (*e.g.* AMBER, CHARMM). 
To facilitate transferability, there are tools such as  `Open Force Field Toolkit <https://github.com/openmm/openmmforcefields>`_ which allow users to work with different force fields.
As of April 2021, the following force fields are supported by the Toolkit: 
   1. AMBER
   2. CHARMM 
   3. Open Force Field Initiative 
   
   


1. **Topology**

   A `PDB <https://www.rcsb.org/>`_ file (or other openMM compatible format) containing molecule information and simulation box definitions.
   
.. warning::
   When simulating proteins, it is typical to use the PDB file of the protein structure as starting input file.
   Some protein PDB files need to be "polished" before being suitable for usage. 
   There are tools that can do this for you, such as `pdb-tools <https://github.com/haddocking/pdb-tools>`_ (`Webserver <https://wenmr.science.uu.nl/pdbtools/>`_)
   openMM has included 


2. **Force field**
    
    #TODO: redirect to openMM docs

System setup
------------

    Sytem up is the set of instructions (using openMM) that are required to put together the input files and define the simulation protocol.
    This can be made with a set of instructions that are executed on Jupyter, script or CLI. 
    The required steps are usually:

       * Define the simulation box.
       * Which molecules and how many are in the simulation.
       * How to simulate it.


    OpenMM provides `cookbooks and tutorials <http://openmm.org/tutorials/index.html>`_ that can used as templates of these instructions.
    
    
    ...
    #TODO: AMBER to openMM import