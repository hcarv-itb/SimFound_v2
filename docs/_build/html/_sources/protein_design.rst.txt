Protein Design 2021
===================

Protein Design course, Henrique F. Carvalho 2021.

This is a rough backbone tutorial of the topics that will be adressed during the course. 


The core of the course is the **Project**.
The project will be executed in a laboratory, just like in real Biology. 
This time, the laboratory will be **Jupyter**, a program that allows users to create and share code, equations, visualizations, and narrative text. 

To be able to work with JupyterLab, you need to have installed in your working computer **Anaconda**. 
Anaconda allows one to install software packages and handles the dependencies, which is a valuable help when working on different computers.

The programming languange used during the course will be `Python <https://www.python.org/>`_.

At the end of the course, your project will be a notebook containing a description of the tasks in **Markdown** and python code in **Code** cells, with no bugs, fully operational under your local anaconda environment. 


Install instructions
--------------------

Install instructions for Jupyter and openMM.

**1.	Install Anaconda**


`Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ is a software package manager, which will be handling all the dependencies of your machine and the programs that run under it. 
Jupyter is one of such cases, but there are a lot more programs for scientific computation available. 
Programs running in python, R and other programming languages can be run the user's own OS, even Windows.

  1.1. Use Anaconda to launch `JupyterLab <https://jupyterlab.readthedocs.io/en/latest/>`_ in Windows. Alternatively, in a CLI (command line interface in UNIX machines), type:

.. code-block:: bash

	jupyter notebook
	
	
.. note::
    * You can modify launch settings (*e.g.* from 'tree' to 'lab') to open automatically the jupyterLab interface by modifying the **conf.py** file in you user installation.

**2. 	Install openMM**


`OpenMM <https://anaconda.org/omnia/openmm>`_ is a MD simulation engine that runs under a python API, which can be conveniently used in a Jupyter Notebook (Python3 kernel).
To open an anaconda powershell prompt, type on the Windows search tool:
	   
.. code-block:: bash

	anaconda power shell 

..

   In the powershell (or CLI), type: 
  
.. code-block:: bash  

	conda install -c omnia openmm

..

   Confirm installation of all packages and dependencies (y/yes, enter). Read the output messages to confirm successful installation. 


**3.	Execute openMM in JupyterLab**

Jupyter is included in the Anaconda environment. 
It can be launched through Anaconda Navigator OR through Windows search tool.
It will open the JupyterLab interface as a (new) tab in user's default internet browser.

   In a jupyterlab launcher, open a new notebook. Type:
   
.. code-block:: python 

	from simtk import openmm

..

   Execute the cell. If no errors are found, openMM is ready to be used.

.. note::
	* TODO: Instructions for port 8089, --no-browser

Project layout:  Files and folders
----------------------------------

Jupyter is launched in the local user folder. In linux, it is launched in the current workding directory. 
The GUI of Jupyter is launched on your browser, but all operations are still being made in your local machine (not on the Web).


The user can create, move, edit, copy, download etc. files and folders using Jupyter. 
It is convenient to have some kind of file organization, but this is left to the user to decide how to organize its data (at least while using your local working machine).

Example
+++++++

 One approach is to organize data within a **Project** in an hierarchical structure of **Systems**. 
 Each system is defined by a set of attributes, for example: *protein*, *ligand*, *temperature*.
 
 Different systems may share some attributes, such as having the same *protein*.  
 When defining a project, systems may grouped by their shared attributes, from least variable to most variable. 
 
 Also, it is common in MD simulations that more than one simulation replicate is performed for each system to improve sampling. 
 Replicates are two systems which have all attributes equal.
 
 Considering a MD simulation project of **2 Proteins**, **3 Ligands** and **4 Temperatures** in **5 replicas**. 
 A file structure could be something like this (not fully expanded): 



::

 protein1
 ├── ligand1
 	├── temperature1
                ├──replicate1
                ├──replicate2
                ├──replicate3
                ├──replicate4
                └──replicate5		
 	├── temperature2
 	├── temperature3
 	└── temperature4
 ├──ligand2
 └──ligand3
 protein2


.. note:: 
  * Further commands can be executed by lauching a **Terminal**. 
    This will provide a bash-based interface to the machine, and endows the user to perform additional operations not restricted to those available under Jupyter GUI.
    
    
Input files for openMM
----------------------

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
   There are tools that can do this for you, such as `pdb-tools <https://github.com/haddocking/pdb-tools>`_.


2. **Force field**

   * GAFF (group 1)
      * XML-based 
      * (extra_molecules_nb).xml
      * (extra_molecules_bb).xml
    


   * CHARMM (group 2)

     forcefield.top
    ...
    
    #TODO: CHARMM to openMM import

3. **System set up instructions**

    Sytem up is the set of instructions (using openMM) that are required to put together the input files and define the simulation protocol.
    This can be made with a set of instructions that are executed on Jupyter, script or CLI. 
    The required steps are usually:

       #. Define the simulation box.
       
       #. Which molecules and how many are in the simulation.
       
       #. how to simulate it.

    ...
    #TODO: AMBER to openMM import