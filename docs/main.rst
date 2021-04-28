Main
====

Project
+++++++

A project is defined using the **Project** class. 
Defining a *project* spawns all the simulation *systems* (with the help of **Systems** generator class).

The file structure and file names of the *project* are generated under the **workdir** using a hierarquical nomenclature that is used for system setup, simulation, and analysis. 
The **results** folder defines where results from the analysis will be outputed.

The ***Project*** module accept an arbitrary number of combination of elements, and generates a tree structure of the Cartesian product of ordered set of elements in *hierarchy* (in its current implementation, can be changed to other combinatronics schemes). 
The limitation resides on the required input files upon simulation execution for all the requested combinations.

For a protein-ligand simulation setup, with variation of some atribute of a ligand, the elements then become **protein(s)**, **ligand(s)** and ligand **parameter(s)**. 

.. note::
   
   * Other parameters can be coded.




System
++++++

Each system is defined by a set of attributes, for example: *protein*.
Each attribute can have a undefined number of elements, for example: *protein*=(protein1, protein2).

The **System** class work as a system generator, and accepts requests from the **Project** class.
 
.. note::

   The number of attributes is not fixed, so the user can set as many as necessary. 
   Given the combinatorial generation of systems based, the number of attributes is recommended to be low.
 
 
Hierarchy
+++++++++
 
The hierarchy refers to the way systems are grouped based on the shared elements of some attribute. 
By specifying the hierarchy, a tree-like structure of systems is generated. 
The order of provided attributes sets the tree **levels**: the first attribute will be the root, and the remaining attributes will be branching points.
  

Replicates
++++++++++

The replicate value defines how many times each system is to be simulated. 
This is an optional attribute, which when not defined reverts to the default value.

 
Example
-------
 
A concrete example of a project is the following. 

Attributes: 
    * protein = protein1, protein2
    * ligand = ligand1, ligand2, ligand3
    * parameter = parameter1, parameter2, parameter3, parameter4
    * replicates (optional) = 5


.. code-block:: python

   workdir=base_path+'Root'
   results=workdir+"/project_results"

   #Check for project folders and creates them if not found)
   tools.Functions.fileHandler([workdir, results], confirmation=False)

   protein=['protein1', 'protein2']
   ligand=['ligand1', 'ligand2', 'ligand3']
   parameters=['parameter1', 'parameter2', 'parameter3', ]

            
            
   project=main.Project(title='projectTutorial', 
                     hierarchy=('protein', 'ligand', 'parameter'), 
                     workdir=workdir, 
                     parameter=parameters, 
                     replicas=5, 
                     timestep=5,
                     protein=protein, 
                     ligand=ligand, 
                     results=results)


project_systems=project.setSystems()

..

Which yields the following tree:


::

 Root
 ├── protein1
     ├── ligand1
     	├── parameter1
                ├──replicate1
                ├──replicate2
                ├──replicate3
                ├──replicate4
                └──replicate5		
     	├── parameter2
     	├── parameter3
     	└── parameter4
     ├──ligand2
     └──ligand3
 └──protein2




.. automodule:: source.main
   :members:
   :undoc-members:
   :show-inheritance: