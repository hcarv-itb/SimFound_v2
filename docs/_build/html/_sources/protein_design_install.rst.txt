Installation and Usage
======================


Jupyter is launched in the local user folder. In linux, it is launched in the current workding directory. 
The GUI of Jupyter is launched on the user's browser, but all operations are still being made in the local machine (not on the Web).


The user can create, move, edit, copy, download etc. files and folders using Jupyter. 

.. note:: 
  * Further commands can be executed by lauching a **Terminal**. 
    This will provide a bash-based interface to the machine, and endows the user to perform additional operations not restricted to those available under Jupyter GUI.
    
  

Installation
------------

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


Usage
-----

#. Testing SFv2

In a jupyter session containing the *source* code of SFv2, open a new notebook. Type:

.. code-block:: python

   from source import main

..

If no errors are found, the **Main** module has been loaded.

#. Testing openMM

In a jupyterlab launcher, open a new notebook. Type:
   
.. code-block:: python 

	from simtk import openmm

..

Execute the cell. If no errors are found, openMM is ready to be used.


.. warning::
	* TODO: Instructions for SFv2 import!