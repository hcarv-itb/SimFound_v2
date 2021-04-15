Analysis Modules
================


Discretize
----------
Two discretization schemes are currently available, **Uniform spherical shell discretization** and **Combinatorial user-defined spherical shell discretization**.
The *Discretize* object stores an internal copy of the provided feature **data** (DataFrame) and **feature name** (optional) to avoid passing project_system calls. 
Therefore, the module can take an externally-generated DataFrame, so long as the index information is consistent with the module operation.


**Uniform spherical**
+++++++++++++++++++++

Discretization based in spherical shells generates spherical shells along a *feature* coordinate (*i.e.* if 1D).
The module takes as input the **thickness** of the spherical shells and the **limits** of shell edges. 
Discretization can also be made for different subsets of the original *feature* DataFrame, by providing the controls of the **start**, **stop** and **stride** parameters.

Returns a *shell profile* DataFrame, containing the sampled frequency for each spherical shell (index) and for each feature.

**Note:** Subset time requires the user to check what are the frames present in the *discretized* (original *featurized*) DataFrames.


**Combinatorial user-defined**
++++++++++++++++++++++++++++++

Discretize, for each frame, which combinations of shells are occupied by all ligands in the system (one state per frame)

The concept of regions is applied to the 1D \$d_{NAC}$ reaction coordinate, whereby a region is defined as a *spherical shell* along the $d_{NAC}$ radius

Set the shell boundaries (limits, thickness). The minimum (0) and maximum (max) value are implicit.

E.g.: shells (4.5, 10, 12, 24) corresponds to: 
1. shell [0, 4.5[ (A)
2. shell [4.5, 10[ (P)
3. shell [10, 12[ (E)
4. shell [10, 24[ (S)
5. shell [24, max[ (B)


Labels *labels* of each region are optional but recommended (otherwise reverts to numerical). Number of *labels* is plus one the number of defined *regions*.



.. automodule:: source.Discretize
   :members:
   :undoc-members:
   :show-inheritance:
   


Featurize
---------


.. automodule:: source.Featurize
   :members:
   :undoc-members:
   :show-inheritance:
   

Trajectory
----------


.. automodule:: source.Trajectory
   :members:
   :undoc-members:
   :show-inheritance:
   
   
Markov State Models
-------------------

.. automodule:: source.MSM
   :members:
   :undoc-members:
   :show-inheritance:
  


