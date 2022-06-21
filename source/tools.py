"""
Created on Fri Nov  6 21:50:14 2020

@author: hcarv

"""

import os
import simtk.openmm as omm
import simtk.unit as unit
import pandas as pd
from functools import wraps
import subprocess
import numpy as np
import re

def log(func):
    """Decorator for system setup"""
    
    @wraps(func)
    def logger(*args, **kwargs):
        print(f'Executing {func.__name__} \n', end='\r')
        return func(*args, **kwargs)
    return logger


class Functions:
    


    @staticmethod
    def pathHandler(project, results):
        
        result_paths = []
        for name, system in project.systems.items(): 
            result_paths.append(system.project_results)
        if results is not None:
            if len(list(set(result_paths))) == 1:
                result_path = list(set(result_paths))[0]
                results = os.path.abspath(f'{result_path}') 
            else:
                results=os.path.abspath(f'{project.results}')
                print('Warning! More than one project result paths defined. reverting to default.')
        else:
            results = results
        
        return results
    
    @staticmethod
    def fileHandler(path, confirmation=False, _new=False, *args):
        """
        Handles the creation of folders.


        """
        
        import os

        for item in path:
            #print(f'Handling: {item}')
            if not os.path.exists(item):
                os.makedirs(item)
                #print(f'Item created: {item}')
            else:
                #print(f'Item exists: {item}')
                if confirmation:
                    if input("Create _new? (True/False)"):
                        os.makedirs(item+'_new', exist_ok=True)
                    else:
                        pass
                elif _new == True:
                    os.makedirs(item+'_new', exist_ok=True)    


    @staticmethod
    def setScalar(parameters, ordered=False, get_uniques=False):
        
        parameter_dict = {}
        
        if ordered:
            parameters= ['50mM', '150mM', '300mM', '600mM', '1M', '2.5M', '5.5M']
            
        for idx, p in enumerate(parameters):

            unit_string = str(p)
            print(unit_string)
            if re.search('K', unit_string):
                scalar=float(str(p).split('K')[0])*unit.kelvin
                #self.parameter_scalar.append(scalar)
                #parameter_label.append(str(scalar).replace(" ",""))
                #print(f'Converted parameter "temperature" (in K) into scalar: {scalar}')
            elif re.search('M', unit_string):
                try:
                    scalar=float(str(p).split('M')[0])*unit.molar
                except ValueError:
                    scalar=float(str(p).split('mM')[0])/1000*unit.molar
                #print(f'Converted parameter "concentration" (in Molar) into scalar: {scalar}')
            else:
                scalar=idx
                print(f'Converted unidentified parameter into scalar: {scalar}')
                
            parameter_dict[p]=str(scalar)
        
        if get_uniques:    

            scalars_unique=set(parameter_dict.values())
            scalars= {}
            for unique in scalars_unique:
                unique_set= []
                for it, scalar in parameter_dict.items():
                    if scalar == unique:
                        unique_set.append(it)
                if len(unique_set):
                    scalars[unique]=unique_set
            return scalars
        
        else:
           return parameter_dict 

    @staticmethod
    def regionLabels(regions, labels):
        """
        Evaluate number of defined regions and corresponding indetifiers.
        Instance will have *region_labels* as provided or with default numerical indetifiers if not.



        """


        region_labels=[]
        if (len(labels) == len(regions) + 1):
            region_labels=labels
        else:
            print('Region labels not properly defined. Number of regions is different from number of labels.') 
            print('No region labels defined. Using default numerical identifiers.')
            
            for idx, value in enumerate(regions, 1):
                region_labels.append(idx)
            region_labels.append(idx+1)
            
        return region_labels 
    
    @staticmethod
    def sampledStateLabels(shells, labels, sampled_states=None):
        """

        Static method that generates state labels for sampled states.
        Requires the *regions* to be provided and optionally the *sampled_states* and *labels*.
        

        Parameters
        ----------
        shells : TYPE
            DESCRIPTION.
        sampled_states : TYPE, optional
            DESCRIPTION. The default is None.
        labels : TYPE
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """      
        #DO NOT TOUCH. MAKING IT CORRECT MAKES IT FOOBAR
        state_names=Functions.stateEncoder(shells, labels)[1] #Call to stateEncoder[0] = state_names
        
        
        if sampled_states is not None:        
            sampled=[]
            for i in sampled_states:
                sampled.append(state_names[i])
        
            return sampled
        else:
            return state_names
        
    @staticmethod    
    def stateEncoder(shells, labels):
        """
        

        Parameters
        ----------
        shells : TYPE
            DESCRIPTION.
        labels : TYPE
            DESCRIPTION.

        Returns
        -------
        state_encodings : TYPE
            DESCRIPTION.
        state_names : TYPE
            DESCRIPTION.

        """
         
        import itertools
        
        
        #region_labels=Functions.regionLabels(shells, labels)
        
        shell_labels=[]
        if (len(labels) == len(shells) + 1):
            shell_labels=labels
        else:
            print('Region labels not properly defined. Number of regions is different from number of labels.') 
            print('No region labels defined. Using default numerical identifiers.')
            
            for idx, value in enumerate(shells, 1):
                shell_labels.append(idx)
            shell_labels.append(idx+1)
        
        
        state_encodings = list(itertools.product([0, 1], repeat=len(shells)+1))

        state_indexes=[[] for i in range(len(state_encodings))]
        
        
        for i, encoding in enumerate(state_encodings):
            for f, value in enumerate(encoding):
                if value == 1:
                    state_indexes[i].append(shell_labels[f])
        for i, name in enumerate(state_indexes):
            state_indexes[i]=''.join(map(str,name))
            
        #print(state_names, state_encodings)
        
        return state_encodings, state_indexes
    
    
    @staticmethod
    def state_mapper(shells, df, labels=None):
        """
        

        Parameters
        ----------
        shells : TYPE
            DESCRIPTION.
        array : TYPE
            DESCRIPTION.
        labels : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """


        encodings, encoding_indexes = Functions.stateEncoder(shells, labels)

        def state_evaluator(x):
            x= tuple(x)
            if x in encodings: #.values():
                return encodings.index(x)

        #Discretize array values in state bins.                 
        
        #state_map=np.digitize(df, shells, right=True) 
        state_map = np.searchsorted(shells, df, side='right')
        state_comb=pd.DataFrame(index=df.index)
        
        
        for s in range(0, len(shells)+1):
            state_comb[s]=(state_map == s).any(1).astype(int)

        state_df=state_comb.apply(state_evaluator, raw=True, axis=1)
        
        return state_df
    
    @staticmethod
    def remap_trajectories_states(df, states):
        original_values, replaced_values, labels = [], [], []
        for idx,(index, label) in enumerate(states.items()):
            original_values.append(index)
            replaced_values.append(idx)
            labels.append(label)
        
        remap_states_df = df.replace(to_replace=original_values, value=replaced_values)
        
        remap_states = {r : l for r, l in zip(replaced_values, labels)}

        #print(df, remap_states_df)
        return remap_states_df, remap_states
    
    @staticmethod
    def density_stats(level_unique, stride, results, unique=None, method='histogram'):
        """
        
        Base function for 3D state-discretization.
        
        TODO: A lot.


        """
        
        import os
        #import path
        from gridData import Grid
        import numpy as np
        import pandas as pd
        if unique != None:
        
            dens=os.path.abspath(f'{results}/superposed_{level_unique}-it{unique}-s{stride}-Molar.dx')
    
        else:
            dens=os.path.abspath(f'{results}/superposed_{level_unique}-s{stride}-Molar.dx')
            
        g = Grid(dens)
        g_max=np.max(g.grid)
        g_min=np.min(g.grid)
        g_mean=np.median(g.grid)
        g_std=np.std(g.grid)
        
        quantiles=np.arange(0,1.01, 0.01)
        g_q=np.quantile(g.grid, quantiles)
        
        #prepare df
        col_index=pd.MultiIndex.from_tuples([(level_unique, unique)], names=['level', 'iterable'])        
        
        if method == 'means':
        
            columns_=['min', 'max', 'mean', 'std']
            data_grid=[[g_min, g_max, g_mean, g_std]]
            
            df=pd.DataFrame(data_grid, columns=columns_, index=col_index)

        
        if method == 'quantiles':
            
            columns_=quantiles
            data_grid=g_q
            
            df=pd.DataFrame(data_grid, columns=col_index)
            

            
        if method == 'histogram':   
        
            data_grid=g.grid.flat
            df=pd.DataFrame(data_grid, columns=col_index)
        
        return df


class XML:
    """
    
    
    """
    def __init__(self, input_file='input.xml'):
        self.input = input_file
    
    @staticmethod
    def get_root(self):
        
        import Path
        import os
        
        from lxml import etree as et
        #import xml.etree.ElementTree as et
        parser = et.XMLParser(recover=True)

        workdir=Path(os.getcwd())
        name=f'{workdir}\input.xml'
        #TODO: fix this for unix
        print(name)
        tree = et.parse(name, parser=parser)        
        root=tree.getroot()
        
        print(root.tag)
        
        for child in root:
            print(child.tag, child.attrib)
            
        #print([elem.tag for elem in root.iter()])
        
        for system in root.iter('system'):
            print(system.attrib)
            
        for description in root.iter('description'):
            print(description.text)
    
        for system in root.findall("./kind/interaction/system/[protein='CalB']"):
            print(system.attrib)
    
        return root

    def readXML(self, i_file='input.xml') -> object:
        """
        Parses the XML file describing the project.



        """
        
        from lxml import etree as et
        
        path_file=f'{self.workdir}\{i_file}'
        #TODO: fix this for unix
        
        print(path_file)
        tree = et.parse(path_file, parser=et.XMLParser(recover=True))   
        root=tree.getroot()
        
        return root
        
        
    def setParametersFromXML(self, params) -> dict:
        """
        


        """

        import objectify
        
        
        root=self.get_root()
        
        if root.tag != 'project':
            raise ValueError
            
        for child in root:
            kind=child.attrib
            print(f'child tag is {child.tag}: {kind}')
        
        print([elem.tag for elem in root.iter()])
            
        systems={}
            
        for system in root.iter('system'):
            attributes=system.attrib
            name=dict(attributes['title'])
            #name, status=attributes['title'], attributes['complete']
            systems.update(name)
        print(systems)
        
        for description in root.iter('description'):
            print(description.text)

        for system in root.findall("./kind/interaction/system/[protein='CalB']"):
            print(system.attrib)
        

        project=objectify.parse(path_file)
    
        return project
    
    
    def exportAs(self, format):
        """
        

        """
        
        if format == 'JSON':
            return self._serialize_to_json()
        elif format == 'XML':
            return self._serialize_to_xml()
        else:
            raise ValueError(format)
        
        
    def _serialize_to_json(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        import json
        
        project_info = {
            'protein': self.protein,
            'ligand': self.ligand,
            'parameter': self.parameter,
            'replicas': self.replicas,
            'name': self.name,
            'timestep':self.timestep,
        }
        return json.dumps(project_info)
        
    def _serialize_to_xml(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        from lxml import etree as et
        
        project_info = et.Element('project', attrib={'name': self.name})
        ligand = et.SubElement(project_info, 'ligand')
        ligand.text = self.ligand
        protein = et.SubElement(project_info, 'protein')
        protein.text = self.protein
        parameter = et.SubElement(project_info, 'parameter')
        parameter.text = self.parameter
        replicas = et.SubElement(project_info, 'replicas')
        replicas.text = self.replicas
        timestep = et.SubElement(project_info, 'timestep')
        timestep.text = self.timestep
            
        return et.tostring(project_info, encoding='unicode')
        
# =============================================================================
#     @staticmethod
#     def get_structure_specs():        
#         from scipy import stats 
#         import numpy as np
#         features = np.array(specs_systems).reshape(len(project.parameter), project.replicas, 7) 
#         fts = ['n. atoms', 'n. res', 'size', 'n. prot', 'n. lig', 'n. water', 'n. NA']
#         print('modes')
#         for idx, i in enumerate(features):
#             print(idx)
#             modes = stats.mode(i)
#             for mode in modes[::2]:
#                 for i, y in zip(mode.flat, fts):
#                     print(y, i)
#         print('min/max')
#         for i in features:
#         
#             for x, y in zip(range(7), fts):
#                 ft = i[:,x]
#                 print(y, ft.min(), ft.max())        
# =============================================================================

# =============================================================================
#     @staticmethod
#     def get_other_structure_specs():
#         import mdtraj as md
#         import re
#         
#         selections = ['name CA', f'name O07', 'water', 'name NA']
#         specs_systems = []
#         now = []
#         for name,system in project.systems.items():
#             for top in system.topology:
#                 if re.search(f'{project.protein[0]}.gro', top) or re.search(f'{project.protein[0]}-eq.gro', top):
#                     print(top)
#                     traj = md.load(top)
#                     specs = (traj.n_atoms, traj.n_residues, traj.unitcell_lengths[0][0])
#                     specs2 = [len(traj.topology.select(sel)) for sel in selections]
#                     specs = specs + tuple(specs2)
#                     print(specs)
#                     specs_systems.append(specs)
#                     break
#                 
#         print(specs_systems)
# =============================================================================

    
    
class Tasks:
    """
    
    Base class to job-specific tasks. 
    
    """
    
    env_name = 'SFv2'
    
    def __init__(self,
                 machine ='Local',
                 n_gpu=1,
                 run_time='47:30:00'):
        
        self.machine = machine
        self.n_gpu = n_gpu
        self.run_time = run_time
        
        
    @staticmethod
    def parallel_task(input_function, container, fixed, n_cores=-1, run_silent=True):
        """
        
        Core function for multiprocessing of a given task.Takes as input a function and an iterator.
        Warning: Spawns as many threads as the number of elements in the iterator.
        (Jupyter and Pandas compatible)
    

        Parameters
        ----------
        input_function : TYPE
            DESCRIPTION.
        container : TYPE
            DESCRIPTION.
        fixed : TYPE
            DESCRIPTION.
        n_cores : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """

        
        import psutil
        from functools import partial
        from multiprocessing import Pool
                
        if n_cores <=0:
            num_cpus = psutil.cpu_count(logical=True)
        else:
            num_cpus = n_cores
        process_pool = Pool(processes=num_cpus)
        #try:    
        if not run_silent:   
            print(f'Performing {input_function.__name__} tasks for {len(container)} elements on {num_cpus} logical cores.', end='\r')
        calc=partial(input_function, specs=fixed)                     
        out=list(process_pool.map(calc, container))
        if not run_silent:
            print(f'Pooling results of {len(out)} tasks..', end='\r')
        process_pool.close()
        process_pool.join()
        
        return out

        
    @staticmethod    
    def run_bash(command, input_, args='', mode=None):
        

        file_path, file_name_ = os.path.split(input_) 
        file_name, extension = file_name_.split('.')
        job_file=f'{file_path}/job.sh'

        
        job_string = f'{command} {args} {input_}'
        

        
        with open(job_file, 'w') as f:
            f.write(
                    f'''
                    cd {file_path}
                    {job_string}
                    exit 0
                    ''')
        f.close()
        
        encoding = 'UTF-8'

        if mode == 'create':
            chmod = f'chmod a+rwx {job_file}'
            subprocess.call(chmod.split())
            process = subprocess.call(job_file, shell=True)
            return process

        else:
            process = subprocess.Popen(job_string.split(), stdout=subprocess.PIPE) 
            output, error = process.communicate()
            
            if mode == 'pipe':
                return output
            elif mode == 'out':
                try:
                    out_file_name = f'{file_path}/{file_name}_{command.split("_")[1]}.{extension}'
                except IndexError:
                    out_file_name = f'{file_path}/{file_name}_{command}.{extension}'
   
                out_pdb = str(output, encoding)
                f=open(out_file_name, "w")
                f.write(out_pdb)
                f.close()
                
                return out_file_name
            else:
                print(f'{error}\n{str(output, encoding)}')
    
    
        
        
    def setMachine(self, gpu_index_='0', force_platform=False):
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
        
        avail=openmmtools.utils.get_available_platforms()
        
        for idx, plat in enumerate(avail, 1):
            print(f'\tAvaliable platform {idx}: {plat.getName()}')
            
        fastest=openmmtools.utils.get_fastest_platform()
        if force_platform:
            self.platform = omm.Platform.getPlatformByName('CUDA')
            self.platformProperties = {'Precision': 'mixed', 'DeviceIndex': gpu_index_}
        
        #print('Fastest: ', fastest.getName())
        else:
            if fastest.getName() == 'CUDA':
                self.platform = fastest
                self.platformProperties = {'Precision': 'mixed',
                                  'DeviceIndex': gpu_index_}
                
                print('Using CUDA as fastest platform: ', gpu_index_)
                
            else:
                print('Cannot set CUDA as fastest platform.')
                try:
                    self.platform = omm.Platform.getPlatformByName('CUDA')
                    self.platformProperties = {'Precision': 'mixed', 'DeviceIndex': gpu_index_}
                    print('Warning. Using a CUDA device found by openMM, maybe its not the best.')
                except:
                    try:
                        self.platform = omm.Platform.getPlatformByName('OpenCL')
                        self.platformProperties = None
                    except:
                        self.platform = omm.Platform.getPlatformByName('CPU')
                        self.platformProperties = None
            
                print(f"Using platform {self.platform.getName()}")
                
        return self.platform, self.platformProperties

    

        
    def getSpecs(self, machine):
        
        
        #Automate fetch partition.
        
        hosts = {'BwUniCluster' : {'scheduler' : 'SBATCH',
                           'cpu_gpu' : 1, 
                           'gpu_node' : self.n_gpu, 
                           'req_gpu' : f'gpu:{self.n_gpu}', 
                           'n' : 1, 
                           'mem' : '2gb', 
                           'time' : self.run_time,
                           'chain_jobs' : 2},
                 
                 'Local' : {'n_gpu' : self.n_gpu,
                            'gpu_index' : 0,
                            'time' : self.run_time}}
        #'partition' : 'gpu_4', #set in __init__
        #'script_path' : '/pfs/work7/workspace/scratch/st_ac131353-SETD2-0/SETD2/',
        try:
            return hosts[machine]
        except KeyError:
            print('Try: ', hosts.keys()) 
    
    def setSpecs(self, **kwargs):
        
        specs = self.getSpecs(self.machine)  
        print('Machine set: ') 
        for k, v in specs.items():
            self.__dict__[k] = v
        
        
        try:
            for attribute, value in kwargs.items():
                self.__dict__[attribute] = value
        except AttributeError:
            pass
        
        print(self.__dict__)            

    def generateSripts(self, name, workdir, replicate, compute_time, **kwargs):
        
        self.setSpecs(**kwargs)
        self.replicate = replicate
        chain_file = os.path.abspath(f'{workdir}/{name}_chain.sh')
        job_file = os.path.abspath(f'{workdir}/{name}.sh')

        try:
            process = subprocess.Popen('ws_find', self.ws_name, stdout=subprocess.PIPE) 
            output, error = process.communicate()
            self.script_path = process
            print('Script path modified: ', self.script_path)
        except AttributeError as v:
            print('Attribute not set: ', v)
        except Exception as vv:
            print(vv.__class__, vv)
            self.script_path = f'/pfs/work7/workspace/scratch/st_ac131353-SETD2/{self.ws_name}/'
        print('Workspace: ', self.script_path)
        with open(job_file, 'w') as f:
            f.write(
f'''#!/bin/bash

#{self.scheduler} -J {name}
#{self.scheduler} --cpus-per-gpu={self.cpu_gpu}
#{self.scheduler} --gpus-per-node={self.gpu_node}
#{self.scheduler} --gres={self.req_gpu}
#{self.scheduler} --ntasks={self.n}
#{self.scheduler} --mem-per-cpu={self.mem}
#{self.scheduler} --time={self.time}
#{self.scheduler} --partition={self.partition}


set +eu
module purge
module load compiler/pgi/2020
module load devel/cuda/11.0
module load devel/miniconda
eval "$(conda shell.bash hook)"
conda activate {Tasks.env_name}
set -eu
echo "----------------"
echo {name}
echo $(date -u) "Job was started"
echo "----------------"

python {self.script_path}/{self.notebook_name}.py {self.replicate} {compute_time}
exit 0''')
        


        
# =============================================================================
#         except Exception as v:
#             print(v, v.__class__)
#             process_pool.terminate()
#             raise v.__class__()            
# =============================================================================
