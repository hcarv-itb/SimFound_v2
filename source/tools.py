"""
Created on Fri Nov  6 21:50:14 2020

@author: hcarv

"""

import os
import simtk.openmm as omm
import simtk.unit as unit
import pandas as pd

class Functions:
    
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

            try:
                scalar=float(str(p).split('K')[0])*unit.kelvin
                #self.parameter_scalar.append(scalar)
                #parameter_label.append(str(scalar).replace(" ",""))
                print(f'Converted parameter "temperature" (in K) into scalar: {scalar}')
            except ValueError:
                try:
                    scalar=float(str(p).split('M')[0])*unit.molar
                except:
                    scalar=float(str(p).split('mM')[0])/1000*unit.molar
                #print(f'Converted parameter "concentration" (in Molar) into scalar: {scalar}')
            except:
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

        
        try:
            if (len(labels) == len(regions) + 1):
                region_labels=labels
        except TypeError:
            print('Region labels not properly defined. Number of regions is different from number of labels.') 
            region_labels=None 
                
        if region_labels == None:
            print('No region labels defined. Using default numerical identifiers.')
            region_labels=[]
            for idx, value in enumerate(regions, 1):
                region_labels.append(idx)
            region_labels.append(idx+1)
            
        return region_labels 
    
    @staticmethod
    def sampledStateLabels(shells, sampled_states=None, labels=None):
        """
        Static method that generates state labels for sampled states.
        Requires the *regions* to be provided and optionally the *sampled_states* and *labels*.



        """
      
        
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

        """
         
        import itertools
        
        
        region_labels=Functions.regionLabels(shells, labels)
        state_encodings = list(itertools.product([0, 1], repeat=len(shells)+1))

        state_names=[[] for i in range(len(state_encodings))]
        
        for i, encoding in enumerate(state_encodings):
            for f, value in enumerate(encoding):
                if value == 1:
                    state_names[i].append(region_labels[f])
        for i, name in enumerate(state_names):
            state_names[i]=''.join(map(str,name))
        
        return state_encodings, state_names  
    
    
    @staticmethod
    def state_mapper(shells, array, labels=None):
        """


        """

        
        import numpy as np
        import pandas as pd
  
        def state_evaluator(x):
            """
 
            
            """
            for c,v in enumerate(Functions.stateEncoder(shells, labels)[0]): #Call to stateEncoder[0] = state_encodings
                if np.array_equal(v, x.values):
                    return c 
                         

        state_map=np.digitize(array, shells, right=True) #Discretize array values in state bins.
            #state_map=NAC.applymap(lambda x:np.digitize(x, states, right=True)) #Deprecated. Much slower.
 
                
        state_comb=pd.DataFrame()
        for s in range(0, len(shells)+1):
            state_comb[s]=(state_map == s).any(1).astype(int)
        state_df=state_comb.apply(state_evaluator, axis=1)


        return state_df.to_frame()
    
    
    def get_descriptors(input_df, level, iterable, describe='sum', quantiles=[0.1, 0.5, 0.99]):
        """
        Get properties of input_df.

        Parameters
        ----------
        input_df : TYPE
            DESCRIPTION.
        level : TYPE
            DESCRIPTION.
        describe : TYPE, optional
            DESCRIPTION. The default is 'sum'.
        quantiles : TYPE, optional
            DESCRIPTION. The default is [0.1, 0.5, 0.99].

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        pairs : TYPE
            DESCRIPTION.
        replicas : TYPE
            DESCRIPTION.
        molecules : TYPE
            DESCRIPTION.
        frames : TYPE
            DESCRIPTION.

        """
        """
        
        
        """
        
        
        #print(f'Descriptor of: {describe}')
   
        df_it=input_df.loc[:,input_df.columns.get_level_values(f'l{level+1}') == iterable] #level +1 due to index starting at l1

        pairs=len(df_it.columns.get_level_values(f'l{level+2}'))
        replicas=len(df_it.columns.get_level_values(f'l{level+2}').unique()) #level +2 is is the replicates x number of molecules
        molecules=int(pairs/replicas)
        
        frames = 0
        #Access the replicate level
        for i in df_it.columns.get_level_values(f'l{level+2}').unique():         
            sub_df = df_it.loc[:,df_it.columns.get_level_values(f'l{level+2}') == i]
            frames_i=int(sub_df.sum(axis=0).unique())
            frames += frames_i
            
        
        frames=int(df_it.sum(axis=0).unique()[0])


        #print(f'Iterable: {iterable}\n\tPairs: {pairs}\n\treplicas: {replicas}\n\tmolecules: {molecules}\n\tCounts: {total}')

        if describe == 'single':
            descriptor=df_it.quantile(q=0.5)/frames*molecules

        elif describe == 'mean':
            descriptor=pd.DataFrame()
            mean=df_it.mean(axis=1)/frames*molecules
            mean.name=iterable
            sem=df_it.sem(axis=1)/frames*molecules    
            sem.name=f'{iterable}-St.Err.'
            descriptor=pd.concat([mean, sem], axis=1)
        
        elif describe == 'quantile':
            descriptor=pd.DataFrame()                    
            for quantile in quantiles:        
                descriptor_q=df_it.quantile(q=quantile, axis=1)/frames*molecules
                descriptor_q.name=f'{iterable}-Q{quantile}'
                descriptor=pd.concat([descriptor, descriptor_q], axis=1)
                
        
            #print(descriptor)
            #descriptor.plot(logy=True, logx=True)
    
        else: #TODO: make more elif statements if need and push this to elif None.
            descriptor=df_it
    
        descriptor.name=iterable           
        
        return descriptor
    

    
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
        
        
        

    
    
class Tasks:
    """
    
    Base class to job-specific tasks. 
    
    """
    
    env_name = 'SFv2'
    
    def __init__(self,
                 machine ='Local',
                 n_gpu=1,
                 run_time='12:00:00'):
        
        self.machine = machine
        self.n_gpu = n_gpu
        self.run_time = run_time
        
        
        
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
                
        return self
    

    

        
    def getSpecs(self, machine):
        
        
        #Automate fetch partition.
        
        hosts = {'BwUniCluster' : {'scheduler' : 'SBATCH',
                           'cpu_gpu' : 1, 
                           'gpu_node' : self.n_gpu, 
                           'req_gpu' : f'gpu:{self.n_gpu}', 
                           'n' : 1, 
                           'mem' : '2gb', 
                           'time' : self.run_time,
                           'partition' : 'gpu_4',
                           'script_path' : '/pfs/work7/workspace/scratch/st_ac131353-SETD2-0/SETD2/',
                           'chain_jobs' : 2},
                 
                 'Local' : {'n_gpu' : self.n_gpu,
                            'gpu_index' : 0,
                            'time' : self.run_time}}
        try:
            return hosts[machine]
        except KeyError:
            print('Try: ', hosts.keys()) 
    
    def setSpecs(self, *args):
        

        specs = self.getSpecs(self.machine)

            
        print('Machine set: ') 
        for k, v in specs.items():
            self.__dict__[k] = v
        
        for arg_tuple in args:
            
            print('adding ', arg_tuple)
            self.__dict__[arg_tuple[0]] = arg_tuple[1]
        
        print(self.__dict__)            
        return self

    def generateSripts(self, name, workdir, *args):
        
        self.setSpecs(*args)
        
        chain_file = os.path.abspath(f'{workdir}/{name}_chain.sh')
        job_file = os.path.abspath(f'{workdir}/{name}.sh')
        
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
conda activate {self.env_name}
set -eu
echo "----------------"
echo {name}
echo $(date -u) "Job was started"
echo "----------------"

python {self.script_path}/SFv2_standAlone.py {name}
exit 0''')

    @staticmethod
    def parallel_task(input_function, container, fixed, n_cores=-1):
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
            
        print(f'Performing {input_function.__name__} tasks for {len(container)} elements on {num_cpus} logical cores.')
        
        process_pool = Pool(processes=num_cpus)
        
        calc=partial(input_function, specs=fixed)                     
        out=list(process_pool.map(calc, container))

        print(f'Pooling results of {len(out)} tasks..')
        process_pool.close()
        process_pool.join()
        
        #print(out)
        return out
