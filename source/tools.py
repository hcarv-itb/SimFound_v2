"""
Created on Fri Nov  6 21:50:14 2020

@author: hcarv

"""

class Functions:
    
    @staticmethod
    def fileHandler(path_or_file, confirmation=False, _new=False, *args):
        """
        


        """
        
        import os
        
    
        #print(f'Handling: {path_or_file}')
        
        for item in path_or_file:
            if not os.path.exists(item):
                os.makedirs(item)
                print(f'Item created: {item}')
                
                return item
                
            else:
                #print(f'Item exists: {item}')
                if confirmation:
                    if input("Create _new? (True/False)"):
                        os.makedirs(item+'_new', exist_ok=True)
                    else:
                        pass
                elif _new == True:
                    os.makedirs(item+'_new', exist_ok=True)    
                    return item+'_new'
                else:
                    return item
                

        


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
    
    Base class to spwan multiproc (Jupyter and Pandas compatible)
    
    
    """
    
    def parallel_task(input_function, container, fixed, n_cores=-1):
        """Core function for multiprocessing of a given task.Takes as input a function and an iterator.
        Warning: Spawns as many threads as the number of elements in the iterator."""
        
        import psutil
        from functools import partial
        from multiprocessing import Pool
                
        if n_cores <=0:
            num_cpus = psutil.cpu_count(logical=True)
        else:
            num_cpus = n_cores
            
        print(f'Performing {input_function} tasks for {len(container)} elements on {num_cpus} logical cores')
        
        process_pool = Pool(processes=num_cpus)
        
        calc=partial(input_function, specs=fixed)                     
        out=list(process_pool.map(calc, container))

        print(f'Pooling results of {len(out)} tasks..')
        process_pool.close()
        process_pool.join()
        

        
        #print(out)
        return out
    
    
# =============================================================================
#     
#     def parallel_task(input_function=None, collection=None):
#         """
#         
#         Core function for multiprocessing of a given task.Takes as input a function and an iterator.
#         Warning: Spawns as many threads as the number of elements in the iterator.
#             
# 
#         """
#         
#         
#         from pathlib import Path
#         import multiprocessing
#         from multiprocessing import Manager, Process
# 
#         n_cpus=multiprocessing.cpu_count()
#         temp_dir='/tmp'
#         
#         
#         
#         def task_exec(manager, input_function, idx, c) -> dict:
#             """
#             Execution of the input function. Output is stored in the manager, which is handled to the job
# 
#             Parameters
#             ----------
#             manager : TYPE
#                 DESCRIPTION.
#             input_function : TYPE
#                 DESCRIPTION.
#             idx : TYPE
#                 DESCRIPTION.
#             c : TYPE
#                 DESCRIPTION.
# 
# 
#             """
#             
#             task=manager
#             function=input_function
#             task[idx]=function(c)           
#         
#         if input_function != None and collection != None:
#         
#             manager=Manager().list()
#             job=[Process(target=task_exec, args=(manager, input_function, idx, c)) for idx, c in enumerate(collection)]
#             _=[p.start() for p in job]
#             _=[p.join() for p in job]
#        
#             return manager
#         
#         else:
#             print('No input function or container provided')
#             
#             
# =============================================================================
    
    
