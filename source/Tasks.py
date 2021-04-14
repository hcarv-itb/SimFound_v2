# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:48:16 2021

@author: hcarv
"""

class Tasks:
    
    
    def parallel_task(input_function=None, collection=None):
        """Core function for multiprocessing of a given task.Takes as input a function and an iterator.
        Warning: Spawns as many threads as the number of elements in the iterator."""
        
        
        from pathlib import Path
        import multiprocessing
        from multiprocessing import Manager, Process

        n_cpus=multiprocessing.cpu_count()
        temp_dir='/tmp'
        
        
        
        def task_exec(manager, input_function, idx, c):
            """Execution of the input function. Output is stored in the manager, which is handled to the job"""
            
            task=manager
            function=input_function
            task[idx]=function(c)           
        
        if input_function != None and collection != None:
        
            manager=Manager().list()
            job=[Process(target=task_exec, args=(manager, input_function, idx, c)) for idx, c in enumerate(collection)]
            _=[p.start() for p in job]
            _=[p.join() for p in job]
       
            return manager
        
        else:
            print('No input function or container provided')