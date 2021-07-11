# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:41:53 2021

@author: hcarv
"""


#SFv2
try:
    import Trajectory
except:
    pass


import mlflow
from mlflow import log_artifact
import mlflow.sklearn

import functools
import os

import pandas as pd




# =============================================================================
def calculator(func, *args):
     
     @functools.wraps(func)
     def wrapper(system_specs, specs):
         
        print('Calculator')
         
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, multi, package)=specs
        
        
        if package == 'MDAnalysis':
            traj, status=Trajectory.Trajectory.loadTraj(topology, trajectory, name)
    
        elif package == 'MDTraj':
            
            traj, status=Trajectory.Trajectory.loadMDTraj(topology, trajectory, name)
    
        #Only work with managable simulations
        indexes=[[n] for n in name.split('-')]
        if status == ('full' or 'incomplete'):
        
            #Get values from task.
            rows_values, columns_values=func
            
            if multi:
                multi_len=len(columns_values)
                pairs=[i for i in range(1, multi_len+1)]
                indexes.append(pairs)
            indexes.append(units_x, units_y)
            
            #Build output dataFrame
            names=[f'l{i}' for i in range(1, len(indexes)+1)]
            column_index=pd.MultiIndex.from_product(indexes, names=names)    
            rows=pd.Index(rows_values.bins, name=units_x)
            df_system=pd.DataFrame(columns_values, columns=column_index, index=rows)
            #df_system=df_system.mask(df_system > 90)
            
            return df_system
    
        else:
            return pd.DataFrame()
        
 
     return wrapper






def MLflow_draft(func, *args):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('MLflow')
        print(os.getcwd())
        
        with mlflow.start_run():
            
            print('Tracking :', func.__name__)
            
            out=func(*args, **kwargs)
            
            print(out)
            
            for param in out:
                
                print(param)
# =============================================================================
#                 mlflow.log_param('x': param)
#                 mlflow.sklearn.log_model(func.__name__, "model")
#             
# =============================================================================

            # Log an artifact (output file)
            with open("output.txt", "w") as f:
                f.write(str(out[0]))
            log_artifact("output.txt")
            
        print("Something is happening after the function is called.")
        return func(*args, **kwargs)
        
    return wrapper


def MLflow(_func=None,*, experiment_name=None, tracking_uri=None, autolog=False, run_name=None, tags=None):
    
    def experiment_decorator(func):
    
        @functools.wraps(func)
        def experiment_wrapper(*args, **kwargs):
            nonlocal experiment_name, tracking_uri

            if tracking_uri is None:
                tracking_uri = os.getenv("MLFLOW_TRACKING_SERVICE_URI", mlflow.get_tracking_uri())
            mlflow.set_tracking_uri(tracking_uri)

            if experiment_name is None:
                experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
                
            experiment_id = (mlflow.set_experiment(experiment_name) if experiment_name is not None else None)

            if autolog:
                mlflow.autolog()

            with mlflow.start_run(experiment_id=experiment_id, 
                                  run_name=run_name, 
                                  tags=tags):
                
                value = func(*args, **kwargs)

            return value

        return experiment_wrapper

    if _func is None:
        return experiment_decorator
    else:
        return experiment_decorator(_func)


    