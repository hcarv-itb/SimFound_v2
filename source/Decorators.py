# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:41:53 2021

@author: hcarv
"""


#SFv2
try:
    import Featurize
    import Trajectory
except:
    pass


import mlflow
from mlflow import log_artifact
import mlflow.sklearn

import functools
import os

import pandas as pd
import numpy as np




# =============================================================================
def calculator(func):
     
     @functools.wraps(func)
     def wrapper(system_specs, specs):
         
        print('Calculator')
        (trajectory, topology, results_folder, name)=system_specs
        (selection,  start, stop, timestep, stride, units_x, units_y, task, store_traj, subset)=specs   
        names, indexes, column_index=Featurize.Featurize.df_template(system_specs, unit=[units_y])
        traj=Trajectory.Trajectory.loadTrajectory(system_specs, specs)
        

        if traj != None and topology != None:
            
            result=func(selection, traj, start, stop, stride)
            rows=pd.Index(np.arange(0, len(result), stride)*timestep, name=units_x)
            df_system=pd.DataFrame(result, columns=column_index, index=rows)
    
        else:
            df_system = pd.DataFrame()
        
        return df_system
 
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


    