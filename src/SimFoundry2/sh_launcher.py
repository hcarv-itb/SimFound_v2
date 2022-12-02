# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:13:07 2021

@author: Benjamin Schmitz
"""

def genSHs(self, 
           env_name='openmmfin', 
           scheduler='SBATCH', 
           max_time_in_h=48, 
           tasks=10, 
           partition='gpu_4', 
           chain_num=5, 
           dep_type='afternotok', 
           workspace='workspace', 
           script_location='workspace/scripts'):
    
    """
    
    Creates shellscript files for each simulation to be run at bwUniCluster 2.0.
    
    Args:
        env_name (str, optional): The name of the anaconda environment on the cluster. Defaults to 'openmmfin'.
        scheduler (str, optional): The scheduler the cluster uses. Defaults to 'SBATCH'.
        max_time_in_h (int, optional): The requested duration of the simulations on the cluster. Defaults to 48.
        tasks (int, optional): The requested number of tasks. Defaults to 10.
        partition (str, optional): The partition that is requested on the cluster. Defaults to 'gpu_4'.
        chain_num (int, optional): The number of chain jobs that should be submitted. Defaults to 5.
        dep_type (str, optional): The wanted dependency type of the sbatch submitter for chain jobs. Defaults to 'afternotok'.
        workspace (str, optional): The workspace on the cluster. Defaults to 'workspace'.
        script_location (str, optional): The location of the script on the cluster. Defaults to 'workspace/scripts'.
    """
    # compiler/pgi/2020 contains nvcc
    psteps = int((self.pdur/self.ptimestep)*(10**6))
    pfric = self.pfric
    for x in self.equilibrations:
        with open(f'{self.shdir}/{x.name}.sh', 'w') as of:
            of.write(
f'''#!/bin/bash
#{scheduler} -J {x.name}_eq
#{scheduler} --gres=gpu:1
#{scheduler} --ntasks={tasks}
#{scheduler} --time={max_time_in_h}:00:00
#{scheduler} --partition={partition}
set +eu
module purge
module load compiler/pgi/2020
module load devel/cuda/11.0
module load devel/miniconda
eval "$(conda shell.bash hook)"
conda activate {env_name}
set -eu
echo "----------------"
echo {x.name}
echo $(date -u) "Job was started"
echo "----------------"

python {script_location}/clustersim.py {workspace} {x.device} {x.temp} {x.fric} {x.forcefield} {x.name} {x.platname} {x.presscoup} {x.press} {x.steps} {x.mon} {x.timestep} {x.fric2} {psteps} {self.ptimestep} {x.boxname} {x.deltaav} {self.repfreq} {pfric} {x.steps2}
exit 0'''
)
    with open(f'{self.shdir}/chainsubmitter.sh', 'w') as f:
        f.write(
'''#!/bin/bash
######################################################
## submitter script for chain jobs ##
######################################################
## Define maximum number of jobs via positional parameter 1, default is 5
max_nojob=${1:-5}
## Define location of the job scripts 
## as list of strings?
chain_link_job=${2:-${PWD}/shscript.sh}
## Define type of dependency via positional parameter 2, default is 'afterok'
dep_type="${3:-afternotok}"
## Define the queue you want to use
queue=${4:-gpu_4}
myloop_counter=1
while [ ${myloop_counter} -le ${max_nojob} ] ; do
## Differ msub_opt depending on chain link number
if [ ${myloop_counter} -eq 1 ] ; then
slurm_opt=""
else
slurm_opt="-d ${dep_type}:${jobID}"
fi
## Print current iteration number and sbatch command
echo "Chain job iteration = ${myloop_counter}"
echo "sbatch --export=myloop_counter=${myloop_counter} ${slurm_opt} ${chain_link_job}"
## Store job ID for next iteration by storing output of sbatch command with empty lines
jobID=$(sbatch -p ${queue} --export=ALL,myloop_counter=${myloop_counter} ${slurm_opt} ${chain_link_job} 2>&1 | sed 's/[S,a-z]* //g')
## Check if ERROR occured
if [[ "${jobID}" =~ "ERROR" ]] ; then
echo " -> submission failed!" ; exit 1
else
echo " -> job number = ${jobID}"
fi
## Increase counter
let myloop_counter+=1
done'''
)
    with open(f'{self.shdir}/submit.sh', 'w') as f:
        f.write(f'''#!/bin/bash\n''')
    for x in self.equilibrations:
        with open(f'{self.shdir}/submit.sh', 'a') as f:
            f.write(f'''bash chainsubmitter.sh {chain_num} {workspace}/shfiles/{x.name}.sh {dep_type} {partition}\n''')