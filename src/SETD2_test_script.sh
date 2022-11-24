#!/bin/bash
#SBATCH -J 500-25-293.15-1-500_eq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=10
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_4

set +eu

module purge
module load compiler/pgi/2020
module load devel/cuda/11.0
module load devel/miniconda
eval "$(conda shell.bash hook)"
conda activate sfv2

set -eu

echo "----------------"
echo $(date -u) "Job was started"
echo "----------------"
                    
python /pfs/work7/workspace/scratch/st_ac131353-SETD2-0/SETD2/SETD2-HPC.py 
exit 0