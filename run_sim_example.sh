#!/bin/bash
#SBATCH --array=0-19 
#SBATCH -n 64 
#SBATCH -t 48:00:00
#SBATCH --mem=492G 
#SBATCH -J LAPD_SIM_ARRAY
#SBATCH --output=/path_to_your_output/LAPD_SIM_ARRAY_%A_%a.out


module purge
module load hpcx-mpi/4.1.5rc2s
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate dedalus3

export FFTW_NO_WISDOM=1 
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
INDEX=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

echo "Starting simulation for array index: $INDEX (Task ID $SLURM_ARRAY_TASK_ID)"

mpiexec -n 64 python3 /path_to_your_3D_S_Retry.py/3D_S_Retry.py $INDEX