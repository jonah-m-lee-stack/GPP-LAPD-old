#!/bin/bash
#SBATCH -n 10
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH -J S_data
#SBATCH --output=/path_to_where_you_want_output/%A_%a.out
#SBATCH --array=0-19 # Adjust this range according to how many simulation indices you have run in ColdMagnetizedPlasma.py (same as num_points on line 36)

module purge

module load ffmpeg/6.0
module load hpcx-mpi/4.1.5rc2s
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate dedalus3


formatted_id=$SLURM_ARRAY_TASK_ID
mpiexec  -n 10 python3 /path_to_your_data.py/data.py $formatted_id
