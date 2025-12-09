#!/bin/bash
#SBATCH -n 10
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH -J S_Coherence
#SBATCH --output=/Path_to_where_you_want_output/S_Coherence_output.out

module purge

module load ffmpeg/6.0
module load hpcx-mpi/4.1.5rc2s
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate dedalus3

mpiexec -n 10 python3 /path_to_your_3D_S_Coherence/3D_S_Coherence.py