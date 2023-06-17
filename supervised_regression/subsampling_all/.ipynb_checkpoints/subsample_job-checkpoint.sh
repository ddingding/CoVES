#!/bin/bash
#SBATCH -c 1
#SBATCH -t 5:00:00
#SBATCH -p medium
#SBATCH --mem=50G
#SBATCH -o /n/groups/marks/users/david/ex62/out/runs/%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e /n/groups/marks/users/david/ex62/out/runs/%j.err 

source activate /home/dd128/mambaforge/envs/dd_tf2_mamba
module load gcc/6.2.0
module load cuda/11.2

cd /n/groups/marks/users/david/github/pareSingleLibrary2/codebase/pairedEnd/ex62_gvp/subsampling_all/

python3 /n/groups/marks/users/david/github/pareSingleLibrary2/codebase/pairedEnd/ex62_gvp/subsampling_all/subsample_one_ds.py $1