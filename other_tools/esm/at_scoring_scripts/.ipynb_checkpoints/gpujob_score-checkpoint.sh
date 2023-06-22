#!/bin/bash
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=30G
#SBATCH -p gpu,gpu_marks
#SBATCH --gres=gpu:1
#SBATCH -o /n/groups/marks/users/david/esm_if/data/seq_to_score/runs/%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e /n/groups/marks/users/david/esm_if/data/seq_to_score/runs/%j.err 

source activate dd_torch
module load gcc/6.2.0
module load cuda/11.2
 
python3 /n/groups/marks/users/david/esm_if/score_one_df.py $1