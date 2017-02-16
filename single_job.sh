#!/bin/bash

#SBATCH --array=1-900
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --partition=sandy,west
#SBATCH --mem=2G
#SBATCH -J "p_pdm_kolling"
#SBATCH --mail-user=dario.cuevas_rivera@tu-dresden.de
#SBATCH --mail-type=FAIL,END
#SBATCH -A p_pdm
#SBATCH --output=slurm-%A_%a.out

python invert_parameters -i $((SLURM_ARRAY_TASK_ID-1))
