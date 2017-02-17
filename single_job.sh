#!/bin/bash

#SBATCH --array=1-5
#SBATCH --time=10   # If only one number, it is assumed to be minutes
#SBATCH --ntasks=1
#SBATCH --partition=sandy,west
#SBATCH --mem-per-cpu=2G
#SBATCH -J "p_pdm_kolling"
#SBATCH --mail-user=dario.cuevas_rivera@tu-dresden.de
#SBATCH --mail-type=FAIL,END
#SBATCH -A p_pdm
#SBATCH --output=slurm-%A_%a.out

SUBJNUMBER=0
module load python/3.5.2-anaconda4.2.0
python /projects/p_pdm/invert_parameters -v -i $((SLURM_ARRAY_TASK_ID-1)) $SUBJNUMBER
