#!/bin/bash

#SBATCH --array=1-915
#SBATCH --time=30   # If only one number, it is assumed to be minutes
#SBATCH --ntasks=1
#SBATCH --partition=sandy,west
#SBATCH --mem-per-cpu=2G
#SBATCH -J "p_pdm_kolling"
#SBATCH --mail-user=dario.cuevas_rivera@tu-dresden.de
#SBATCH --mail-type=END
#SBATCH -A p_pdm
#SBATCH --output=./data/logs/slurm-%A_%a.out

SUBJNUMBER=1
SHAPE='unimodal_s'
module load python/3.5.2-anaconda4.2.0
python /home/cuevasri/kolling/actinf/invert_parameters.py -v -i $((SLURM_ARRAY_TASK_ID-1)) --shape $SHAPE $SUBJNUMBER
