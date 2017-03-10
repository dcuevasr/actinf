#!/bin/bash

if [ ! $# == 1 ];
then
    echo "need exactly one argument"
    exit
fi

for s in $1;
do
    sed "s/SUBJNUMBER=[0-9]*/SUBJNUMBER=$s/" < single_job.sh > single_temp.sbatch
    sbatch single_temp.sbatch
done
