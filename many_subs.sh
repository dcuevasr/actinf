#!/bin/bash

if [ ! $# == 1 ] && [ ! $# == 2 ];
then
    echo "One or two arguments"
    exit
fi

ARRAY="1-915"
if [ $# -gt 1 ]; then
    echo 'yo'
    if [ $2 == 'sigmoid_s' ]; then
	ARRAY="1-465"
    fi
    if [ $2 == 'exponential' ]; then
	ARRAY="1-48"
    fi
fi 

for s in $1;
do
    #sed "s/SUBJNUMBER=[0-9]*/SUBJNUMBER=$s/" < single_job.sh | sed "s/SHAPE=\'[_A-Za-z]\'/SHAPE=$2/" > single_temp.sbatch
    sed "s/SUBJNUMBER=[0-9]*/SUBJNUMBER=$s/" < single_job.sh > single_temp.sbatch
    if [ ! -z $2 ]
    then
   	sed -i "s/SHAPE='[_A-Za-z]*'/SHAPE='$2'/" single_temp.sbatch
	sed -i "s/1-915/$ARRAY/" single_temp.sbatch
    fi
    sbatch single_temp.sbatch
done
