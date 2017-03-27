#!/bin/bash

# Runs a loop over jobs to simulate what SLURM would do. I think

for i in $( seq 1 5 )
do
    python invert_parameters.py -v -i $i 9 &>./data/logs/out_$i.log &
    echo $!
done

