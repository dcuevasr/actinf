#!/bin/bash
# Splits the whole job for a single subject into mini-jobs that run in
# parallel. Echoes back the pids.

JOBS=4

MUINI=-15
MUEND=45
SDINI=1
SDEND=15

SUBJECT=0


STP=$((($MUEND-$MUINI)/$JOBS))

#echo "Activating anaconda environment"
source activate actinf

for j in $( seq $(($MUINI+$STP)) $STP $MUEND )
do
    CMUINI=$(($j - $STP))
    CMUEND=$j
    python invert_parameters.py $CMUINI $CMUEND $SDINI $SDEND $SUBJECT &>/dev/null &
    PIDS[j]=$!
done

echo ${PIDS[*]}
