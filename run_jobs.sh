
	#!/bin/bash
# Splits the whole job for a single subject into mini-jobs that run in
# parallel. Echoes back the pids.

function finish {
    for p in ${PIDS[@]}
    do
	kill -SIGTERM $p
    done
    echo "Sent SIGINT to python processes\n Do not panic! They are doing their stuff. Just wait."
}


JOBS=4

MUINI=-15
MUEND=45
SDINI=1
SDEND=15

SUBJECT=2


STP=$((($MUEND-$MUINI)/$JOBS))


source activate actinfth

count=0
for j in $( seq $(($MUINI+$STP)) $STP $MUEND )
do
    CMUINI=$(($j - $STP))
    CMUEND=$j
    #mafi='./data/logs/out_$(date +%s).pi'
    python invert_parameters.py -v -m $CMUINI $CMUEND -s $SDINI $SDEND $SUBJECT &>"./data/logs/out_$(date +%N).log" &
    PIDS[count]=$!
    count+=1
done
subshell_pid=$!

trap finish SIGINT SIGTERM
echo 'Waiting for processes to finish...\n'
for p in ${PIDS[@]}
do
    wait $p
done



