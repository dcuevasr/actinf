#/bin/bash
source activate actinfth

for X in $(seq 0 839);
do
    python invert_parameters.py -v -i $X 0 >> log_all.txt;
done
