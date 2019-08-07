#!/bin/bash

for ptarget in 32
do
    for ptimes in 5
    do
        /home/server5/anaconda3/envs/lst/bin/python /home/server5/lst/rethinking-network-pruning/cifar/analyse/ga.py --ptarget ${ptarget} --ptimes ${ptimes} --save_name "${ptarget}_${ptimes}.xls"
    done
done
