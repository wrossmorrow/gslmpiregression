#!/bin/bash

K=5

PROCESS=( 2 4 6 8 )
OBSVERS=( 100 1000 10000 100000 1000000 )
for P in "${PROCESS[@]}" ; do 
	for N in "${OBSVERS[@]}" ; do 
		echo "results from job T = ${P}, N = ${N}, K = ${K}"
		cat "gslregress-${P}-${N}-${K}.txt"
		echo " "
	done
done
