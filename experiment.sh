#!/bin/bash

K=5

PROCESS=( 2 4 6 8 )
OBSVERS=( 100 1000 10000 100000 1000000 )
for P in "${PROCESS[@]}" ; do 
	for N in "${OBSVERS[@]}" ; do 
		echo "submitting job for ${P}, ${N}"
		./submit-gslregress.sh ${P} ${N} ${K}
	done
done