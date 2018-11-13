#!/bin/bash

module load devel openmpi

GSL_SHARED_LIB="/share/software/user/open/gsl/2.3/lib"
GSL_LIBS="-L${GSL_SHARED_LIB} -lgsl -lgslcblas -lm"
GSL_INCL="-I/share/software/user/open/gsl/2.3/include"

mpicc gslregress_ffs.c -o gslregress_ffs ${GSL_LIBS} ${GSL_INCL}
if [[ $? -ne 0 ]] ; then exit 1 ; fi

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GSL_SHARED_LIB} ; export LD_LIBRARY_PATH
mpirun -np 4 gslregress_ffs 1000 5 ffs-data
