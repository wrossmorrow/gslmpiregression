#!/bin/bash

module load devel openmpi/3.1.2 icc/2019

CFLAGS="-xHost -O3 -prec-div -no-ftz -restrict"

GSL_SHARED_LIB="/share/software/user/open/gsl/2.3/lib"
GSL_LIBS="-L${GSL_SHARED_LIB} -lgsl -lgslcblas -lm"
GSL_INCL="-I/share/software/user/open/gsl/2.3/include"

MPI_INCL=$( mpicc -showme:compile )
MPI_LIBS=$( mpicc -showme:link )

icc gslregress.c -o bin/gslregress ${CFLAGS} ${GSL_LIBS} ${GSL_INCL} ${MPI_INCL} ${MPI_LIBS}
if [[ $? -ne 0 ]] ; then exit 1 ; fi

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GSL_SHARED_LIB} ; export LD_LIBRARY_PATH
mpirun -np 4 bin/gslregress 1000 5
