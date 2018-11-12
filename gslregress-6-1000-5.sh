#!/bin/bash
# 
#SBATCH --job-name=gslregress-6-1000-5
#SBATCH --output=gslregress-6-1000-5.txt
#
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=100  
# 
# 

module load devel openmpi

GSL_SHARED_LIB="/share/software/user/open/gsl/2.3/lib"
GSL_LIBS="-L${GSL_SHARED_LIB} -lgsl -lgslcblas -lm"
GSL_INCL="-I/share/software/user/open/gsl/2.3/include"

mpicc gslregress.c -o gslregress ${GSL_LIBS} ${GSL_INCL}

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GSL_SHARED_LIB} ; export LD_LIBRARY_PATH
mpirun -np 6 gslregress 1000 5
