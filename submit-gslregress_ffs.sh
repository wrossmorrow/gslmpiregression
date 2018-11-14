#!/bin/bash
if [[ $# -lt 3 ]] ; then exit 1; fi
BASE=gslregress_ffs
SCRIPT=run-${BASE}-${1}-${2}-${3}.sh
echo "#!/bin/bash
# 
#SBATCH --job-name=${BASE}-${1}-${2}-${3}
#SBATCH --output=${BASE}-${1}-${2}-${3}.txt
#
#SBATCH --ntasks=${1}
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=100  
# 
# 

module load devel openmpi

GSL_SHARED_LIB=\"/share/software/user/open/gsl/2.3/lib\"
GSL_LIBS=\"-L\${GSL_SHARED_LIB} -lgsl -lgslcblas -lm\"
GSL_INCL=\"-I/share/software/user/open/gsl/2.3/include\"

CFLAGS=\"-xHost -O3 -prec-div -no-ftz -restrict\"

mpicc ${BASE}.c -o bin/${BASE} \${CFLAGS} \${GSL_LIBS} \${GSL_INCL}

LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\${GSL_SHARED_LIB} ; export LD_LIBRARY_PATH
mkdir -p rundata/${1}-${2}-${3}
mpirun -np ${1} bin/${BASE} ${2} ${3} rundata/${1}-${2}-${3}/ffs-data" > ${SCRIPT}
sbatch ${SCRIPT}
