
#ifndef _GSLREGRESS_H_
#define _GSLREGRESS_H_

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>

// comment out to suppress (most) messages, including data print
// #define _GSLREGRESS_VERBOSE

// optimization tolerance
#define GSLREGRESS_OPT_TOL 1.0e-4

// initial step size
#define GSLREGRESS_STEP_SIZE 1.0

// maximum number of iterations
#define GSLREGRESS_MAX_ITER 1000

// macro for printing
#define EVAL_TYPE(s) ( s == 1 ? "objective" : ( s == 2 ? "gradient" : ( s == 3 ? "objective and gradient" : "unknown" ) ) )

// helper function to get simple uniform random numbers
double urand() { return ((double)rand()) / ((double)RAND_MAX); }

// "start" time to peg to process start, in order to get an idea of synchronization
// because MPI_Wtime() may not be global. With this, we can use cat ... | sort -n 
// to get what is probably a sequential picture of the logs
static double start;

double now() { return MPI_Wtime() - start; }

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * PROBLEM DATA STRUCTURE
 * 
 * We use this to capture/wrap problem data we need to store. Passed through GSL's minimizer routines. 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

typedef struct gsl_ols_params {
	int Nobsv;
	int Nvars;
	int Nfeat;
	int Ncols;
	double * data;
	double * x;
	double * r; // Ncols-length array for residuals
	double * b; // buffer... holds s and ds together here, ds in b[0,Ncols) and s in b[Ncols]
	double s;
} gsl_ols_params;

#endif