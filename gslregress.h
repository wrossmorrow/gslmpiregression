
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
} gsl_ols_params;

#endif