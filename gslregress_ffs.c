
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * DISTRIBUTED, GSL-SOLVED (SIMULATED DATA) OLS REGRESSION
 * 
 * This file contains a fake ("simulated data") OLS regression using GSL with objective distributed using
 * MPI. You can pass two (non-MPI) parameters to the executable: the number of observations N and the 
 * number of features K. Then the problem solved is 
 * 
 * 		min 1/(2N) sum_{n=1}^N ( sum_{k=1}^K D(k,n) x(k) + x(K+1) - y(n) )^2
 *		wrt x(1),...,x(K+1)
 * 
 * That is, we minimize the (average, halved) sum-of-squares error of an affine (linear + constant) model 
 * of the data columns/outcome pairs (D(:,n),y(n)). 
 * 
 * The code generates this data using a fixed set of "true" coefficients drawn at random in the root process. 
 * It then ships subsets of the columns/outcome pairs to the worker processes in a hopefully balanced way. 
 * The root process then can setup the optimizer, and runs iterations over a distributed objective that uses
 * a signal-broadcast-reduce sequence of collective communication calls to orchestrate objective evaluations. 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <mpi.h>

#include "clocktimer.h"
#include "gslregress.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * SUBPROBLEM OBJECTIVE
 * 
 * This function is called to do the work of a "subproblem" or "batched" part of the objective. 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void subproblem_objective( const double * x , gsl_ols_params * p )
{
	int i , k;

	// compute the residuals from the data and coefficients
	p->s = 0.0;

#ifdef _GSLREGRESS_VECTOR
#pragma vector always
	for( i = 0 ; i < p->Ncols ; i++ ) { 
		p->r[ i ] = x[ p->Nfeat ] - p->data[ i*(p->Nvars) + p->Nfeat ];
#pragma vector always
		for( k = 0 ; k < p->Nfeat ; k++ ) { 
			p->r[ i ] += (p->data)[ i*(p->Nvars) + k ] * x[ k ];
		}
		p->s += p->r[i] * p->r[i];
	}
#else 
	for( i = 0 ; i < p->Ncols ; i++ ) { 
		// intialize with the constant minus observation value
		p->r[ i ] = x[ p->Nfeat ] - p->data[ i*(p->Nvars) + p->Nfeat ];
		for( k = 0 ; k < p->Nfeat ; k++ ) { 
			p->r[ i ] += (p->data)[ i*(p->Nvars) + k ] * x[ k ]; // accumulate dot product into the residual
		}
		p->s += p->r[i] * p->r[i]; // accumulate sum-of-squares
	}
#endif

	p->s /= 2.0; // absorb typical factor-of-two normalization in OLS
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * DISTRIBUTED OBJECTIVE
 * 
 * This function is called by the root ("optimizer") process as the function that provides the objective to
 * minimize. 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double distributed_objective( const gsl_vector * x , void * params )
{
	int i;
	double f;
	gsl_ols_params * p = ( gsl_ols_params * )params;
	int evaluate = 1;

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: evaluating objective at %0.6f" , now() , x->data[0] );
	for( i = 1 ; i < p->Nvars ; i++ ) { printf( " , %0.6f" , x->data[i] ); }
	printf( "\n" );
#endif
	
	// send the evaluate flag, to tell worker processes what to do
	MPI_Bcast( (void*)(&evaluate) , 1 , MPI_INT , 0 , MPI_COMM_WORLD );

	// send variables, and local evaluation
	if( x->stride == 1 ) { // hopefully... 
		MPI_Bcast( (void*)(x->data) , p->Nvars , MPI_DOUBLE , 0 , MPI_COMM_WORLD );
		subproblem_objective( x->data , p );
	} else { // collapse stride
		for( i = 0 ; i < p->Nvars ; i++ ) { p->x[i] = gsl_vector_get( x , i ); }
		MPI_Bcast( (void*)(p->x) , p->Nvars , MPI_DOUBLE , 0 , MPI_COMM_WORLD );
		subproblem_objective( p->x , p );
	}

	// reduction step
	MPI_Reduce( (void*)(&(p->s)) , &f , 1 , MPI_DOUBLE , MPI_SUM , 0 , MPI_COMM_WORLD );

	// normalization
	f /= ((double)(p->Nobsv));

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: obtained %0.6f...\n" , now() , f );
#endif

	return f;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * STANDARD (UNWEIGHTED) ORDINARY LEAST SQUARES
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void gsl_ols( gsl_ols_params * params , double * ols_c ) 
{

	int i , n;
	double chisq = 0.0;
	gsl_matrix * X = gsl_matrix_alloc( params->Nobsv , params->Nvars );
	gsl_vector * y = gsl_vector_alloc( params->Nobsv );
	gsl_vector * c = gsl_vector_alloc( params->Nvars );
	gsl_matrix * S = gsl_matrix_alloc( params->Nvars , params->Nvars );
	for( n = 0 ; n < params->Nobsv ; n++ ) {
		for( i = 0 ; i < params->Nfeat ; i++ ) {
			gsl_matrix_set( X , n , i , params->data[ n * params->Nvars + i ] );
		}
		gsl_matrix_set( X , n , params->Nfeat , 1.0 );
		gsl_vector_set( y , n , params->data[ n * params->Nvars + params->Nfeat ] );
	}

	gsl_multifit_linear_workspace * ols = gsl_multifit_linear_alloc( params->Nobsv , params->Nvars );
	gsl_multifit_linear( X , y , c , S , &chisq , ols );
	gsl_multifit_linear_free( ols );
	gsl_matrix_free( X );
	gsl_vector_free( y );
	gsl_matrix_free( S );

	if( ols_c != NULL ) {
		for( i = 0 ; i < params->Nvars ; i++ ) { 
			ols_c[i] = gsl_vector_get( c , i );
		}
	}

	printf( "%0.6f: GSL OLS estimates: %0.3f" , now() , gsl_vector_get( c , 0 ) );
	for( i = 1 ; i < params->Nvars ; i++ ) { printf( " , %0.3f" , gsl_vector_get( c , i ) ); }
	printf( "\n" );
	gsl_vector_free( c );

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * NON-DISTRIBUTED OPTIMIZATION
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static double * residuals;

double non_distributed_objective( const gsl_vector * x , void * params )
{
	int i , k;
	double f;
	gsl_ols_params * p = ( gsl_ols_params * )params;

	// compute the residuals from the data and coefficients
	f = 0.0;

#ifdef _GSLREGRESS_VECTOR
#pragma vector always
	for( i = 0 ; i < p->Nobsv ; i++ ) { 
		residuals[i] = x->data[ x->stride * p->Nfeat ] - (p->data)[ i * (p->Nvars) + p->Nfeat ]; 
#pragma vector always
		for( k = 0 ; k < p->Nfeat ; k++ ) { 
			residuals[i] += (p->data)[ i*(p->Nvars) + k ] * x->data[ x->stride * k ];
		}
		f += residuals[i] * residuals[i]; // accumulate sum-of-squares
	}
#else
	for( i = 0 ; i < p->Nobsv ; i++ ) { 
		// intialize with the constant minus observation value
		residuals[ i ] = gsl_vector_get( x , p->Nfeat ) - (p->data)[ i * (p->Nvars) + p->Nfeat ]; 
		for( k = 0 ; k < p->Nfeat ; k++ ) { 
			residuals[ i ] += (p->data)[ i*(p->Nvars) + k ] * gsl_vector_get( x , k ); // accumulate dot product into the residual
		}
		f += residuals[i] * residuals[i]; // accumulate sum-of-squares
	}
#endif

	f /= 2.0 * ((double)(p->Nobsv)) ; // absorb typical factor-of-two normalization in OLS
	
	return f;
}

void gsl_minimize( gsl_ols_params * params , const double * x0 ) 
{
	int i;

	residuals = ( double * )malloc( params->Nobsv * sizeof( double ) );
	for( i = 0 ; i < params->Nobsv ; i++ ) { residuals[i] = 0.0; }

	// minimizer object
	const gsl_multimin_fminimizer_type * T = gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_fminimizer * s = gsl_multimin_fminimizer_alloc( T , params->Nvars );

	// evaluation function
	gsl_multimin_function sos;
	sos.n = params->Nvars; // features and constant
	sos.f = &non_distributed_objective; // defined elsewhere
	sos.params = (void*)params; // we'll pass the data object, allocated here, to objective evaluations

	// step size
	gsl_vector * ss = gsl_vector_alloc( params->Nvars );
	gsl_vector_set_all( ss , 1.0 );

	// initial point (random guess)
	gsl_vector * x = gsl_vector_alloc( params->Nvars );
	for( i = 0 ; i < params->Nvars ; i++ ) { gsl_vector_set( x , i , x0[i] ); }

	// "register" these with the minimizer
	gsl_multimin_fminimizer_set( s , &sos , x , ss );

	// iterations
	int status = GSL_CONTINUE;
	int iter = 0; 
	double size;
	do {

		// iterate will call the distributed objective
		status = gsl_multimin_fminimizer_iterate( s );
		iter++;

		if( status ) { break; } // iteration failure? 

		size = gsl_multimin_fminimizer_size( s );
		status = gsl_multimin_test_size( size , GSLREGRESS_OPT_TOL );

	} while( status == GSL_CONTINUE && iter < GSLREGRESS_MAX_ITER );

	// print out result obtained
	printf( "%0.6f: estimated coeffs: %0.3f" , now() , gsl_vector_get( s->x , 0 ) );
	for( i = 1 ; i < params->Nvars ; i++ ) { printf( " , %0.3f" , gsl_vector_get( s->x , i ) ); }
	printf( "\n" );

	// clean up after optimizer
	gsl_vector_free( x );
	gsl_vector_free( ss );
	gsl_multimin_fminimizer_free( s );

	free( residuals );

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * OPTIMIZER PROCESS
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void optimizer_process( int P , int B , int R , int F , char * prefix , const double * x0 , gsl_ols_params * params )
{
	FILE * fp;
	char filename[1024];

	int i , status , iter = 0;
	double size;

	// initial barrier, basically separating the data simulation from the solve attempt
	MPI_Barrier( MPI_COMM_WORLD );

	// read local file... like we would in any worker process

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: optimizer process: reading data...\n" , now() );
#endif

	// create process-specific filename, open the file, and read the data
	sprintf( filename , "%s_%i.dat" , prefix , 0 );
	fp = fopen( filename , "rb" );
	fread( (void*)(params->data) , sizeof( double ) , params->Ncols * params->Nvars , fp );
	fclose( fp );

	// setup GSL optimizer

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: optimizer process: setting up GSL optimizer\n" , now() );
#endif

	// minimizer object
	const gsl_multimin_fminimizer_type * T = gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_fminimizer * s = gsl_multimin_fminimizer_alloc( T , params->Nvars );

	// evaluation function
	gsl_multimin_function sos;
	sos.n = params->Nvars; // features and constant
	sos.f = &distributed_objective; // defined elsewhere
	sos.params = (void*)(params); // we'll pass the data object, allocated here, to objective evaluations

	// step size
	gsl_vector * ss = gsl_vector_alloc( params->Nvars );
	gsl_vector_set_all( ss , 1.0 );

	// initial point (random guess, but the same as possibly used above)
	gsl_vector * x = gsl_vector_alloc( params->Nvars );
	if( x0 == NULL ) {
		for( i = 0 ; i < params->Nvars ; i++ ) { gsl_vector_set( x , i , 2.0 * urand() - 1.0 ); }
	} else {
		for( i = 0 ; i < params->Nvars ; i++ ) { gsl_vector_set( x , i , x0[i] ); }
	}

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: optimizer process: registering problem\n" , now() );
#endif

	// "register" these with the minimizer
	gsl_multimin_fminimizer_set( s , &sos , x , ss );

	// synchronize before starting iterations
	// 
	// NOTE: This is a ** BAD ** idea. This deadlocks the code with GSL, at least. 
	// When we "register" the function calls, GSL will call them (the objective at least). 
	// If we expect to wait until iterations start with this synchronization, we deadlock. 
	// 
	// MPI_Barrier( MPI_COMM_WORLD ); 

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: optimizer process: starting iterations\n" , now() );
#endif

	// iterations
	status = GSL_CONTINUE;
	do {

		// iterate will call the distributed objective
		status = gsl_multimin_fminimizer_iterate( s );
		iter++;

#ifdef _GSLREGRESS_VERBOSE
		printf( "%0.6f: optimizer process: iteration %i\n" , now() , iter );
#endif

		if( status ) { break; } // iteration failure? 

		size = gsl_multimin_fminimizer_size( s );
		status = gsl_multimin_test_size( size , GSLREGRESS_OPT_TOL );

	} while( status == GSL_CONTINUE && iter < GSLREGRESS_MAX_ITER );

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: optimizer process: finished iterations\n" , now() );
#endif

	// non-verbose print
	printf( "%0.6f: estimated coeffs: %0.3f" , now() , gsl_vector_get( s->x , 0 ) );
	for( i = 1 ; i < params->Nvars ; i++ ) { printf( " , %0.3f" , gsl_vector_get( s->x , i ) ); }
	printf( "\n" );

	// ** IMPORTANT ** 
	//
	// worker threads will _always_ loop back to the evaluation broadcast
	// so we have to signal them that we're done. 

	status = 0;
	MPI_Bcast( (void*)(&status) , 1 , MPI_INT , 0 , MPI_COMM_WORLD );

	// clean up after optimizer
	gsl_vector_free( x );
	gsl_vector_free( ss );
	gsl_multimin_fminimizer_free( s );

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * WORKER PROCESS
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void worker_process( int p , char * prefix , gsl_ols_params * params )
{
	FILE * fp;
	char filename[1024];
	int i , status;

	// initial barrier
	MPI_Barrier( MPI_COMM_WORLD );

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: process %i: reading data...\n" , now() , p );
#endif

	// create process-specific filename, open the file, and read the data
	sprintf( filename , "%s_%i.dat" , prefix , p );
	fp = fopen( filename , "rb" );
	fread( (void*)(params->data) , sizeof( double ) , params->Ncols * params->Nvars , fp );
	fclose( fp );

	// do any local setup required with this data...

	// synchronize before starting iterations
	// 
	// NOTE: This is a ** BAD ** idea. This deadlocks the code with GSL, at least. 
	// When we "register" the function calls, GSL will call them (the objective at least). 
	// If we expect to wait until iterations start with this synchronization, we deadlock. 
	// 
	// MPI_Barrier( MPI_COMM_WORLD );

	// entering iteration phase... 
	while( 1 ) {

		// get status, and evaluate to see if we should continue
		MPI_Bcast( (void*)(&status) , 1 , MPI_INT , 0 , MPI_COMM_WORLD );
		if( status != 1 ) { 
#ifdef _GSLREGRESS_VERBOSE
			printf( "%0.6f: process %i: exiting worker loop\n" , now() , p );
#endif
			break; 
		}

		// get variables
		MPI_Bcast( (void*)(params->x) , params->Nvars , MPI_DOUBLE , 0 , MPI_COMM_WORLD );

#ifdef _GSLREGRESS_VERBOSE
		printf( "%0.6f: process %i evaluating at %0.6f" , now() , p , params->x[0] );
		for( i = 1 ; i < params->Nvars ; i++ ) { printf( " , %0.6f" , params->x[i] ); }
		printf( "\n" );
#endif

		// local evaluation, writes into params.s
		subproblem_objective( params->x , params );

		// sum-reduce to accumulate parts back in the root process
		MPI_Reduce( (void*)(&(params->s)) , NULL , 1 , MPI_DOUBLE , MPI_SUM , 0 , MPI_COMM_WORLD );

	}

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * EXECUTABLE ROUTINE
 * 
 * Set up MPI, all-process problem data, setup and start optimization in the root process and run loop
 * in the worker processes. 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int main( int argc , char * argv[] ) 
{

	int i , n , r , p , P , N , K , B , R;

	int status;

	double * coeffs;

	double method_start;

	FILE * fp;
	char filename_prefix[1024];
	char filename[1024];

	// here we create a parameters structure for our use
	gsl_ols_params params;

	// seed random number generator
	srand( time(NULL) );

	// always initialize MPI... after this call argc and argv are like normal executable
	// arguments we can use
	MPI_Init( &argc , &argv );

	// this process's rank and total number
	MPI_Comm_rank( MPI_COMM_WORLD , &p );
	MPI_Comm_size( MPI_COMM_WORLD , &P );

	// processor name, if we want it for prints
	char procname[ MPI_MAX_PROCESSOR_NAME ];
	int procnamelength;
	MPI_Get_processor_name( procname , &procnamelength );

	// read N and K and a file prefix
	if( argc < 4 ) { return 1; }

	// initial barrier
	MPI_Barrier( MPI_COMM_WORLD );
	start = MPI_Wtime();

	// get N and K from command line arguments (all processes can do this)
	N = (int)strtol( argv[1] , NULL , 10 ); // number of "columns"
	K = (int)strtol( argv[2] , NULL , 10 ); // "column" size

	// get the filename prefix to use
	strcpy( filename_prefix , argv[3] );

	R = N % P; // remainder (to spread evenly over processes)
	B = ( N - R ) / P; // block size (even division)

	// setup local variables

	params.Nobsv = N;
	params.Nvars = K + 1; // number of variables: K features plus a constant
	params.Nfeat = K; 
	params.Ncols = B + ( p < R ? 1 : 0 );

#ifdef _GSLREGRESS_VERBOSE
	printf( "%0.6f: process %i: number of variables... %i\n" , now() , p , params.Nvars );
	printf( "%0.6f: process %i: number of features.... %i\n" , now() , p , params.Nfeat );
	printf( "%0.6f: process %i: number of columns..... %i\n" , now() , p , params.Ncols );
	printf( "%0.6f: process %i: filename prefix....... %s\n" , now() , p , filename_prefix );
#endif

	// variables
	params.x = ( double * )malloc( params.Nvars * sizeof( double ) );
	for( i = 0 ; i < params.Nvars ; i++ ) { (params.x)[i] = 0.0; }

	// residuals... as many as observations we are tracking in this process
	params.r = ( double * )malloc( params.Ncols * sizeof( double ) );
	for( i = 0 ; i < params.Ncols ; i++ ) { (params.r)[i] = 0.0; }

	// size the data array: we will expect to get Ncols "columns" each of length K+1 = Nvars (contiguous)
	params.data = ( double * )malloc( ( params.Ncols * params.Nvars ) * sizeof( double ) );

	// root/worker differentiation starts...

	if( p == 0 ) {

		// allocate space for coefficients
		coeffs = ( double * )malloc( params.Nvars * sizeof( double ) );
		for( i = 0 ; i < params.Nvars ; i++ ) { coeffs[i] = urand(); }

		printf( "%0.6f: real coefficients: %0.3f" , now() , coeffs[0] );
		for( i = 1 ; i < params.Nvars ; i++ ) { printf( " , %0.3f" , coeffs[i] ); }
		printf( "\n" );

		// _write_ data for each process out to a file prefixed by CLI argument
		int count = 0;
		for( r = 0 ; r < P ; r++ ) {
			sprintf( filename , "%s_%i.dat" , filename_prefix , r );
			fp = fopen( filename , "wb" );
			count = ( B + ( r < R ? 1 : 0 ) );
			for( n = 0 ; n < count ; n++ ) {
				params.data[ n * params.Nvars + K ] = coeffs[ K ]; // initialize with constant
				for( i = 0 ; i < K ; i++ ) {
					params.data[ n * params.Nvars + i ] = urand(); // random features
					params.data[ n * params.Nvars + K ] += params.data[ n * params.Nvars + i ] * coeffs[i]; // accumulate observation
				}
				params.data[ n * params.Nvars + K ] += 2.0 * urand() - 1.0; // plus error
			}
			count *= params.Nvars;
			fwrite( params.data , sizeof( double ) , count , fp );
			fclose( fp );
		}

#ifdef _GSLREGRESS_VERBOSE
		printf( "%0.6f: wrote data\n" , now() );
#endif

		/*
		// initial condition
		double * x0 = ( double * )malloc( params.Nvars * sizeof( double ) );
		for( i = 0 ; i < params.Nvars ; i++ ) { x0[i] = 2.0 * urand() - 1.0; }

		// do a "standard" regression with the GSL tools
		method_start = MPI_Wtime();
		printf( "%0.6f: GSL OLS Regression...\n" , now() );
		gsl_ols( &params , NULL );
		printf( "%0.6f:   took %0.6fs (don't compare to others)\n" , now() , MPI_Wtime() - method_start );

		// do a "serial" minimization, exactly what we do below but without distributing the objective
		method_start = MPI_Wtime();
		printf( "%0.6f: Serial GSL Multimin Estimation...\n" , now() );
		gsl_minimize( &params , x0 );
		printf( "%0.6f:   took %0.6fs \n" , now() , MPI_Wtime() - method_start );

		// do a distributed optimization
		method_start = MPI_Wtime();
		printf( "%0.6f: Distributed GSL Multimin Estimation... \n" , now() );
		optimizer_process( P , B , R , K+1 , filename_prefix , x0 , &params );
		printf( "%0.6f:   took %0.6fs \n" , now() , MPI_Wtime() - method_start );

		free( x0 );
		*/

		// do a distributed optimization
		method_start = MPI_Wtime();
		printf( "%0.6f: Distributed GSL Multimin Estimation... \n" , now() );
		optimizer_process( P , B , R , K+1 , filename_prefix , NULL , &params );
		printf( "%0.6f:   took %0.6fs \n" , now() , MPI_Wtime() - method_start );

	} else { worker_process( p , filename_prefix , &params ); }

	// cleanup local stuff
	free( params.data );
	free( params.x );
	free( params.r );

	// free the real coefficients
	free( coeffs );

	// always finalize MPI
	MPI_Finalize();

	return 0;

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * Copyright 2018, W. Ross Morrow
 * CIRCLE RSS, Stanford GSB
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
