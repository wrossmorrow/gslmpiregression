
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
 * min 1/(2N) sum_{n=1}^N ( sum_{k=1}^K D(k,n) x(k) + x(K+1) - y(n) )^2
 *wrt x(1),...,x(K+1)
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
#include <math.h>

#include <mpi.h>

#include <gsl/gsl_multimin.h>

// comment out to suppress (most) messages, including data print
#define _GSLREGRESS_VERBOSE

// optimization tolerance
#define GSLREGRESS_OPT_TOL 1.0e-4

// maximum number of iterations
#define GSLREGRESS_MAX_ITER 1000

// "start" time to peg to process start, in order to get an idea of synchronization
// because MPI_Wtime() may not be global. With this, we can use cat ... | sort -n 
// to get what is probably a sequential picture of the logs
static double start;

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

typedef struct gls_ols_params {
  int Nobsv;
  int Nvars;
  int Nfeat;
  int Ncols;
  double * data;
  double * x;
  double * r;
  double s;
} gls_ols_params;

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

void subproblem_objective( const double * x , gls_ols_params * p )
{
  int i , k;

  // compute the residuals from the data and coefficients
  p->s = 0.0;
  for( i = 0 ; i < p->Ncols ; i++ ) { 
    p->r[ i ] = x[ p->Nfeat ] - p->data[ i*(p->Nvars) + p->Nfeat ]; // intialize with the constant minus observation value
    for( k = 0 ; k < p->Nfeat ; k++ ) { 
      p->r[ i ] += (p->data)[ i*(p->Nvars) + k ] * x[ k ]; // accumulate dot product into the residual
    }
    p->s += p->r[i] * p->r[i]; // accumulate sum-of-squares
  }
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
  gls_ols_params * p = ( gls_ols_params * )params;
  int evaluate = 1;

#ifdef _GSLREGRESS_VERBOSE
  printf( "%0.6f: evaluating objective at %0.6f" , MPI_Wtime()-start , x->data[0] );
  for( i = 1 ; i < p->Nvars ; i++ ) { printf( " , %0.6f" , x->data[i] ); }
  printf( "\n" );
#endif

  if( x->stride != 1 ) { // hopefully... 
    return NAN;
  }
  
  // send the evaluate flag, to tell worker processes what to do
  MPI_Bcast( (void*)(&evaluate) , 1 , MPI_INT , 0 , MPI_COMM_WORLD );

  // send variables
  MPI_Bcast( (void*)(x->data) , p->Nvars , MPI_DOUBLE , 0 , MPI_COMM_WORLD );

  // local evaluation
  subproblem_objective( x->data , p );

  // reduction step
  MPI_Reduce( (void*)(&(p->s)) , &f , 1 , MPI_DOUBLE , MPI_SUM , 0 , MPI_COMM_WORLD );

  // normalization
  f /= ((double)(p->Nobsv));

#ifdef _GSLREGRESS_VERBOSE
  printf( "%0.6f: obtained %0.6f...\n" , MPI_Wtime()-start , f );
#endif

  return f;
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

  int i , n , r, p , P , N , K , B , R;

  int status;

  double * coeffs;

  // here we create a parameters structure for our use
  gls_ols_params params;

  // always initialize MPI... after this call argc and argv are like normal executable
  // arguments we can use
  MPI_Init( &argc , &argv );

  // this process's rank and total number
  MPI_Comm_rank( MPI_COMM_WORLD , &p );
  MPI_Comm_size( MPI_COMM_WORLD , &P );

  // 
  char procname[ MPI_MAX_PROCESSOR_NAME ];
  int procnamelength;
  MPI_Get_processor_name( procname , &procnamelength );

  // read N and K
  if( argc < 3 ) { return 1; }

  // initial barrier
  MPI_Barrier( MPI_COMM_WORLD );
  start = MPI_Wtime();

  // get N from command line arguments (all processes can do this)
  N = (int)strtol( argv[1] , NULL , 10 ); // number of "columns"
  K = (int)strtol( argv[2] , NULL , 10 ); // "column" size

  R = N % P; // remainder (to spread evenly over processes)
  B = ( N - R ) / P; // block size (even division)

  // setup local variables

  params.Nobsv = N;
  params.Nvars = K + 1; // number of variables: K features plus a constant
  params.Nfeat = K; 
  params.Ncols = B + ( p < R ? 1 : 0 );

#ifdef _GSLREGRESS_VERBOSE
  printf( "%0.6f: process %i: number of variables... %i\n" , MPI_Wtime()-start , p , params.Nvars );
  printf( "%0.6f: process %i: number of features.... %i\n" , MPI_Wtime()-start , p , params.Nfeat );
  printf( "%0.6f: process %i: number of columns..... %i\n" , MPI_Wtime()-start , p , params.Ncols );
#endif

  // variables
  params.x = ( double * )malloc( params.Nvars * sizeof( double ) );
  for( i = 0 ; i < params.Nvars ; i++ ) { (params.x)[i] = 0.0; }

  // residuals... as many as observations we are tracking in this process
  params.r = ( double * )malloc( params.Ncols * sizeof( double ) );
  for( i = 0 ; i < params.Ncols ; i++ ) { (params.r)[i] = 0.0; }

  // root/worker differentiation starts...

  if( p == 0 ) {

    // allocate space for coefficients
    coeffs = ( double * )malloc( params.Nvars * sizeof( double ) );
    for( i = 0 ; i < params.Nvars ; i++ ) { coeffs[i] = urand(); }

#ifdef _GSLREGRESS_VERBOSE
    printf( "%0.6f: process %i: real coefficients: %0.2f" , MPI_Wtime()-start , p , coeffs[0] );
    for( i = 1 ; i < params.Nvars ; i++ ) { printf( " , %0.2f" , coeffs[i] ); }
    printf( "\n" );
#endif

    // size the data array: root process will create all the data, and send it out
    // This "mimics" a read-disperse model of smaller data problems, where the complexity 
    // is not in the data size but in the model evaluation
    params.data = ( double * )malloc( ( N * params.Nvars ) * sizeof( double ) );
    for( n = 0 ; n < N ; n++ ) {
      params.data[ n * params.Nvars + K ] = coeffs[ K ]; // initialize with constant
      for( i = 0 ; i < K ; i++ ) {
	params.data[ n * params.Nvars + i ] = urand(); // random features
	params.data[ n * params.Nvars + K ] += params.data[ n * params.Nvars + i ] * coeffs[i]; // accumulate observation
      }
      params.data[ n * params.Nvars + K ] += 2.0 * urand() - 1.0; // plus error
    }

#ifdef _GSLREGRESS_VERBOSE
    printf( "%0.6f: all data: \n" , MPI_Wtime()-start );
    for( n = 0 ; n < N ; n++ ) {
      printf( "%0.6f:  column %i: %0.2f" , MPI_Wtime()-start , n , params.data[n*(K+1)+0] );
      for( i = 1 ; i <= K ; i++ ) {
	printf( ", %0.2f" , params.data[n*(K+1)+i] );
      }
      printf( "\n" );
    }
#endif

    // initial barrier
    MPI_Barrier( MPI_COMM_WORLD );

    int * counts;
    int * offset;

    // prepare data for scatter
    counts = ( int * )malloc( P * sizeof( int ) );
    offset = ( int * )malloc( P * sizeof( int ) );

    // root process needs all counts
    for( i = 0 ; i < P ; i++ ) {
      counts[i] = B + ( i < R ? 1 : 0 );
      counts[i] *= K + 1; // multiply by regression size (K), plus one for the obseration (y)
    }

    offset[0] = 0;
    for( i = 1 ; i < P ; i++ ) {
      offset[i] = offset[i-1] + counts[i-1];
    }

    // sending-end scatterv, send from "data"
    MPI_Scatterv( (void*)(params.data) , counts , offset , MPI_DOUBLE , NULL , 0 , MPI_DOUBLE , 0 , MPI_COMM_WORLD );

    // free scatterv data after send (should we save this for later?)
    free( counts );
    free( offset );

    // setup GSL optimizer

#ifdef _GSLREGRESS_VERBOSE
    printf( "%0.6f: process %i: setting up GSL optimizer\n" , MPI_Wtime()-start , p );
#endif

    // minimizer object
    const gsl_multimin_fminimizer_type * T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer * s = gsl_multimin_fminimizer_alloc( T , params.Nvars );

    // evaluation function
    gsl_multimin_function sos;
    sos.n = params.Nvars; // features and constant
    sos.f = &distributed_objective; // defined elsewhere
    sos.params = (void*)(&params); // we'll pass the data object, allocated here, to objective evaluations

    // step size
    gsl_vector * ss = gsl_vector_alloc( params.Nvars );
    gsl_vector_set_all( ss , 1.0 );

    // initial point (random guess)
    gsl_vector * x = gsl_vector_alloc( params.Nvars );
    for( i = 0 ; i < params.Nvars ; i++ ) {
      gsl_vector_set( x , i , 2.0 * urand() - 1.0 );
    }

#ifdef _GSLREGRESS_VERBOSE
    printf( "%0.6f: process %i: registering problem\n" , MPI_Wtime()-start , p );
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
    printf( "%0.6f: process %i: starting iterations\n" , MPI_Wtime()-start , p );
#endif

    // iterations
    status = GSL_CONTINUE;
    int iter = 0; 
    double size;
    do {

      // iterate will call the distributed objective
      status = gsl_multimin_fminimizer_iterate( s );
      iter++;

#ifdef _GSLREGRESS_VERBOSE
      printf( "%0.6f: process %i: iteration %i\n" , MPI_Wtime()-start , p , iter );
#endif

      if( status ) { break; } // iteration failure? 

      size = gsl_multimin_fminimizer_size( s );
      status = gsl_multimin_test_size( size , GSLREGRESS_OPT_TOL );

    } while( status == GSL_CONTINUE && iter < GSLREGRESS_MAX_ITER );

#ifdef _GSLREGRESS_VERBOSE
    printf( "%0.6f: process %i: finished iterations\n" , MPI_Wtime()-start , p );
#endif

    // only non-verbose print
    printf( "%0.6f: process %i: real coefficients: %0.2f" , MPI_Wtime()-start , p , coeffs[0] );
    for( i = 1 ; i < params.Nvars ; i++ ) { printf( " , %0.2f" , coeffs[i] ); }
    printf( "\n" );
    printf( "%0.6f: process %i: estimated coeffs: %0.2f" , MPI_Wtime()-start , p , ((s->x)->data)[0] );
    for( i = 1 ; i < params.Nvars ; i++ ) { printf( " , %0.2f" , ((s->x)->data)[i] ); }
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

  } else {

    // initial barrier
    MPI_Barrier( MPI_COMM_WORLD );

    // size the data array: we will expect to get Ncols "columns" each of length K+1 = Nvars (contiguous)
    params.data = ( double * )malloc( ( params.Ncols * params.Nvars ) * sizeof( double ) );

    // receiving-end scatterv, write result into "data"
    MPI_Scatterv( NULL , NULL , NULL , MPI_DOUBLE , (void*)(params.data) , params.Ncols * params.Nvars , MPI_DOUBLE , 0 , MPI_COMM_WORLD );

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
	printf( "%0.6f: process %i: exiting worker loop\n" , MPI_Wtime()-start , p );
#endif
	break; 
      }

#ifdef _GSLREGRESS_VERBOSE
      printf( "%0.6f: process %i: continuing\n" , MPI_Wtime()-start , p );
#endif

      // get variables
      MPI_Bcast( (void*)(params.x) , params.Nvars , MPI_DOUBLE , 0 , MPI_COMM_WORLD );

#ifdef _GSLREGRESS_VERBOSE
      printf( "%0.6f: process %i evaluating at %0.6f" , MPI_Wtime()-start , p , params.x[0] );
      for( i = 1 ; i < params.Nvars ; i++ ) { printf( " , %0.6f" , params.x[i] ); }
      printf( "\n" );
#endif

      // local evaluation, writes into params.s
      subproblem_objective( params.x , &params );

      // sum-reduce to accumulate parts back in the root process
      MPI_Reduce( (void*)(&(params.s)) , NULL , 1 , MPI_DOUBLE , MPI_SUM , 0 , MPI_COMM_WORLD );

    }

  }

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
