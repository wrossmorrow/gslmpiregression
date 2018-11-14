
#ifndef _CLOCKTIMER_H_
#define _CLOCKTIMER_H_

typedef struct clktmr {
	unsigned int a1 , d1;
	unsigned int a2 , d2;
} clktmr;

double convert( clktmr * c )
{
	double val;
	val = c->d2 - c->d1;
	val *= pow( 2.0 , 32.0 );
	val = (val + c->a2) - c->a1;
	return val;
}

void tic( clktmr * c ) { 
	asm volatile("rdtsc" : "=a" (c->a1), "=d" (c->d1)); 
}

double toc( clktmr * c ) { 
	asm volatile("rdtsc" : "=a" (c->a2), "=d" (c->d2)); 
	return convert( c ); 
}

#endif