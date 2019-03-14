#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main (int argc, char *argv[]) 
{


//can't have tid and i defined outside parallel region, threads will overwrite its value
//int nthreads, i, tid;
int nthreads;
//float total;
float total = 0;
/*** Spawn parallel region ***/
#pragma omp parallel
  {
  /* Obtain thread number */
  int tid = omp_get_thread_num();
  int i;
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  #pragma omp for schedule(dynamic,10)
  for (i=0; i<1000000; i++){
     //#pragma omp atomic
     total = total + i*1.0;
  }
  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
