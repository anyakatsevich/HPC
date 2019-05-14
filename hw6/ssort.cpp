// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
void myprint(int* A, int N, MPI_Comm comm) {
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);

  for (int i = 0; i < np; i++) {
    MPI_Barrier(comm);
    if (rank == i) {
      printf("process %d ==> ", rank);
      for (long k = 0; k < N; k++) {
        printf("%4ld ", A[k]);
      }
      printf("\n");
    }
    MPI_Barrier(comm);
  }
}
int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 1000000;

  int* vec = (int*)malloc(N*sizeof(int));
  int* splitters = (int*)malloc((p-1)*sizeof(int));
  int* send_displ = (int*)malloc(p*sizeof(int));
  int* receive_displ = (int*)malloc(p*sizeof(int));
  int* sendCount = (int*)malloc(p*sizeof(int));
  int* receiveCount = (int*)malloc(p*sizeof(int));  
  int* vecSorted = (int*)calloc(2*N, sizeof(int));

  send_displ[0] = 0;
  receive_displ[0] = 0;

  int* collectedSplitters;// = NULL;
  if ( !rank)
    collectedSplitters = (int*)malloc(sizeof(int)*p*(p-1));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
    //vec[i] = (int)(rand()/100000);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector

  for(int j=0; j < p-1; j++){
      splitters[j] = vec[(j+1)*(N/p)];
  }
  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  MPI_Gather(splitters, p-1, MPI_INT, collectedSplitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if (!rank){
    std::sort(collectedSplitters, collectedSplitters + p*(p-1));
    for(int j = 0; j < p-1; j++){
    //    splitters[j] = (int)(0.5*(collectedSplitters[(j+1)*(p-1)] + collectedSplitters[(j+1)*(p-1) -1]));
         splitters[j] = collectedSplitters[(j+1)*(p-1)];
  //       printf("%d ",splitters[j]);
     }
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(splitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);
  for(int j = 1; j<p;j++){
    send_displ[j] = std::lower_bound(vec, vec+N, splitters[j-1]) - vec; 
    sendCount[j-1] = send_displ[j] - send_displ[j-1];
  }
  sendCount[p-1] = N - send_displ[p-1];
  MPI_Alltoall(sendCount, 1, MPI_INT, receiveCount, 1, MPI_INT, MPI_COMM_WORLD);
  for(int j=1; j< p; j++){
      receive_displ[j] = receive_displ[j-1] + receiveCount[j-1];
  }
  //myprint(sendCount, p, MPI_COMM_WORLD);
  //myprint(receiveCount, p, MPI_COMM_WORLD);
  int total_num = receive_displ[p-1] + receiveCount[p-1];
  MPI_Alltoallv(vec, sendCount, send_displ, MPI_INT, vecSorted, receiveCount, receive_displ, MPI_INT, MPI_COMM_WORLD);
  std::sort(vecSorted, vecSorted+total_num);
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == rank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  //myprint(vecSorted, 10, MPI_COMM_WORLD);
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename,"w+");

  if(NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }

    for(int n = 0; n < total_num; ++n)
      fprintf(fd, " %d\n", vecSorted[n]);
   
    fclose(fd);
  
  free(splitters);
  free(send_displ);
  free(receive_displ);
  free(sendCount);
  free(receiveCount);
  free(vecSorted);
  free(vec);
  MPI_Finalize();
  return 0;
}
