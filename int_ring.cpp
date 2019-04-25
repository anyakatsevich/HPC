#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_pingpong(int N, int rank, int size, long numInts, MPI_Comm comm) {

int *num= (int *)malloc(numInts*sizeof(int));
//num[0] = 0;
//for (long i = 1; i < numInts; i++)
//	num[i] = 3;

MPI_Status status;
MPI_Barrier(comm);
double tt = MPI_Wtime();
for (int i = 0; i < N; i++){
	if (!i && !rank){ //i=0, rank = 0
		num[0] = 0;
		for (int j = 1; j < numInts; j++)
			num[j] = 3;
		MPI_Send(num, numInts, MPI_INT, 1, 999, comm);
			
	}
	else{
		int previous = (rank-1)%size;
		int next = (rank+1)%size;
    		MPI_Recv(num, numInts, MPI_INT, previous, 999, comm, &status);
		num[0]+= rank;
		MPI_Send(num, numInts, MPI_INT, next, 999, comm);


	}
}

if (!rank){
	MPI_Recv(num, numInts, MPI_INT, size-1, 999, comm, &status);
	if (numInts == 1)
		printf("When sending one number: Output = %d, should be = %d\n", num[0], (int)((size - 1.0)*size*N/2.0));
}

tt = MPI_Wtime() - tt;
return tt;	
  
}

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);

  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);


  int N = atoi(argv[1]);
  long numInts = 1;
  double tt = time_pingpong(N, rank, size, numInts, comm);
  if (!rank) printf("pingpong latency: %e ms\n", tt/(N*sizeof(int)*numInts*size) * 1000);

  //numInts = 100;
  numInts = 500000;
  tt = time_pingpong(N, rank, size, numInts, comm);
  if (!rank) printf("pingpong bandwidth: %e GB/s\n", (numInts*N*sizeof(int)*size)/tt/1e9);

  MPI_Finalize();
}

