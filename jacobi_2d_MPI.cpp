#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>


/* compute global residual, assuming ghost values are updated */
double compute_residual(double *lu, int **map, int lN, double invhsq){
    int i, j;
    double tmp, gres = 0.0, lres = 0.0;

    for (i = 1; i <= lN; i++){
        for(j=1; j <= lN; j++){
            tmp = ((4.0*lu[map[i][j]] - lu[map[i-1][j]] - lu[map[i+1][j]] - lu[map[i][j-1]] - lu[map[i][j+1]]) * invhsq - 1);
            lres += tmp * tmp;
        }
    }
    /* use allreduce for convenience; a reduce would also be sufficient */
    MPI_Reduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return sqrt(gres);
}


int main(int argc, char * argv[]){
    int mpirank, rank_x, rank_y, i, p, rt_p, j, N, lN;
    long iter, max_iters;
    MPI_Status status, status1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if(fmod(sqrt(p), 1.0) > 1e-7 && mpirank == 0){
        printf("p = %d, sqrt of p = %f\n", p, sqrt(p));
        printf("Exiting. p must be a square\n");
        MPI_Abort(MPI_COMM_WORLD, 0);        
    }
    rt_p = (int)sqrt(p);
    rank_y = mpirank/rt_p;
    rank_x = mpirank - rank_y*rt_p;
    /* get name of host running MPI process */
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Rank (%d,%d) running on %s.\n", rank_x, rank_y, processor_name);

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%ld", &max_iters);
    /* compute number of unknowns handled by each process */
    lN = N / rt_p;
    if ((N % rt_p != 0) && mpirank == 0 ) {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of sqrt(p)\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    if (mpirank == 0)
    	printf("lN = %d\n", lN);    
    int **map, *map_mem;
    map = (int **)malloc((lN+2)*sizeof(int*));
    map_mem = (int *)malloc((lN+2)*(lN+2)*sizeof(int));
    for(i=0;i<lN+2;i++)
        map[i] = &map_mem[i*(lN+2)];
    for(i=0;i<lN+2;i++){
        for(j=0;j<lN+2;j++){
            map[i][j] = i*(lN+2)+j;
        }
    }
  
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    /* Allocation of vectors, including left/upper and right/lower ghost points */
    double * lu = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
    double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
    double * lutemp;

    double h = 1.0 / (N + 1);
    double hsq = h * h;
    double invhsq = 1./hsq;
    double gres, gres0, tol = 1e-5;

    //for (int run_num = 0; run_num < 1000; run_num++){
    //lu = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
    //lu_new =  (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
    /* initial residual */
    gres0 = compute_residual(lu, map, lN, invhsq);
    gres = gres0;

    //for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
    for(iter = 0; iter <  max_iters; iter++) {
    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
        for(j = 1; j <= lN; j++)
            lunew[map[i][j]]  = 0.25 * (hsq + lu[map[i-1][j]] + lu[map[i+1][j]] + lu[map[i][j-1]] + lu[map[i][j+1]]);
    }

    /* communicate ghost values */
    for (j = 1; j <= lN; j++){
        if (rank_x < rt_p - 1) {
            MPI_Send(&(lunew[map[lN][j]]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
            MPI_Recv(&(lunew[map[lN+1][j]]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
        }
        if (rank_x > 0) {
            MPI_Send(&(lunew[map[1][j]]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
            MPI_Recv(&(lunew[map[0][j]]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
        }
    }

    for (i = 1; i <= lN; i++){
        if (rank_y < rt_p - 1) {
            
            /* If not the last process, send/recv bdry values to the right */
            MPI_Send(&(lunew[map[i][lN]]), 1, MPI_DOUBLE, mpirank+rt_p, 124, MPI_COMM_WORLD);
            MPI_Recv(&(lunew[map[i][lN+1]]), 1, MPI_DOUBLE, mpirank+rt_p, 123, MPI_COMM_WORLD, &status);
        }
        if (rank_y > 0) {
            /* If not the first process, send/recv bdry values to the left */
            MPI_Send(&(lunew[map[i][1]]), 1, MPI_DOUBLE, mpirank-rt_p, 123, MPI_COMM_WORLD);
            MPI_Recv(&(lunew[map[i][0]]), 1, MPI_DOUBLE, mpirank-rt_p, 124, MPI_COMM_WORLD, &status1);
        }
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
        gres = compute_residual(lu, map, lN, invhsq);
        if (0 == mpirank) {
            printf("Iter =%ld, Residual = %g\n", iter, gres);
        }
    }
    }

    

    /* Clean up */
    free(lu);
    free(lunew);

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if (0 == mpirank) {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }
    MPI_Finalize();
    return 0;
}
