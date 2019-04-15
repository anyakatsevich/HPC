#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define BLOCK_SIZE 1024

void matVecMult(double* c, const double* A, const double* b, long N, long m) {
	for (long j = 0; j < m; j++) {
		for (long i = 0; i < N; i++) 
			c[j] += A[N*j + i] * b[i];
	}
}


__global__
void vec_mult_kernel(double* C, const double* A, const double* b, long N, long m) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx_x < N && idx_y < m) {
	//	C[N*idx_y + idx_x] = A[N*idx_y + idx_x] * b[idx_x];
		C[m*idx_x + idx_y] = A[N*idx_y + idx_x] * b[idx_x];
	}
}

void Check_CUDA_Error(const char *message) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
		exit(-1);
	}
}

__global__ void reduction_kernel2(double* sum, const double* a, long N, long m) {
	__shared__ double smem[BLOCK_SIZE];
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y;
	if (idx < N) smem[threadIdx.x] = a[m*idx + idx_y];
	else smem[threadIdx.x] = 0;

	__syncthreads();
	if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
	__syncthreads();
	if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
	__syncthreads();
	if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
	__syncthreads();
	if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x + 64];
	__syncthreads();
	if (threadIdx.x <  32) {
		smem[threadIdx.x] += smem[threadIdx.x + 32];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 16];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 8];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 4];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 2];
		__syncwarp();
		if (threadIdx.x == 0) {
			sum[m*blockIdx.x + idx_y] = smem[0] + smem[1];
		}
	}
}

int main() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("\n(Device #0) Name: %s\nCompute capability: %d.%d\n", prop.name, prop.major,prop.minor);
	printf("Total memory: %f GB\n", prop.totalGlobalMem*1.0e-9);
	//printf("Shared memory per block: %f KB\n", prop.sharedMemPerBlock/1024.0);
	printf("Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	//printf("Max threads per block: %d\n\n\n", prop.maxThreadsPerBlock);

	//long N = (1UL<<25);
	long N = BLOCK_SIZE*100;
	long m = 250;
	//printf("Matrix dimensions: %ld x %ld, vector dimensions: %ld\n\n", m, N, N); 
	int block_size = 32; //32 x 32 block of threads
	int num_x = (N + block_size - 1) / block_size;
	int num_y = (m + block_size - 1) / block_size;

	dim3 block_dims(block_size, block_size);
	dim3 grid_dims(num_x, num_y);

	double *A, *b, *c, *c_ref;
	
        c_ref = (double* ) malloc(m*sizeof(double));
        cudaMallocHost((void**)&A, N * m * sizeof(double));
	cudaMallocHost((void**)&b, N * sizeof(double));
	cudaMallocHost((void**)&c, m * sizeof(double));
	

	for (long i = 0; i < N; i++) {
		b[i] = 1.0 / (i + 1.0);
		for (long j = 0; j < m; j++) {
			A[N*j + i] = 1 / (i + 1.0);
		}
	}
	for (long j = 0; j < m; j++) {
		c[j] = 0.0;
		c_ref[j] = 0.0;
	}
	double tt = omp_get_wtime();	
	matVecMult(c_ref, A, b, N, m);
	printf("CPU Bandwidth = %f GB/s\n", m*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
	double *A_d, *b_d, *c_d, *C_d; 
	cudaMalloc(&A_d, N * m * sizeof(double));
	cudaMalloc(&C_d, N * m * sizeof(double));
	cudaMalloc(&b_d, N * sizeof(double));
	tt = omp_get_wtime();
	cudaMemcpyAsync(A_d, A, N * m * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(b_d, b, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	vec_mult_kernel << <grid_dims, block_dims>> >(C_d, A_d, b_d, N, m);
	cudaDeviceSynchronize();

	long N_work = m;
	for (long i = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE); i > 1; i = (i + BLOCK_SIZE - 1) / (BLOCK_SIZE)) N_work += i*m;

	double *temp_d;
	cudaMalloc(&temp_d, N_work * sizeof(double));
	c_d = temp_d;
	long Nb = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
	dim3 grid_dims2(Nb, m);
	dim3 block_dims2(BLOCK_SIZE, 1);
	reduction_kernel2 << <grid_dims2, block_dims2 >> >(c_d, C_d, N, m);
	cudaDeviceSynchronize();
	while (Nb > 1) {
		long N = Nb;
		Nb = (Nb + BLOCK_SIZE - 1) / (BLOCK_SIZE);
		dim3 grid_dims(Nb, m);
		reduction_kernel2 << <grid_dims, block_dims2 >> >(c_d + N*m, c_d, N, m);
		cudaDeviceSynchronize();
		c_d += N*m;
	}
	
	cudaDeviceSynchronize();
	cudaMemcpyAsync(c, c_d, m * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	printf("GPU Bandwidth = %f GB/s\n", m* N * sizeof(double) / (omp_get_wtime() - tt) / 1e9);
	double err = 0.0;
	for (long j = 0; j < m; j++)
		err += (c[j] - c_ref[j])*(c[j] - c_ref[j]);
	
	printf("err = %f\n", err);
	cudaFree(A_d); cudaFree(b_d); cudaFree(temp_d); cudaFree(C_d); //cudaFree(c_d);
        cudaFreeHost(A); cudaFreeHost(b); cudaFreeHost(c); free(c_ref);
	return 0; 
}
