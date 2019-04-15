#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define BLOCK_SIZE 1024

__global__ void reduction_kernel2(double* sum, const double* a, long N) {
	__shared__ double smem[BLOCK_SIZE];
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

	if (idx < N) smem[threadIdx.x] = a[idx];
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
		if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
	}
}

double reduction(double* a, long N) {
	// assume a is already loaded to device memory
	double *y_d;
	long N_work = 1;
	for (long i = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE); i > 1; i = (i + BLOCK_SIZE - 1) / (BLOCK_SIZE)) N_work += i;
	cudaMalloc(&y_d, N_work * sizeof(double)); // extra memory buffer for reduction across thread-blocks

	double* sum_d = y_d;
	long Nb = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
	reduction_kernel2 << <Nb, BLOCK_SIZE >> >(sum_d, a, N);
	while (Nb > 1) {
		long N = Nb;
		Nb = (Nb + BLOCK_SIZE - 1) / (BLOCK_SIZE);
		reduction_kernel2 << <Nb, BLOCK_SIZE >> >(sum_d + N, sum_d, N);
		sum_d += N;
	}

	double sum;
	cudaMemcpyAsync(&sum, sum_d, 1 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return sum;

}


void Check_CUDA_Error(const char *message) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
		exit(-1);
	}
}


__global__ void update(double* res, double* u, double* u_temp, const double* f, int N) {
	double h = 1 / (N + 1.0);
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int j = idx / (N + 2);
	int i = idx % N + 2;
	if (0 < j && j < N + 1 && 0 < i && i < N + 1) {
		double resid = (1 / (h*h))*(4 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - (N + 2)] - u[idx + (N + 2)]) - f[idx];
		res[(j - 1)*N + i] = resid*resid;
		u_temp[idx] = h*h*f[idx] + u[idx - 1] + u[idx + 1] + u[idx - (N + 2)] + u[idx + (N + 2)];
		u_temp[idx] = 0.25*u_temp[idx];
	}
}
int main() {
	cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("\nDevice Name: %s\n\n", prop.name);

	long N = 1e3; 
	int max_itr = 21;
	double *u;
	cudaMallocHost((void**)&u, (N+2)*(N+2)*sizeof(double));
	double* f;
	cudaMallocHost((void**)&f, (N+2)*(N+2) * sizeof(double)); 
	for (long j = 0; j < (N+2)*(N+2); j++) {
		f[j] = 1.0;
		u[j] = 0.0;
	}

	
	double *f_d, *resArray_d;
	double *u_d, *u_temp_d;

	cudaMalloc(&f_d, (N + 2)*(N + 2) * sizeof(double));
	cudaMalloc(&resArray_d, (N + 2)*(N + 2) * sizeof(double));
	cudaMalloc(&u_d, (N + 2)*(N + 2) * sizeof(double));
	cudaMalloc(&u_temp_d, (N + 2)*(N + 2) * sizeof(double));
	
	double t = omp_get_wtime();
	cudaMemcpyAsync(f_d, f, (N + 2)*(N + 2)* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(u_d, u, (N + 2)*(N + 2) * sizeof(double), cudaMemcpyHostToDevice);	
	
	cudaDeviceSynchronize();
	
	long Nb = (N + 2)*(N + 2) / BLOCK_SIZE;

	for (int i = 0; i < max_itr; i++) {

		update <<<Nb, BLOCK_SIZE>>>(resArray_d, u_d, u_temp_d, f, N);
		cudaDeviceSynchronize();
		double *uTemp = u_temp_d;
		u_temp_d = u_d;
		u_d = uTemp;

		double sum = reduction(resArray_d, N*N);
		if (i%5 == 0)
		printf("res %d = %f\n", i, sqrt(sum));

	}
	
	printf("\nelapsed time: %fs\n", omp_get_wtime() - t);
	cudaFree(u_d); cudaFree(u_temp_d); cudaFree(resArray_d); cudaFree(f_d); 
	cudaFreeHost(u); cudaFreeHost(f);

	
	return 0; 
}
