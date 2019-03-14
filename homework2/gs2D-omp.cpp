#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"



void gaussSeidel(int max_itr, int k, int *odd_idxs, int *even_idxs, double *u_odd, double *u_even, double *f) {

	int N = 2 * k - 1;
	int num_itr = 0;
	double res = 0;
	
    #pragma omp parallel shared(num_itr, u_even, u_odd, res)
    {
		

    while (num_itr < max_itr){
#pragma omp for schedule(static)
		for (int i = 0; i < 2 * k*k - 2 * k + 1; i++) {
			int idx = even_idxs[i];
			u_even[idx] = 0.25*(f[2 * idx] / ((N + 1.0)*(N + 1.0)) + u_odd[idx - 1] + u_odd[idx] + u_odd[idx + k] + u_odd[idx - k - 1]);
		}
#pragma omp barrier
#pragma omp flush(u_even)
#pragma omp for schedule(static)
		for (int i = 0; i < 2 * k*k - 2 * k; i++)
		{
			int idx = odd_idxs[i];
			u_odd[idx] = 0.25*(f[2 * idx + 1] / ((N + 1.0)*(N + 1.0)) + u_even[idx] + u_even[idx + 1] + u_even[idx + k + 1] + u_even[idx - k]);

		}

                #pragma omp barrier
                #pragma omp flush(u_even, u_odd)
                res = 0.0;
                #pragma omp for reduction(+:res) schedule(static)
        for(int i = 0; i < 2*k*k - 2*k;i++){
        
              int idx = odd_idxs[i];
                 double Au_ij= 4*u_odd[idx] - u_even[idx] - u_even[idx + 1] - u_even[idx + k + 1] - u_even[idx - k];
                 double diff_ij1 = Au_ij*(N + 1.0)*(N + 1.0) - f[2 * idx + 1];
                idx = even_idxs[i];
                Au_ij = 4*u_even[idx] - u_odd[idx - 1] - u_odd[idx] - u_odd[idx +  k] - u_odd[idx - k - 1];
                double diff_ij2 = Au_ij*(N + 1.0)*(N + 1.0) - f[2 * idx];
                 res = res + diff_ij1*diff_ij1 + diff_ij2*diff_ij2;
                }
                if (omp_get_thread_num() ==0)
                {
                int idx = even_idxs[2*k*k-2*k];
                double Au_ij = 4*u_even[idx] - u_odd[idx - 1] - u_odd[idx] - u_odd[idx+  k] - u_odd[idx - k - 1];
                double diff_ij = Au_ij*(N + 1.0)*(N + 1.0) - f[2 * idx];
                res = res + diff_ij*diff_ij;
                printf("residual #%d = %f\n", num_itr+1, sqrt(res));
                num_itr++;
                }
             
                #pragma omp barrier


	

}
}
free(u_even); free(u_odd);
}
int main(int argc, char** argv) {
	int max_itr = 15;
	Timer t;
	omp_set_num_threads(4);
	
	for (int pwr = 1; pwr < 4; pwr++) {
	
		int N = 2*pow(10, pwr) + 1;
		int k = (N+1)/2;

		printf("\nN=%d\n", N);

		int num_idx = 2 * k*k - 2 * k + 1;
		int* odd_idxs = (int *)malloc(sizeof(int)*(num_idx - 1));
		int* even_idxs = (int *)malloc(sizeof(int)*num_idx);
		int ctr_even = 0;
		int ctr_odd = 0;
		for (int i = 1; i < 2 * k; i++) {
			for (int j = 1; j < 2 * k; j++) {
				int m = (2 * k + 1)*j + i;
				if (m % 2 == 0) {
					even_idxs[ctr_even] = m / 2;
					ctr_even++;
				}
				else {
					odd_idxs[ctr_odd] = (m - 1) / 2;
					ctr_odd++;
				}
			}
		}


		double* u_even = (double *)calloc(sizeof(double), 2 * k*k + 2 * k + 1);
		double* u_odd = (double *)calloc(sizeof(double), 2 * k*k + 2 * k);
		double* f = (double*)malloc((2 * k + 1)*(2 * k + 1) * sizeof(double));
		for (int j = 0; j < (2 * k + 1)*(2 * k + 1); j++) {
			f[j] = 1;
		}

	
		t.tic();

		gaussSeidel(max_itr, k, odd_idxs, even_idxs, u_odd, u_even, f);
		double time = t.toc();

		printf("time = %f\n", time);


		free(f);
		free(odd_idxs);
		free(even_idxs);
	}
 //  system("pause");
  return 0;
}




