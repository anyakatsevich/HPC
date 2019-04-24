#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>



void gaussSeidel(int max_itr, double tol, int N, double *u, double *f) {
	double h = 1 / (N + 1.0);
	double* u_temp = (double *)calloc(sizeof(double), (N+2)*(N+2));
	double res = 1;
	int itr_num = 0;
	while (res > tol && itr_num < max_itr){
	//for(int itr_num = 0; itr_num < max_itr; itr_num++){
		double sum1 = 0; double sum2 = 0; 
		#pragma omp parallel for collapse(2) reduction(+:sum1)
		for (int i = 1; i < N + 1; i++) {
			for (int j = 1; j < N + 1; j+=2) {
				int idx = (N + 2)*i + j;
				if (i%2 == 1)
					idx++;
				double resid = (1 / (h*h))*(4 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - (N + 2)] - u[idx + (N + 2)]) - f[idx];

				sum1 += resid*resid;
				u_temp[idx] = h*h*f[idx] + u[idx - 1] + u[idx + 1] + u[idx - (N + 2)] + u[idx + (N + 2)];
				u_temp[idx] = 0.25*u_temp[idx];
			}
		}
		#pragma omp flush(u_temp)
		#pragma omp parallel for collapse(2) reduction(+:sum2)
		for (int i = 1; i < N + 1; i++) {
			for (int j = 1; j < N + 1; j+=2) {
				int idx = (N + 2)*i + j;
				if (i%2 == 0)
					idx++;
				double resid = (1 / (h*h))*(4 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - (N + 2)] - u[idx + (N + 2)]) - f[idx];

				sum2 += resid*resid;
				u_temp[idx] = h*h*f[idx] + u_temp[idx - 1] + u_temp[idx + 1] + u_temp[idx - (N + 2)] + u_temp[idx + (N + 2)];
				u_temp[idx] = 0.25*u_temp[idx];
			}
		}
		#pragma omp flush(u_temp)
		res = sqrt(sum1 + sum2);
		printf("res %d = %f\n", itr_num, res);
		double *uTemp = u_temp;
        	u_temp = u;
		u = uTemp;
		itr_num++;

	}
	//free(u_temp);

}


int main(int argc, char** argv) {

	int max_itr = 40;
	
	omp_set_num_threads(4);
	for (int pwr = 0; pwr < 4; pwr++) {
		int N = 2 * pow(10, pwr) + 5;
		printf("\nN=%d\n", N);




		double* u = (double *)calloc(sizeof(double), (N + 2)*(N + 2));
		double* f = (double*)malloc((N + 2)*(N + 2) * sizeof(double));
		for (int j = 0; j < (N + 2)*(N + 2); j++) {
			f[j] = 1;
		}
		double t = omp_get_wtime();

		gaussSeidel(max_itr, 0.1, N, u, f);
		double time = omp_get_wtime() - t;
		printf("time = %f\n", time);



		free(f);
		free(u);
	}
  
  return 0;
}

