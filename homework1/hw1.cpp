#include <stdio.h>
#include <math.h>
#include <"utils.h">


double computeResidual(int N, double *u, double *f) {
	double res = 0;
	for (int i = 1; i < N + 1; i++)
		res = res + pow((-1 * u[i - 1] + 2 * u[i] - 1 * u[i + 1])*(N + 1.0)*(N+1.0) - f[i], 2);
	res = sqrt(res);
	return res;

}

void jacobi(int N, double *u, double *f) {
	double res_init = computeResidual(N, u, f);
	printf("initial residual = %10f\n", res_init);
	double res = 0;
	int num_itr = 0;
	double store, assign;
	Timer t;
	t.tic();   
	double time; 
	while (num_itr < 5000) {
		if (num_itr == 100) {
			time = t.toc();
			printf("time for 100 iterations = %10f\n", time);
		}
		store = 0.5*f[1] / ((N + 1.0)*(N + 1.0)) + 0.5*(u[0] + u[2]);
		for (int i = 1; i <= N ; i++) {
			assign = store;
			if (i < N) {
				store = 0.5*f[i + 1] / ((N + 1.0)*(N + 1.0)) + 0.5*(u[i] + u[i + 2]);
			}
			u[i] = assign;
		}
		res = computeResidual(N, u, f);
		//printf("init residual = %10f, residual #%d = %10f\n", res_init, num_itr, res);
		if (res / res_init < 1e-6)
			break;
		num_itr++;
	}
	printf("final residual =%10f\n", res);
	printf("total number of iterations = %d\n", num_itr);
}

void gaussSeidel(int N, double *u, double *f) {
	double res_init = computeResidual(N, u, f);
	printf("initial residual = %10f\n", res_init);
	double res = 0;
	int num_itr = 0;
	Timer t;
	t.tic(); 
	double time;
	while (num_itr < 5000) {
		if (num_itr == 100) {
			time = t.toc();
			printf("time for 100 iterations = %10f\n", time);
		}
		for (int i = 1; i <= N; i++)
			u[i] = 0.5*f[i] / ((N + 1.0)*(N + 1.0)) + 0.5*(u[i - 1] + u[i + 1]);

		res = computeResidual(N, u, f);
		//printf("init residual = %10f, residual #%d = %10f\n", res_init, num_itr, res);
		if (res / res_init < 1e-6)
			break;
		num_itr++;
	}
	printf("final residual =%10f\n", res);
	printf("total number of iterations = %d\n", num_itr);

}

int main(int argc, char** argv) {
	int Ns[2] = {100,10000};
	for (int i = 0; i < 2; i++) {
		int N = Ns[i];
		double* u = (double*)malloc((N + 2) * sizeof(double));
		double* f = (double*)malloc((N + 2) * sizeof(double));
		for (int j = 0; j< N + 2; j++) {
			u[j] = 0;
			f[j] = 1;
		}

		printf("Starting Jacobi method:\n");
		jacobi(N, u, f);
		for (int j = 0; j< N + 2; j++) {
			u[j] = 0;
		}

		printf("Starting Gauss Seidel method:\n");
		gaussSeidel(N, u, f);

		free(u);
		free(f);
	}





	system("pause");
	return 0;
}
