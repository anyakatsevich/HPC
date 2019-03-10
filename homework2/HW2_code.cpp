#include <stdio.h>
#include <math.h>
#include "utils.h"

double computeResidual(int k, int *odd_idxs, int *even_idxs, double *u_odd, double *u_even, double *f) {
	int N = 2 * k - 1;
   double res=0;
   double Au_ij, double diff_ij;
   for (int i = 1; i < k + 1; i++) {
	   for (int j = 0; j < 2 * k - 1; j++) {
		   idx = i*k + odd_idxs[j];
		   Au_ij= 4*u_odd[idx] - u_even[idx] - u_even[idx + 1] - u_even[idx + 2 * k + 1] - u_even[idx - 2 * k];
		   diff_ij = Au_ij*(N + 1.0)*(N + 1.0) - f[2 * idx + 1];
		   res = res + diff_ij*diff_ij;
		   idx = i*k + even_idxs[j];
		   Au_ij = 4*u_even[2*idx] - u_odd[idx - 1] - u_odd[idx + 1] - u_odd[idx + 2 * k] - u_odd[idx - 2 * k - 1];
		   diff_ij = Au_ij*(N + 1.0)*(N + 1.0) - f[2 * idx + 1];
		   res = res + diff_ij*diff_ij;
	   }

   }
   return sqrt(res);
   // TO BE DONE
  /* for (int i = 1; i < N+1; i++)
	   for (int j = 1; j < N + 1; j++) {
		   Au_ij = 4 * u[i + (N + 2)*j] - u[i - 1 + (N + 2)*j] - u[i + (N + 2)*(j - 1)] - u[i + 1 + (N + 2)*j] - u[i + (N + 2)*(j + 1)];
		   diff_ij = Au_ij*(N+1.0)*(N+1.0) - f[i + (N + 2)*j];
		   res = res + diff_ij*diff_ij;
	   }
    res = sqrt(res);
    return res;*/

}

void jacobi( int k, int *odd_idxs, int *even_idxs, double *u_odd, double *u_even, double *f) {
	int N = 2 * k - 1;
	int i;
	
	//double* u_even = (double *)malloc(sizeof(double), 2 * k*k + 2 * k + 1);
	//double* u_odd = (double *)malloc(sizeof(double), 2 * k*k + 2 * k);
	double* u_evenNew = (double *)calloc(sizeof(double), 2 * k*k + 2 * k + 1);
	double* u_oddNew = (double *)calloc(sizeof(double), 2 * k*k + 2 * k);
	/*for (int i = 0; i < 2 * k*k + 2 * k; i++) {
		u_even[i] = u[2 * i];
		u_odd[i] = u[2 * i + 1];
	}
	u_even[2*k*k + 2*k] = u[(2 * k + 1)*(2 * k + 1)-1];*/
	double res_init = computeResidual(N, u_even, u_odd, f);
	printf("residual #0 = %10f\n", res_init);
	double res = 0;
	int num_itr = 0;
	int idx;
  
    while (num_itr < 5000){
		for (int i = 1; i < k + 1; i++) {
			for (int j = 0; j < 2*k-1; j++) {
				idx = i*k + odd_idxs[j];
				u_oddNew[idx] = 0.25*(f[2 * idx + 1] / ((N + 1.0)*(N + 1.0)) + u_even[idx] + u_even[idx + 1] + u_even[idx + 2 * k + 1] + u_even[idx - 2 * k]);
				idx = i*k + even_idxs[j];
				u_evenNew[idx] = 0.25*(f[2*idx]/((N+1.0)*(N+1.0)) +  u_odd[idx - 1] + u_odd[idx + 1] + u_odd[idx + 2 * k] + u_odd[idx - 2 * k - 1]);
			}

		}
	
		u_even = u_evenNew;
		u_odd = u_oddNew;
	
		res = computeResidual(N, u_even, u_odd, f);
		printf("residual #%d = %10f\n", num_itr, res);
		if (res/res_init < 1e-6)
			break;
		num_itr++;
  }

  printf("total number of iterations = %10f\n",num_itr);
}

/*void gaussSeidel( int N, double *u, double *f) {
  double res_init = computeResidual(N,u,f);
  printf("residual 0 = %10f\n", res_init);
  double res = 0;
  int num_itr = 0;
  
  
  while (num_itr < 5000){
	for (int i = 1; i <= N; i++)
		u[i] = 0.5*f[i]/((N+1.0)*(N+1.0))+0.5*(u[i-1]+u[i+1]);
	
	res = computeResidual(N, u, f);
        printf("residual #%d = %10f\n", num_itr, res);
	if (res/res_init < 1e-6)
		break;
	num_itr++;
  }

  printf("total number of iterations = %10f\n",num_itr);

}*/

int main(int argc, char** argv) {
	int k = 10;
	int* odd_idxs = (int *)malloc(sizeof(int)*(2 * k - 1));
	int* even_idxs = (int *)malloc(sizeof(int)*(2 * k - 1));
	for (i = 0; i < k - 1; i++) {
		odd_idxs[i] = i + 1;
		even_idxs[i] = i + 1;
	}
	odd_idxs[k - 1] = k + 1;
	even_idxs[k - 1] = k;
	for (i = k; i < 2 * k - 1; i++) {
		odd_idxs[i] = i + 2;
		even_idxs[i] = i + 2;
	}
	double* u_even = (double *)calloc(sizeof(double), 2 * k*k + 2 * k + 1);
	double* u_odd = (double *)calloc(sizeof(double), 2 * k*k + 2 * k);
	double *f = (double*)malloc((2*k + 1) * sizeof(double));
	//int Ns[2] = { 100,200 };
  //for (int i = 0; i < 2;i++){
	 // int N = Ns[i];
	 // double* u = (double*) malloc((N+2)*sizeof(double));
	 // double* f = (double*) malloc((N+2)*sizeof(double));
	  for (int j = 0; j< 2*k+1; j++){
		//  u[j]=0;
		  f[j]=1;
	}

	printf("Starting Jacobi method:\n");
	//jacobi(N,u,f);
	jacobi(k, odd_idxs, even_idxs, u_odd, u_even, f);
	//printf("Starting Gauss Seidel method:\n");
	//gaussSeidel(N,u,f);
  }

  
//    Timer t;
//    t.tic();   
 //   double time = t.toc();
    
    free(u);
    free(f);

  return 0;
}




