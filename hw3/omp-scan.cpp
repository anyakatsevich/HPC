#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
//#export OMP_NUM_THREADS = 4

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}


void scan_omp(long* prefix_sum, const long* A, long n, int p) {
  // TODO: implement multi-threaded OpenMP scan
 if (n == 0) return; 
 prefix_sum[0] = 0;
  
  long inc = (n-1)/p;
  long* end_pts = (long* )malloc((p+1) * sizeof(long));
  long* totals = (long* )malloc(p * sizeof(long));
  for (int i = 0; i < p; i++)
    end_pts[i] = 1 + i * inc;
  end_pts[p] = n;
  
  omp_set_num_threads(p);
  #pragma omp parallel for schedule(dynamic)
  for (int k = 0; k < p; k++)
  { 
    long k_i = end_pts[k];
    long k_f = end_pts[k+1];
    prefix_sum[k_i] = A[k_i - 1];
    for (long i = k_i+1; i < k_f; i++)
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    totals[k] = prefix_sum[k_f - 1];
  } 
  
  for (int i = 1; i < p; i++)
     totals[i] += totals[i-1];
  #pragma omp parallel for schedule(dynamic)
  for (int i = 1; i < p; i++) {
     for (int j = end_pts[i]; j < end_pts[i+1]; j++)
      prefix_sum[j] += totals[i-1];
  }


}

int main() {
  int max_threads =  omp_get_max_threads();
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  
  for (long i = 0; i < N; i++) A[i] = rand();
  
  
  double tt_seq = omp_get_wtime();
  scan_seq(B0, A, N);
  tt_seq = omp_get_wtime() - tt_seq;
  printf("sequential-scan took %fs\n\n", tt_seq);
  
  for(int p = 1; p<=max_threads; p++) {
  printf("%d threads:\n", p); 
  double tt = omp_get_wtime();
  scan_omp(B1, A, N, p);
  tt = omp_get_wtime() - tt;
  printf("parallel-scan took %fs\n", tt);
  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n\n", err);
  
  }

  free(A);
  free(B0);
  free(B1);
  return 0;
}
