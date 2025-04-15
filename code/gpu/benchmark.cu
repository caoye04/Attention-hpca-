#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <math.h> 
#include <float.h> 

void square_attention (int n, float* Q, float* K, float* V, float* Y);
extern const char* version_name;
double wall_time ()
{
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
}

void read_from_file(float* arr, int len, const char* file_name, int idx) {
  char full_file_name[100];
  sprintf(full_file_name, "%s%d", file_name, idx);
  FILE* file = fopen(full_file_name, "rb");
  if (file == NULL) {
      perror("File open error");
      return;
  }
  size_t read_count = fread(arr, sizeof(float), len, file);
  if (read_count != len) {
      if (feof(file)) {
          printf("Warning: file length < required length\n");
      } else {
          perror("Read file error");
      }
  }
  fclose(file); 
}

void die (const char* message)
{
  perror (message);
  exit (EXIT_FAILURE);
}

int main(int argc, char const *argv[])
{
	printf ("Description:\t%s\n\n", version_name);

	/* Test size */ 
	int test_sizes[] =
    {63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 383, 384, 385, 447, 448, 449, 511, 512, 513, 575, 576, 577, 639, 640, 641, 703, 704, 705, 767, 768, 769, 831, 832, 833, 895, 896, 897, 959, 960, 961, 1023, 1024, 1025, 1087, 1088, 1089, 1151, 1152, 1153, 1215, 1216, 1217, 1279, 1280, 1281, 1343, 1344, 1345, 1407, 1408, 1409, 1471, 1472, 1473, 1535, 1536, 1537, 1599, 1600, 1601, 1663, 1664, 1665, 1727, 1728, 1729, 1791, 1792, 1793, 1855, 1856, 1857, 1919, 1920, 1921, 1983, 1984, 1985, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193};

	int nsizes = sizeof(test_sizes)/sizeof(test_sizes[0]);
	int nmax = test_sizes[nsizes-1];
	float *cpu_Q, *gpu_Q, *cpu_K, *gpu_K, *cpu_V, *gpu_V, *cpu_Y, *gpu_Y, *cpu_Yt;

	cpu_Q  = (float*) malloc (nmax * nmax * sizeof(float));
	cpu_K  = (float*) malloc (nmax * nmax * sizeof(float));
	cpu_V  = (float*) malloc (nmax * nmax * sizeof(float));
	cpu_Y  = (float*) malloc (nmax * nmax * sizeof(float));
	cpu_Yt = (float*) malloc (nmax * nmax * sizeof(float));

	cudaMalloc(&gpu_Q, sizeof(float) * nmax * nmax);
	cudaMalloc(&gpu_K, sizeof(float) * nmax * nmax);
	cudaMalloc(&gpu_V, sizeof(float) * nmax * nmax);
	cudaMalloc(&gpu_Y, sizeof(float) * nmax * nmax);

	/* For each test size */
	double res = 0.0;
	double count = 0.0;
	for (int isize = 0; isize < sizeof(test_sizes)/sizeof(test_sizes[0]); ++isize)
	{
		int n = test_sizes[isize];

		read_from_file (cpu_Q,  n*n, "/home/2024-fall/dataset/hw3/Q_value/q_", isize);
		read_from_file (cpu_K,  n*n, "/home/2024-fall/dataset/hw3/K_value/k_", isize);
		read_from_file (cpu_V,  n*n, "/home/2024-fall/dataset/hw3/V_value/v_", isize);
		read_from_file (cpu_Yt, n*n, "/home/2024-fall/dataset/hw3/output_value/output_", isize);
		memset (cpu_Y, 0, n * n * sizeof(float));

		cudaMemcpy(gpu_Q, cpu_Q, sizeof(float) * n*n, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_K, cpu_K, sizeof(float) * n*n, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_V, cpu_V, sizeof(float) * n*n, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_Y, cpu_Y, sizeof(float) * n*n, cudaMemcpyHostToDevice);
		
	    /* Measure performance (in Gflops/s). */
		double Gflops_s, seconds = -1.0;
	    double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
	    /* Time a "sufficiently long" sequence of calls to reduce noise */
	    int n_iterations = 0;

		for (n_iterations = 1; seconds < timeout; n_iterations *= 2)
		{
			/* Warm-up */
			square_attention (n, gpu_Q, gpu_K, gpu_V, gpu_Y);
			cudaDeviceSynchronize();
			seconds = -wall_time();

			for (int it = 0; it < n_iterations; ++it)
			{	
				square_attention (n, gpu_Q, gpu_K, gpu_V, gpu_Y);
				cudaDeviceSynchronize();
			}
				
			seconds += wall_time();
			/*  compute Mflop/s rate */
			Gflops_s = 4.e-9 * n_iterations * n * n * n / seconds;
		}

		memset (cpu_Y , 0, n * n * sizeof(float));
		cudaMemcpy(gpu_Y, cpu_Y, sizeof(float) * n*n, cudaMemcpyHostToDevice);
		
		square_attention (n, gpu_Q, gpu_K, gpu_V, gpu_Y);
		cudaDeviceSynchronize();

		printf ("Size: %d\tGflop/s: %.3g (%d iter, %.3f seconds)\n", n, Gflops_s, n_iterations, seconds);
		res += Gflops_s;
		count += 1;

		cudaMemcpy(cpu_Y, gpu_Y, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
		double total_err = 0.0;
		for (int i = 0; i < n*n ; ++i)
		total_err += abs(cpu_Y[i] - cpu_Yt[i]);
		// printf("total_err: %.8lf\n", total_err);
		if (total_err > 100*n*FLT_EPSILON*n){
			die("*** FAILURE *** Error in calculation exceeds componentwise error bounds.\n" );
		}
	}
	cudaFree(gpu_Q);
	cudaFree(gpu_K);
	cudaFree(gpu_V);
	cudaFree(gpu_Y);

	free(cpu_Q);
	free(cpu_K);
	free(cpu_V);
	free(cpu_Y);
	free(cpu_Yt);

	res /= count;
	printf("Average %lf \n",res);
	return 0;
}