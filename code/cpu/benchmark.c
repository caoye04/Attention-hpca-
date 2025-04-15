#define _POSIX_C_SOURCE 199309L
#include "mpi.h"  
#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#include <sys/time.h> // For struct timeval, gettimeofday
#define MPI_TYPE MPI_FLOAT

extern const char* version_name;
extern void square_attention (int lda, float* Q, float* K, float* V, float* Y, int rank, int size);

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
  if (read_count != (size_t) len) {
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

int main(int argc, char* argv[])
{

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
    {
      printf ("Description:\t%s\n\n", version_name);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Test size */ 
    int test_sizes[] =
     {63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 383, 384, 385, 447, 448, 449, 511, 512, 513, 575, 576, 577, 639, 640, 641, 703, 704, 705, 767, 768};//, 769, 831, 832, 833, 895, 896, 897, 959, 960, 961, 1023, 1024, 1025, 1087, 1088, 1089, 1151, 1152, 1153, 1215, 1216, 1217, 1279, 1280, 1281, 1343, 1344, 1345, 1407, 1408, 1409, 1471, 1472, 1473, 1535, 1536, 1537, 1599, 1600, 1601, 1663, 1664, 1665, 1727, 1728, 1729, 1791, 1792, 1793, 1855, 1856, 1857, 1919, 1920, 1921, 1983, 1984, 1985, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193};

    int nsizes = sizeof(test_sizes)/sizeof(test_sizes[0]);
    int nmax = test_sizes[nsizes-1];
    float* buf;
    MPI_Alloc_mem(sizeof(float) * 5 * nmax * nmax, MPI_INFO_NULL, &buf);
  
    double start, end, res = 0.0, count = 0.0;
    int all_size = sizeof(test_sizes)/sizeof(test_sizes[0]);
    for (int isize = 0; isize < all_size; ++isize)
    {
        int n = test_sizes[isize];
        float* Q = buf + 0;
        float* K = Q + nmax*nmax;
        float* V = K + nmax*nmax;
        float* Y = V + nmax*nmax;
        float* Yt = Y + nmax*nmax;
        memset (Y, 0, n * n * sizeof(float));
        if (rank == 0)
        {
          read_from_file (Q,  n*n, "/home/2024-fall/dataset/hw3/Q_value/q_", isize);
          read_from_file (K,  n*n, "/home/2024-fall/dataset/hw3/K_value/k_", isize);
          read_from_file (V,  n*n, "/home/2024-fall/dataset/hw3/V_value/v_", isize);
          read_from_file (Yt, n*n, "/home/2024-fall/dataset/hw3/output_value/output_", isize);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        float Gflops_s, seconds = -1.0;
        float timeout = 0.1; // "sufficiently long" := at least 1/10 second.
        int  n_iterations = 0;

        for (n_iterations = 1; seconds < timeout; n_iterations *= 2)
        {
          /* Warm-up */
          square_attention (n, Q, K, V, Y, rank, size);

          MPI_Barrier(MPI_COMM_WORLD);
          start = MPI_Wtime();
          /* Benchmark n_iterations runs of square_gemm */
          for (int it = 0; it < n_iterations; ++it)
            square_attention (n, Q, K, V, Y, rank, size);

          MPI_Barrier(MPI_COMM_WORLD); 
          end = MPI_Wtime() - start;
          seconds = end;
          /*  compute Mflop/s rate */
          Gflops_s = 4.e-9 * n_iterations * n * n * n / seconds; // Roughly two GEMM operations
        }

        if (rank == 0)
        {
          printf ("Size: %d\tGflop/s: %.3g (%d iter, %.3f seconds)\n", n, Gflops_s, n_iterations, seconds);
        	res += Gflops_s;
		      count += 1;
        }
        memset (Y, 0, n * n * sizeof(float));
        square_attention (n, Q, K, V, Y, rank, size);

        // check res in rank 0 !!!
        if (rank == 0)
        {
          double total_err = 0.0;
          for (int i = 0; i < n*n ; ++i)
            total_err += abs(Y[i] - Yt[i]);
          // printf("total_err: %.8lf\n", total_err);
          if (total_err > 100*n*FLT_EPSILON*n){
            die("*** FAILURE *** Error in calculation exceeds componentwise error bounds.\n" );
          }
        }
        MPI_Barrier(MPI_COMM_WORLD); 
    }
    if (rank == 0)
    {
      res /= count;
      printf("Average %lf \n",res);
    }

    MPI_Free_mem(buf);
    MPI_Finalize();

    return 0;
}
