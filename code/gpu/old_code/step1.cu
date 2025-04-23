#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

const char *version_name = "Optimized implementation.";

__global__ void matmul_QKT(int n, float *Q, float *K, float *QKT, float scale)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += Q[row * n + k] * K[col * n + k];
        }
        QKT[row * n + col] = sum * scale;
    }
}

__global__ void softmax_kernel(int n, float *QKT)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n)
    {
        float max_val = -INFINITY;
        for (int j = 0; j < n; j++)
        {
            if (QKT[row * n + j] > max_val)
            {
                max_val = QKT[row * n + j];
            }
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < n; j++)
        {
            QKT[row * n + j] = expf(QKT[row * n + j] - max_val);
            sum_exp += QKT[row * n + j];
        }

        for (int j = 0; j < n; j++)
        {
            QKT[row * n + j] /= sum_exp;
        }
    }
}

__global__ void matmul_softmax_V(int n, float *softmax_QKT, float *V, float *Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += softmax_QKT[row * n + k] * V[k * n + col];
        }
        Y[row * n + col] = sum;
    }
}

void square_attention(int n, float *gpu_Q, float *gpu_K, float *gpu_V, float *gpu_Y)
{

    float *gpu_QKT;
    cudaMalloc(&gpu_QKT, n * n * sizeof(float));

    float scale = 1.0f / sqrtf(n);

    dim3 block_dim(16, 16);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    dim3 softmax_block_dim(256);
    dim3 softmax_grid_dim((n + softmax_block_dim.x - 1) / softmax_block_dim.x);

    matmul_QKT<<<grid_dim, block_dim>>>(n, gpu_Q, gpu_K, gpu_QKT, scale);

    softmax_kernel<<<softmax_grid_dim, softmax_block_dim>>>(n, gpu_QKT);

    matmul_softmax_V<<<grid_dim, block_dim>>>(n, gpu_QKT, gpu_V, gpu_Y);

    cudaFree(gpu_QKT);

    cudaDeviceSynchronize();
}