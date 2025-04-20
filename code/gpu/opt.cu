#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

const char* version_name = "Warp-optimized Softmax implementation.";

#define TILE_SIZE 16
#define WARP_SIZE 32

__global__ void matmul_QKT_shared(int n, float* Q, float* K, float* QKT, float scale) {
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < n && tile * TILE_SIZE + tx < n)
            tile_Q[ty][tx] = Q[row * n + tile * TILE_SIZE + tx];
        else
            tile_Q[ty][tx] = 0.0f;
        
        if (col < n && tile * TILE_SIZE + ty < n)
            tile_K[ty][tx] = K[col * n + tile * TILE_SIZE + ty]; 
        else
            tile_K[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_Q[ty][k] * tile_K[k][tx];
        }

        __syncthreads();
    }
    
    if (row < n && col < n)
        QKT[row * n + col] = sum * scale;
}

__global__ void softmax_warp_optimized(int n, float* QKT) {
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (row < n) {
        float max_val = -INFINITY;
        for (int j = lane_id; j < n; j += WARP_SIZE) {
            float val = QKT[row * n + j];
            max_val = fmaxf(max_val, val);
        }
        
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        
        max_val = __shfl_sync(0xffffffff, max_val, 0);
        
        float sum_exp = 0.0f;
        for (int j = lane_id; j < n; j += WARP_SIZE) {
            float val = expf(QKT[row * n + j] - max_val);
            QKT[row * n + j] = val; // 存储中间结果
            sum_exp += val;
        }
        
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
        
        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
        
        for (int j = lane_id; j < n; j += WARP_SIZE) {
            QKT[row * n + j] /= sum_exp;
        }
    }
}

__global__ void matmul_softmax_V_shared(int n, float* softmax_QKT, float* V, float* Y) {
    __shared__ float tile_S[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < n && tile * TILE_SIZE + tx < n)
            tile_S[ty][tx] = softmax_QKT[row * n + tile * TILE_SIZE + tx];
        else
            tile_S[ty][tx] = 0.0f;
            
        if (tile * TILE_SIZE + ty < n && col < n)
            tile_V[ty][tx] = V[(tile * TILE_SIZE + ty) * n + col];
        else
            tile_V[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_S[ty][k] * tile_V[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n)
        Y[row * n + col] = sum;
}

void square_attention(int n, float* gpu_Q, float* gpu_K, float* gpu_V, float* gpu_Y) {
    float* gpu_QKT;
    cudaMalloc(&gpu_QKT, n * n * sizeof(float));
    
    float scale = 1.0f / sqrtf(n);
    
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((n + TILE_SIZE - 1) / TILE_SIZE, 
                  (n + TILE_SIZE - 1) / TILE_SIZE);
    
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (n + warps_per_block - 1) / warps_per_block;
    
    matmul_QKT_shared<<<grid_dim, block_dim>>>(n, gpu_Q, gpu_K, gpu_QKT, scale);

    softmax_warp_optimized<<<num_blocks, threads_per_block>>>(n, gpu_QKT);
    
    matmul_softmax_V_shared<<<grid_dim, block_dim>>>(n, gpu_QKT, gpu_V, gpu_Y);
    
    cudaFree(gpu_QKT);
    
    cudaDeviceSynchronize();
}