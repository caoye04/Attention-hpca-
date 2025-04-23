#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

const char* version_name = "finally opt!";

#define TILE_SIZE 64
#define REG_TILE_SIZE 4
#define BLOCK_DIM (TILE_SIZE / REG_TILE_SIZE)
#define WARP_SIZE 32

__global__ void matmul_QKT_register_blocked(int n, float* Q, float* K, float* QKT, float scale) {
    __shared__ float s_Q[TILE_SIZE][TILE_SIZE+1];
    __shared__ float s_K[TILE_SIZE][TILE_SIZE+1]; 
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty * REG_TILE_SIZE;
    int col = bx * TILE_SIZE + tx * REG_TILE_SIZE;
    
    float sum[REG_TILE_SIZE][REG_TILE_SIZE] = {0.0f};
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
            int r = by * TILE_SIZE + i;
            
            for (int j = tx * 2; j < TILE_SIZE; j += blockDim.x * 2) {
                int c = t * TILE_SIZE + j;
                
                if (r < n && c < n) {
                    s_Q[i][j] = Q[r * n + c];
                } else {
                    s_Q[i][j] = 0.0f;
                }
                
                if (r < n && c + 1 < n) {
                    s_Q[i][j+1] = Q[r * n + c + 1];
                } else {
                    s_Q[i][j+1] = 0.0f;
                }
            }
        }
        
        for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
            int r = t * TILE_SIZE + i;
            
            for (int j = tx * 2; j < TILE_SIZE; j += blockDim.x * 2) {
                int c = bx * TILE_SIZE + j;
                
                if (r < n && c < n) {
                    s_K[j][i] = K[c * n + r];
                } else {
                    s_K[j][i] = 0.0f;
                }
                
                if (r < n && c + 1 < n) {
                    s_K[j+1][i] = K[(c+1) * n + r];
                } else {
                    s_K[j+1][i] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; ++k) {
            float q_vals[REG_TILE_SIZE];
            #pragma unroll
            for (int i = 0; i < REG_TILE_SIZE; ++i) {
                q_vals[i] = s_Q[ty * REG_TILE_SIZE + i][k];
            }
            
            float k_vals[REG_TILE_SIZE];
            #pragma unroll
            for (int j = 0; j < REG_TILE_SIZE; ++j) {
                k_vals[j] = s_K[tx * REG_TILE_SIZE + j][k];
            }
            
            #pragma unroll
            for (int i = 0; i < REG_TILE_SIZE; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_TILE_SIZE; ++j) {
                    sum[i][j] += q_vals[i] * k_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    for (int i = 0; i < REG_TILE_SIZE; ++i) {
        int r = row + i;
        if (r < n) {
            for (int j = 0; j < REG_TILE_SIZE; ++j) {
                int c = col + j;
                if (c < n) {
                    QKT[r * n + c] = sum[i][j] * scale;
                }
            }
        }
    }
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
            QKT[row * n + j] = val;
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

__global__ void matmul_softmax_V_register_blocked(int n, float* softmax_QKT, float* V, float* Y) {
    __shared__ float s_softmax[TILE_SIZE][TILE_SIZE+1]; 
    __shared__ float s_V[TILE_SIZE][TILE_SIZE+1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty * REG_TILE_SIZE;
    int col = bx * TILE_SIZE + tx * REG_TILE_SIZE;
    
    float sum[REG_TILE_SIZE][REG_TILE_SIZE] = {0.0f};
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
            int r = by * TILE_SIZE + i;
            
            for (int j = tx * 2; j < TILE_SIZE; j += blockDim.x * 2) {
                int c = t * TILE_SIZE + j;
                
                if (r < n && c < n) {
                    s_softmax[i][j] = softmax_QKT[r * n + c];
                } else {
                    s_softmax[i][j] = 0.0f;
                }
                
                if (r < n && c + 1 < n) {
                    s_softmax[i][j+1] = softmax_QKT[r * n + c + 1];
                } else {
                    s_softmax[i][j+1] = 0.0f;
                }
            }
        }
        
        for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
            int r = t * TILE_SIZE + i;
            
            for (int j = tx * 2; j < TILE_SIZE; j += blockDim.x * 2) {
                int c = bx * TILE_SIZE + j;
                
                if (r < n && c < n) {
                    s_V[j][i] = V[r * n + c];
                } else {
                    s_V[j][i] = 0.0f;
                }
                
                if (r < n && c + 1 < n) {
                    s_V[j+1][i] = V[r * n + c + 1];
                } else {
                    s_V[j+1][i] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            float softmax_vals[REG_TILE_SIZE];
            #pragma unroll
            for (int i = 0; i < REG_TILE_SIZE; ++i) {
                softmax_vals[i] = s_softmax[ty * REG_TILE_SIZE + i][k];
            }
            
            float v_vals[REG_TILE_SIZE];
            #pragma unroll
            for (int j = 0; j < REG_TILE_SIZE; ++j) {
                v_vals[j] = s_V[tx * REG_TILE_SIZE + j][k];
            }
            
            #pragma unroll
            for (int i = 0; i < REG_TILE_SIZE; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_TILE_SIZE; ++j) {
                    sum[i][j] += softmax_vals[i] * v_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    for (int i = 0; i < REG_TILE_SIZE; ++i) {
        int r = row + i;
        if (r < n) {
            for (int j = 0; j < REG_TILE_SIZE; ++j) {
                int c = col + j;
                if (c < n) {
                    Y[r * n + c] = sum[i][j];
                }
            }
        }
    }
}

void square_attention(int n, float* gpu_Q, float* gpu_K, float* gpu_V, float* gpu_Y) {
    float* gpu_QKT;
    cudaMalloc(&gpu_QKT, n * n * sizeof(float));
    
    float scale = 1.0f / sqrtf(n);
    
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_QKT_register_blocked<<<grid_dim, block_dim>>>(n, gpu_Q, gpu_K, gpu_QKT, scale);
    
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (n + warps_per_block - 1) / warps_per_block;
    softmax_warp_optimized<<<num_blocks, threads_per_block>>>(n, gpu_QKT);
    
    matmul_softmax_V_register_blocked<<<grid_dim, block_dim>>>(n, gpu_QKT, gpu_V, gpu_Y);
    
    cudaFree(gpu_QKT);
    cudaDeviceSynchronize();
}