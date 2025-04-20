# 并行计算基础HW3—Attention

## 1. 作业背景

Attention 机制最初用于提升序列到序列模型（Seq2Seq）在处理长文本时的效果，通过允许模型动态聚焦输入序列的不同部分，解决了传统方法的瓶颈。它通过计算查询（Q）、键（K）和值（V）之间的相似度来加权每个值，从而生成新的表示。Attention 被广泛应用于 NLP、计算机视觉等领域，尤其是 Transformer 模型中。随着模型应用的扩展，优化 Attention 计算的效率变得尤为重要，常见的优化技巧包括稀疏化计算、低秩近似、多机分布式计算以及 GPU 加速等方法，以减少计算和内存开销，提升效率。

## 2. 作业任务：方阵的 Attention 机制计算

### 2.1. 任务描述

完成 Attention 算子在GPU 单卡上的实现与优化。Attention 算子：

$$
Y=\operatorname{Softmax}\left(\frac{Q K^{\mathrm{T}}}{\sqrt{N}}\right) V
$$


其中，$Q, K, V$ 均为 $N \times N$ 的单精度行优先存储的稠密矩阵。
其中 Softmax 的定义为

$$
\operatorname{Softmax}\left(\left[z_1, \ldots, z_n\right]\right)=\left[\frac{e^{z_1}}{\sum_{j=1}^n e^{z_j}}, \ldots, \frac{e^{z_n}}{\sum_{j=1}^n e^{z_j}}\right]
$$


在 Attention 计算中，Softmax 的计算是逐行进行的，具体算法实现可以参考＂code／cpu／naive．c＂中的代码。在熟悉代码核心后，优化版本的程序实现在＂code／gpu／opt．cu＂中。

### 2.2 正确性验证

1．采用 float 单精度浮点数据类型进行运算，运算结果通过作业基础代码中的正确性验证，eps 为机器的单精度极小值，约为 $10^{-6}$ 左右。

$$
\| \text { custom-attention }(n, Q, K, V)-\text { Label } \|<100 n^2 \epsilon
$$

其中，$\|\cdot\|$ 表示将矩阵逐元素取绝对值求和（即向量的 1 范数）。

2．Attention 的主要计算开销在两次矩阵乘法，其计算复杂度为 $O\left(N^3\right)$ ，在计算性能指标的时候采用 $\left(4 N^3\right)$ 计算，如果采用了一些非 $O\left(N^3\right)$ 算法而导致通过不了正确性测试，这种情况可以适当且合理地放宽精度的要求，但需要在作业报告中指出。

3．开展必要的性能分析，比如某些矩阵规模性能出现明显的降低，可以采用性能分析工具进行性能分析。

## 3. 作业评分

1. 通过正确性检验（ $20 \%$ ）。

2. 评测 Attention 算子在不同输入（共 102 个测例）下的性能结果，按照提交后的性能排序结果，以及代码质量进行打分（ $50 \%$ ）。

3. 详细描述在实现 Attention 算子中采取的优化手段，代码对应的部分，以及对应的实验结果，可以采用性能工具或者模型来解释目前取得的性能结果（ $20 \%$ ）。

4. 给出一张完整的实验结果图，描述当前算法的性能，横坐标为矩阵规模，纵坐标为 Gflop $/ s(10 \%)$ 。

5. 可以参考矩阵乘法相关的外部库（如 BLAS 或 BLIS 等数学库）的实现与优化思路，但禁止直接使用外部库作为作业成果。

> 作业提示：
>
> 1．在保证正确性的前提下，可对计算流程中的某些冗余部分进行删减。
>
> 2．本作业的核心模块是两次矩阵乘法，可以借鉴第一次作业 SGEMM的优化思路，在 GPU 上优化好矩阵运算的子模块。
>
> 3．有任何问题欢迎与助教和老师交流。

## 4. 相关代码：

```c++
//   /code/cpu/naive.c
#include "mpi.h"  
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
const char* version_name = "Naive implementation.";
 
void square_attention (int n, float* Q, float* K, float* V, float* Y, int rank, int size)
{
    if (rank == 0)
    {
    // QK^T 矩阵初始化
    float* QK_T = (float*)malloc(n * n * sizeof(float));
    if (!QK_T) {
        printf("Memory Allocation Error\n");
        return;
    }

    // 计算 Q * K^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            QK_T[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                QK_T[i * n + j] += Q[i * n + k] * K[j * n + k];  // Q * K^T
            }
        }
    }

    // 归一化 QK^T
    float scale = 1.0f / sqrtf(n);  // 对 QK^T 进行缩放
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            QK_T[i * n + j] *= scale;
        }
    }

    // Softmax 计算
    for (int i = 0; i < n; i++) {
        float max_val = -INFINITY;
        // 找到每行的最大值
        for (int j = 0; j < n; j++) {
            if (QK_T[i * n + j] > max_val) {
                max_val = QK_T[i * n + j];
            }
        }

        float sum_exp = 0;
        // 计算Softmax的分母
        for (int j = 0; j < n; j++) {
            QK_T[i * n + j] = expf(QK_T[i * n + j] - max_val);  // 减去最大值来避免溢出
            sum_exp += QK_T[i * n + j];
        }

        // 计算Softmax结果
        for (int j = 0; j < n; j++) {
            QK_T[i * n + j] /= sum_exp;
        }
    }

    // 计算 Y = softmax(QK^T) * V
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Y[i * n + j] = 0;  // 初始化 Y[i][j]
            for (int k = 0; k < n; k++) {
                Y[i * n + j] += QK_T[i * n + k] * V[k * n + j];  // softmax(QK^T) * V
            }
        }
    }

    // 释放内存
    free(QK_T);
    }

}
```

```cu
// /code/gpu/opt.cu
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

const char* version_name = "Optimized implementation.";

void square_attention (int n, float* gpu_Q, float* gpu_K, float* gpu_V, float* gpu_Y)
{

}
```

第一次作业GEMM的相关代码

```c++
const char* sgemm_desc = "Simple blocked sgemm with folded kernels.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
#define SMALL_BLOCK_SIZE 16
#define SMALL_BLOCK_M_SIZE 32
#define SMALL_BLOCK_N_SIZE 8
#define min(a,b) (((a)<(b))?(a):(b))

// 朴素矩阵乘法，处理小块和边界情况
static void do_block_opt(int lda, int M, int N, int K, float * restrict A, float * restrict B, float * restrict C)
{
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j) {
            register float b = B[k + j * lda];
            for (int i = 0; i < M; ++i)
                C[i + j * lda] += A[i + k * lda] * b;
        }
}

// 64×1矩阵乘法的AVX-512实现
static void do_block_avx_64k1(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[4]; 

    for (int block = 0; block < 4; block++) {
        c[block] = _mm512_load_ps(&C[block * 16]);
    }

    __m512 a[4];
    __m512 b;

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 4; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        b = _mm512_set1_ps(B[k + ldb * 0]);

        for (int block = 0; block < 4; block++) {
            c[block] = _mm512_fmadd_ps(a[block], b, c[block]);
        }
    }

    for (int block = 0; block < 4; block++) {
        _mm512_store_ps(&C[block * 16], c[block]);
    }
}

// 16×16矩阵乘法的AVX-512实现
static void do_block_avx_16k16(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[16];

    for (int j = 0; j < 16; j++) {
        c[j] = _mm512_load_ps(&C[ldc * j]);
    }

    __m512 a;
    __m512 b;

    for (int k = 0; k < K; k++) {
        a = _mm512_load_ps(&A[lda * k]);

        for (int j = 0; j < 16; j++) {
            b = _mm512_set1_ps(B[k + ldb * j]);
            c[j] = _mm512_fmadd_ps(a, b, c[j]);
        }
    }

    for (int j = 0; j < 16; j++) {
        _mm512_store_ps(&C[ldc * j], c[j]);
    }
}

// 32×8矩阵乘法的AVX-512实现
static void do_block_avx_32k8(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[2][8];

    for (int block = 0; block < 2; block++) {
        for (int j = 0; j < 8; j++) {
            c[block][j] = _mm512_load_ps(&C[ldc * j + block * 16]);
        }
    }

    __m512 a[2];
    __m512 b[8];

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 2; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        for (int j = 0; j < 8; j++) {
            b[j] = _mm512_set1_ps(B[k + ldb * j]);
        }

        for (int block = 0; block < 2; block++) {
            for (int j = 0; j < 8; j++) {
                c[block][j] = _mm512_fmadd_ps(a[block], b[j], c[block][j]);
            }
        }
    }

    for (int block = 0; block < 2; block++) {
        for (int j = 0; j < 8; j++) {
            _mm512_store_ps(&C[ldc * j + block * 16], c[block][j]);
        }
    }
}

// 48×8矩阵乘法的AVX-512实现
static void do_block_avx_48k8(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[3][8];

    for (int block = 0; block < 3; block++) {
        for (int j = 0; j < 8; j++) {
            c[block][j] = _mm512_load_ps(&C[ldc * j + block * 16]);
        }
    }

    __m512 a[3];
    __m512 b[8];

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 3; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        for (int j = 0; j < 8; j++) {
            b[j] = _mm512_set1_ps(B[k + ldb * j]);
        }

        for (int block = 0; block < 3; block++) {
            for (int j = 0; j < 8; j++) {
                c[block][j] = _mm512_fmadd_ps(a[block], b[j], c[block][j]);
            }
        }
    }

    for (int block = 0; block < 3; block++) {
        for (int j = 0; j < 8; j++) {
            _mm512_store_ps(&C[ldc * j + block * 16], c[block][j]);
        }
    }
}

// 64×4矩阵乘法的AVX-512实现
static void do_block_avx_64k4(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[4][4];

    for (int block = 0; block < 4; block++) {
        for (int j = 0; j < 4; j++) {
            c[block][j] = _mm512_load_ps(&C[ldc * j + block * 16]);
        }
    }

    __m512 a[4];
    __m512 b[4];

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 4; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        for (int j = 0; j < 4; j++) {
            b[j] = _mm512_set1_ps(B[k + ldb * j]);
        }

        for (int block = 0; block < 4; block++) {
            for (int j = 0; j < 4; j++) {
                c[block][j] = _mm512_fmadd_ps(a[block], b[j], c[block][j]);
            }
        }
    }

    for (int block = 0; block < 4; block++) {
        for (int j = 0; j < 4; j++) {
            _mm512_store_ps(&C[ldc * j + block * 16], c[block][j]);
        }
    }
}

// 根据矩阵大小选择最佳内核处理大块
static void do_block_large(int lda, int M, int N, int K, float* restrict A, float* restrict B, float* restrict C)
{
    if((M % SMALL_BLOCK_M_SIZE == 0) && (N % SMALL_BLOCK_N_SIZE == 0))
    {
        for (int j = 0; j < N; j += SMALL_BLOCK_N_SIZE)
            for (int i = 0; i < M; i += SMALL_BLOCK_M_SIZE)
                do_block_avx_32k8(lda, lda, lda, K, A + i, B + j * lda, C + i + j * lda);
        return;
    }
    
    if (N == 1) {
        if (M == 64) {
            do_block_avx_64k1(lda, lda, lda, K, A, B, C);
            return;
        }
    }

    float* restrict AA = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_SIZE * K, 64);
    float* restrict BB = (float*)_mm_malloc(sizeof(float) * K * SMALL_BLOCK_SIZE, 64);
    float* restrict CC = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE, 64);

    int M_Left = M % SMALL_BLOCK_SIZE;
    int N_Left = N % SMALL_BLOCK_SIZE;

    if (M_Left > 0) {
        for(int j = 0; j < K; j++) {
            __m512 Avec = _mm512_load_ps(&A[M - M_Left + j * lda]);
            _mm512_store_ps(&AA[j * SMALL_BLOCK_SIZE], Avec);
        }
    }

    if (N_Left > 0) {
        for(int j = 0; j < SMALL_BLOCK_SIZE; j++) {
            int i;
            for(i = 0; i < K - 15; i += 16) {
                __m512 Bvec = _mm512_load_ps(&B[i + (N - N_Left + j) * lda]);
                _mm512_store_ps(&BB[i + j * K], Bvec);
            }
            for(; i < K; i++)
                BB[i + j * K] = B[i + (N - N_Left + j) * lda];
        }
    }

    for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) {
        int N_part = min(SMALL_BLOCK_SIZE, N - j);
        for (int i = 0; i < M; i += SMALL_BLOCK_SIZE) {
            int M_part = min(SMALL_BLOCK_SIZE, M - i);

            if (M_part == SMALL_BLOCK_SIZE && N_part == SMALL_BLOCK_SIZE) {
                do_block_avx_16k16(lda, lda, lda, K, A + i, B + j * lda, C + i + j * lda);
            }
            else if(N_part == SMALL_BLOCK_SIZE) {
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
                    
                do_block_avx_16k16(SMALL_BLOCK_SIZE, lda, SMALL_BLOCK_SIZE, K, AA, B + j * lda, CC);
                
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE];
            }
            else if(M_part == SMALL_BLOCK_SIZE) {
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
                    
                do_block_avx_16k16(lda, K, SMALL_BLOCK_SIZE, K, A + i, BB, CC);
                
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE];
            }
            else {
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
                    
                do_block_avx_16k16(SMALL_BLOCK_SIZE, K, SMALL_BLOCK_SIZE, K, AA, BB, CC);
                
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE];
            }  
        }
    }

    _mm_free(AA);
    _mm_free(BB);
    _mm_free(CC);
}

// 矩阵乘法主函数，处理整个矩阵
void square_sgemm(int lda, float* restrict A, float* restrict B, float* restrict C)
{
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
        int N = min(BLOCK_SIZE, lda-j);
        for (int i = 0; i < lda; i += BLOCK_SIZE) {
            int M = min(BLOCK_SIZE, lda-i);
            do_block_large(lda, M, N, lda, A + i, B + j*lda, C + i + j*lda);
        }
    }
}
```

