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