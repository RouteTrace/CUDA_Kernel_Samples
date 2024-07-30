#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"


void softmax(float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        // 第row行
        float* input_tmp  = input + row * N;
        float* output_tmp = output + row * N;
        float max_val = *(std::max_element(input_tmp, input_tmp + N));  // 计算输入数组的最大值
        float sum = 0;
        for (int i = 0; i < N; i++) {
            output_tmp[i] = std::exp(input_tmp[i] - max_val);  // 每个数先减去最大值，再求exp，避免溢出
            sum += output_tmp[i];
        }
        for (int i = 0; i < N; i++) {
            output_tmp[i] /= sum;
        }
    }
}

__global__ void softmax_kernel(float* input, float* output, int M, int N) {
    __shared__ float s_max_val;
    __shared__ float s_sum;
    int laneId = threadIdx.x % warpSize;
    // 当前行
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);  // 每个线程负责计算的数据个数

    // 求每一行最大值
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if (laneId == 0) s_max_val = max_val;  // 最大值汇总到第一个线程，第一个线程将它搬运到s_mem

    // 求每一行的和，且要减去最大值
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) s_sum = sum;  // sum值汇总到第一个线程，第一个线程将它搬运到s_mem

    // 计算每一行的softmax
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}


int main() {
    const int M = 2048;
    const int N = 64;
    const int repeat_times = 10;

    float* input      = (float*)malloc(M * N * sizeof(float));  // 输入是M*N的矩阵
    float* output     = (float*)malloc(M * N * sizeof(float));  // 输出是M*N的矩阵
    float* output_ref = (float*)malloc(M * N * sizeof(float));  // 输出是M*N的矩阵(cpu)

    // 初始化输入
    randomize_matrix(input, M*N);

    // cpu
    float total_time_h = TIME_RECORD(repeat_times, ([&]{softmax(input, output_ref, M, N);}));
    printf("[softmax_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    float* input_device  = nullptr;
    float* output_device = nullptr;
    cudaCheck(cudaMalloc(&input_device,  M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&output_device, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(input_device, input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    int grid_size = M;
    int block_size = 32;

    // gpu
    float total_time_d = TIME_RECORD(repeat_times, ([&]{softmax_kernel<<<grid_size, block_size>>>(input_device, output_device, M, N);}));
    printf("[softmax_gpu]: total_time_d = %f ms\n", total_time_d / repeat_times);
    cudaCheck(cudaMemcpy(output, output_device, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_matrix(output, output_ref, M*N);

    return 0;
}