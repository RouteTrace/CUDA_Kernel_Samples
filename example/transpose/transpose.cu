#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

void host_transpose(float* input, int M, int N, float* output) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input[j * N + i];
        }
    }
}

// 朴素实现
__global__ void device_transpose_v0(const float* input, float* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

// 合并写入，减少不合并读取的影响
__global__ void device_transpose_v1(const float* input, float* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = input[col * N + row];
    }
}

// 显式调用__ldg，减少不合并读取的影响
__global__ void device_transpose_v2(const float* input, float* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = __ldg(&input[col * N + row]);
    }
}

// 使用共享内存中转，合并读取+写入
template <const int TILE_DIM>
__global__ void device_transpose_v3(const float* input, float* output, int M, int N) {
    __shared__ float S[TILE_DIM][TILE_DIM];
    const int bx = blockDim.x * TILE_DIM;
    const int by = blockDim.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M && x1 < N) {
        S[threadIdx.y][threadIdx.x] = input[y1 * N + x1];  // 合并读取
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N && x2 < M) {
        output[y2 * M + x2] = S[threadIdx.y][threadIdx.x];  // 合并写入
    }
}

int main() {
    size_t M = 12800;
    size_t N = 12800;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    // 1. host
    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr = (float *)malloc(sizeof(float) * N * M);
    randomize_matrix(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr);
    // printf("init_matrix:\n");
    // print_matrix(h_matrix, M, N);
    // printf("host_transpose:\n");
    // print_matrix(h_matrix_tr, N, M);

    // 2. device
    float *d_matrix, *d_matrix_tr;
    cudaMalloc((void **) &d_matrix, sizeof(float) * M * N);
    cudaMalloc((void **) &d_matrix_tr, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    // 2.1 call transpose_v0
    dim3 block_size0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size0(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));
    float total_time0 = TIME_RECORD(repeat_times, ([&]{device_transpose_v0<<<grid_size0, block_size0>>>(d_matrix, d_matrix_tr, M, N);}));
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("[device_transpose_v0] Average time: (%f) seconds\n", total_time0 / repeat_times);
    // print_matrix(h_matrix_tr, N, M);
    
    // 2.2 call transpose_v1
    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    float total_time1 = TIME_RECORD(repeat_times, ([&]{device_transpose_v1<<<grid_size1, block_size1>>>(d_matrix, d_matrix_tr, M, N);}));
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("[device_transpose_v1] Average time: (%f) seconds\n", total_time1 / repeat_times);
    // print_matrix(h_matrix_tr, N, M);

    // 2.3 call transpose_v2
    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    float total_time2 = TIME_RECORD(repeat_times, ([&]{device_transpose_v2<<<grid_size2, block_size2>>>(d_matrix, d_matrix_tr, M, N);}));
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("[device_transpose_v2] Average time: (%f) seconds\n", total_time2 / repeat_times);
    // print_matrix(h_matrix_tr, N, M);

    // 2.4 call transpose_v3
    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE) * CEIL(M, BLOCK_SIZE));
    float total_time3 = TIME_RECORD(repeat_times, ([&]{device_transpose_v3<BLOCK_SIZE><<<grid_size3, block_size3>>>(d_matrix, d_matrix_tr, M, N);}));
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("[device_transpose_v3] Average time: (%f) seconds\n", total_time3 / repeat_times);
    // print_matrix(h_matrix_tr, N, M);

    // free memory
    free(h_matrix);
    free(h_matrix_tr);
    cudaFree(d_matrix);
    cudaFree(d_matrix_tr);
    return 0;
}