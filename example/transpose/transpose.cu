#include <stdio.h>
#include <stdlib.h>
#include <random>

#define Ceil(a, b) ((a) + (b) - 1) / (b)
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
#define TIME_RECORD(N, func)                                                                    \
    [&] {                                                                                       \
        float total_time = 0;                                                                   \
        for (int repeat = 0; repeat <= N; ++repeat) {                                           \
            cudaEvent_t start, stop;                                                            \
            cudaCheck(cudaEventCreate(&start));                                                 \
            cudaCheck(cudaEventCreate(&stop));                                                  \
            cudaCheck(cudaEventRecord(start));                                                  \
            cudaEventQuery(start);                                                              \
            func();                                                                             \
            cudaCheck(cudaEventRecord(stop));                                                   \
            cudaCheck(cudaEventSynchronize(stop));                                              \
            float elapsed_time;                                                                 \
            cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));                        \
            if (repeat > 0) total_time += elapsed_time;                                         \
            cudaCheck(cudaEventDestroy(start));                                                 \
            cudaCheck(cudaEventDestroy(stop));                                                  \
        }                                                                                       \
        if (N == 0) return (float)0.0;                                                          \
        return total_time;                                                                      \
    }()

void randomize_matrix(float *mat, int N) {
    std::random_device rd;  
    std::mt19937 gen(rd()); // 使用随机设备初始化生成器  

    // 创建一个在[0, 2000)之间均匀分布的分布对象  
    std::uniform_int_distribution<> dis(0, 2000); 
    for (int i = 0; i < N; i++) {
        // 生成随机数，限制范围在[-1.0,1.0]
        mat[i] = (dis(gen)-1000)/1000.0;  
    }
}

void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void init_matrix(int* matrix, int M, int N) {
    srand(time(NULL));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i*N+j] = rand() % 1000;
        }
    }
}

void print_matrix(int* a, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%3d ", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void host_transpose(int* input, int M, int N, int* output) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input[j * N + i];
        }
    }
}

__global__ void device_transpose_v0(const int* input, int* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

__global__ void device_transpose_v1(const int* input, int* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = input[col * N + row];
    }
}

int main() {
    size_t M = 12800;
    size_t N = 12800;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    // 1. host
    int *h_matrix = (int *)malloc(sizeof(int) * M * N);
    int *h_matrix_tr = (int *)malloc(sizeof(int) * N * M);
    init_matrix(h_matrix, M, N);
    host_transpose(h_matrix, M, N, h_matrix_tr);

    // printf("init_matrix:\n");
    // print_matrix(h_matrix, M, N);
    // printf("host_transpose:\n");
    // print_matrix(h_matrix_tr, N, M);

    // 2. device
    int *d_matrix, *d_matrix_tr;
    cudaMalloc((void **) &d_matrix, sizeof(int) * M * N);
    cudaMalloc((void **) &d_matrix_tr, sizeof(int) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(int) * M * N, cudaMemcpyHostToDevice);

    // 2.1 call transpose_v0
    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(Ceil(M, BLOCK_SIZE), Ceil(N, BLOCK_SIZE));
    float total_time0 = TIME_RECORD(repeat_times, ([&]{device_transpose_v0<<<grid_size1, block_size1>>>(d_matrix, d_matrix_tr, M, N);}));
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("[device_transpose_v0] Average time: (%f) seconds\n", total_time0 / repeat_times);
    // print_matrix(h_matrix_tr, N, M);
    
    // 2.2 call transpose_v1
    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(Ceil(N, BLOCK_SIZE), Ceil(M, BLOCK_SIZE));
    float total_time1 = TIME_RECORD(repeat_times, ([&]{device_transpose_v1<<<grid_size2, block_size2>>>(d_matrix, d_matrix_tr, M, N);}));
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("[device_transpose_v1] Average time: (%f) seconds\n", total_time1 / repeat_times);
    // print_matrix(h_matrix_tr, N, M);
    
    return 0;
}