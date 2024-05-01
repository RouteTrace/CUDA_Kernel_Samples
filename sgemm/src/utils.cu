#include <stdio.h>
#include "utils.cuh"

void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

void randomize_matrix(float *mat, size_t N) {
    // NOTICE: 使用gettimeofdays替代srand((unsigned)time(NULL));time精度过低，产生相同随机数
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float) (rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, size_t N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %lu elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N) {
    int i;
    printf("[");
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n");
}

bool verify_matrix(float *mat1, float *mat2, size_t N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

float call_kernel(int kernel_num, bool record, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    float total_time = 0;
    float repeat_times = 0;
    if (record) repeat_times = REPEAT_TIMES;
    if (kernel_num == 0) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        total_time = TIME_RECORD(repeat_times, ([&]{cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);}));
        cublasDestroy(handle);
    }
    else if (kernel_num == 1) {
        dim3 blockDim(32, 32);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
        
    } else {
        printf("Error: kernel %d not found.\n", kernel_num);
        exit(EXIT_FAILURE);
    }
    return total_time;
}