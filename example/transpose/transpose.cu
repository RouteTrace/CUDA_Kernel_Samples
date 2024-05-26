#include <stdio.h>
#include <stdlib.h>

#define Ceil(a, b) ((a) + (b) - 1) / (b)

void init_matrix(int* matrix, size_t M, size_t N) {
    srand(time(NULL));
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i*N+j] = rand() % 1000;
		}
	}
}

void print_matrix(int* a, size_t M, size_t N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			printf("%3d ", a[i * N + j]);
		}
		printf("\n");
	}
    printf("\n");
}

void host_transpose(int* input, size_t M, size_t N, int* output) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input[j * N + i];
        }
    }
}

__global__ void device_transpose_v0(int* input, int* output, size_t M, size_t N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
	const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

template <const size_t BLOCK_SIZE>
__global__ void device_transpose_v1(int* input, int* output, size_t M, size_t N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
	const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

int main() {
    size_t M = 8;
    size_t N = 4;
    constexpr size_t BLOCK_SIZE = 8;

    // 1. host
    int *h_matrix = (int *)malloc(sizeof(int) * M * N);
    int *h_matrix_tr = (int *)malloc(sizeof(int) * N * M);
    init_matrix(h_matrix, M, N);
    host_transpose(h_matrix, M, N, h_matrix_tr);

    printf("init_matrix:\n");
    print_matrix(h_matrix, M, N);
    printf("host_transpose:\n");
    print_matrix(h_matrix_tr, N, M);

    // 2. device
    int *d_matrix, *d_matrix_tr;
    cudaMalloc((void **) &d_matrix, sizeof(int) * M * N);
    cudaMalloc((void **) &d_matrix_tr, sizeof(int) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Ceil(M, BLOCK_SIZE), Ceil(N, BLOCK_SIZE));

    // 2.1 call transpose_v0
    device_transpose_v0<<<grid, block>>>(d_matrix, d_matrix_tr, M, N);
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("device_transpose_v0:\n");
    print_matrix(h_matrix_tr, N, M);

    // 2.2 call transpose_v2
    device_transpose_v1<BLOCK_SIZE><<<grid, block>>>(d_matrix, d_matrix_tr, M, N);
    cudaMemcpy(h_matrix_tr, d_matrix_tr, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return 0;
}