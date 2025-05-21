// sigmoid : y = 1 / (1 + exp(-x))

__global__ void sigmoid(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// float4
__global__ void sigmoid_float4(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = 1.0f / (1.0f + expf(-tmp_x.x));
        tmp_y.y = 1.0f / (1.0f + expf(-tmp_x.y));
        tmp_y.z = 1.0f / (1.0f + expf(-tmp_x.z));
        tmp_y.w = 1.0f / (1.0f + expf(-tmp_x.w));
        FLOAT4(y[idx]) = tmp_y;
    }
}