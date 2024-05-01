# CUDA SGEMM 优化
## 开发环境
设备：NVIDIA GeForce GTX 1050

## 开发流程
1. 在src下编写kernel.cu
2. 在include编写对应头文件，并在include/kernel.cuh中包含该头文件
3. 在src/utils.cu的call_kernel函数中调用编写的kernel
4. 编译：
```bash
mkdir build && cd build
cmake ..
make
```
5. 运行：
```bash
# run cuBLAS(0) or custom kernel(>0)
./main 0  # cuBLAS
./main 1  # kernel1
...
```
6. 测试并画图：
```bash
pip install matplotlib
bash tools/test.sh  # 日志保存在./test, 图片保存在./images
```

## Kernel1：Native 实现
<div align=center>
<img src="./images/kernel_culas_vs_1.png" width = "700"/>
</div>

### 代码
```cpp
__global__ __launch_bounds__(1024) 
void sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int id_x = blockIdx.x * blockDim.x + threadIdx.x; // id x
    int id_y = blockIdx.y * blockDim.y + threadIdx.y; // id y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[id_y * K + i] * B[i * N + id_x]; // 两次全局内存访问和一次FMA（累加乘）
    }
    C[id_y * N + id_x] = alpha * tmp + beta * C[id_y * N + id_x];
}
```

### 计算步骤 (图解)
将每个逻辑线程与矩阵C的每一个元素相对应，每个线程负责C中一个元素的计算：
<div align=center>
<img src="./images/image.png" width = "500"/><img src="./images/image-1.png" width = "500"/>
</div>

### 分析
未经过优化的矩阵乘法性能不足cuBLAS的1/10，具体分析如下：

1. **访存比低**：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2；
2. **访存延迟高**：访问**全局内存**，**延迟高**，需要几百个时钟周期 (cycle)
3. **较低的访存比无法有效隐藏访存延迟**
4. 访存量：矩阵C的每个元素计算需要访问2K个单精度浮点数，完成全部计算需要 $2*K*M*N$
5. 相同位置元素被重复读取（C中同一行元素计算共享A中同一行元素，C中同一列元素计算共享B中同一列元素）

> 动态全局内存是在运行时动态分配的内存，使用 `cudaMalloc()` 和 `cudaFree()` 函数来分配和释放。

# 参考
1. https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE