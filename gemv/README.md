# Sgemv-矩阵乘以向量
编译运行：
```bash
nvcc sgemv_k32.cu -o sgemv_k32 -lcublas && sgemv_k32
```

# 参考
1. [深入浅出GPU优化系列：gemv优化](https://zhuanlan.zhihu.com/p/494144694)
2. [GitHub - How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemv/Sgemv_v0.cu)