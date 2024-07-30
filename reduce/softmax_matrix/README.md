# 归约计算-矩阵Softmax
内容：对MxN的**矩阵的每一行**求softmax

参考[sgemv](../../gemv/sgemv_k32.cu)，用一个warp负责一行(列)的计算。

注意，warp归约计算最后的结果将汇总到第一个线程中，第一个线程要把这个数据搬运到s_mem，以供同一个warp中的其他线程使用。

测试，M = 2048，N = 64：
```
[softmax_row_cpu]: total_time_h = 2.761523 ms
[softmax_row_gpu]: total_time_d = 0.054515 ms
[softmax_col_cpu]: total_time_h = 4.959296 ms
[softmax_col_gpu]: total_time_d = 0.141539 ms
```