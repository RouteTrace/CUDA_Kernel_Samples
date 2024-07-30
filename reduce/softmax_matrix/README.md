# 归约计算-矩阵Softmax
内容：对MxN的**矩阵的每一行**求softmax

参考[sgemv](../../gemv/sgemv_k32.cu)，用一个warp负责一行的计算。

注意，warp归约计算最后的结果将汇总到第一个线程中，第一个线程要把这个数据搬运到s_mem，以供同一个warp中的其他线程使用。