# 规约计算
内容：求给定数组求和

1. device_reduce_v0：仅使用全局内存，且 N 必须是 BLOCK_SIZE 的整数倍
2. device_reduce_v1：使用（静态）共享内存，不再要求 N 是 BLOCK_SIZE 的整数倍，规约的过程中不会改变全局内存的数据

## 参考：
1. cuda编程基础与实践 (樊哲勇)