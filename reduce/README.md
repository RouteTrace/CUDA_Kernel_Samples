# 归约计算
内容：求给定数组求和

1. device_reduce_v0：仅使用**全局内存**，且 N 必须是 BLOCK_SIZE 的整数倍
2. device_reduce_v1：使用（静态）**共享内存**，不再要求 N 是 BLOCK_SIZE 的整数倍，归约的过程中不会改变全局内存的数据
3. device_reduce_v2：在v1基础上修改，使用（动态）**共享内存**，性能不变
4. device_reduce_v3：在v2基础上修改，通过原子函数，不再需要到CPU上再归约一次

## 结果
N=100000000，BLOCK_SIZE = 128 的测试结果：
```
[reduce_host]: sum = -1209.635986, total_time_h = 388.534760 ms
[reduce_v0]: sum = -22739588.000000, total_time_0 = 31.805029 ms
[reduce_v1]: sum = -1208.930542, total_time_1 = 19.669153 ms
[reduce_v2]: sum = -1208.930542, total_time_2 = 19.637846 ms
[reduce_v3]: sum = -1208.927124, total_time_3 = 15.914701 ms
```

## 参考：
1. cuda编程基础与实践 (樊哲勇)