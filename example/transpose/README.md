矩阵转置。

优化思路：

矩阵转置不会涉及数据的重用, 直接操作 GMEM 本身没有问题, 但此时应该注意 GMEM 的访存特性, 其中很重要的即 GMEM 的**访存合并**, 即连续线程访问的 GMEM 中的数据地址是连续的, 可以将多个线程的内存访问合并为一个(或多个)内存访问, 从而减少访存次数, 提高带宽利用率.
> 参考：
> [CUDA笔记-内存合并访问](https://zhuanlan.zhihu.com/p/641639133)
> [CUDA内存访问](https://zhuanlan.zhihu.com/p/632244210)


参考: 
1. https://blog.csdn.net/m0_46197553/article/details/125646380
2. https://blog.csdn.net/LostUnravel/article/details/137613493
3. https://blog.csdn.net/feng__shuai/article/details/114630831