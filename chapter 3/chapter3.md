# 编译3.3-add1.cu时 出现的问题
# Q1: nvcc 3.3-add1.cu 

使用默认计算能力 cuda 8.0 的计算能力为2.0 最大的网格个数为65535 不满足 代码要求 781250,会导致程序无法正确执行
使用的cuda 12.2 的计算能力为3.0 未出现该问题


# Q2: nvcc -gencode arch=compute_60,code=sm_60 3.3-add1.cu 
虚拟架构为60 真实架构为60 输出结果不正确 ,初步分析 没有将GPU中的数据拷贝回来
```sh
jetson@ubuntu:~/cuda-project/chapter 3$ ./a.out 
host_x[0] = 1.230000, host_y[0] = 2.340000
block_size = 128, grid_size = 781250, thread_nums = 100000000
host_z[0] = 0.000000
z[0] = 0.000000
```
将指令改为 nvcc -gencode arch=compute_60,code=sm_62 3.3-add1.cu 即可
> https://zhuanlan.zhihu.com/p/631850036
# Q3: nvcc -arch=sm_60 3.3-add1.cu
可以正常运行 
选项 -code=sm_ZW 指定了GPU的真实架构为Z.W. 对应的可执行文件只能在主版本号为 Z,次版本号大于或等于W的 GPU 中运行