# include <stdio.h>
 
// 在调用核函数时制定了一个2*4的二维线程块,程序的输出是
__global__ void hello_from_gpu()
{
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	printf("tid = %d\n",tid);
	printf("Hello World from block-%d and thread-(%d, %d)\n",bx,tx,ty);
}

int main()
{ 
	const dim3 block_size(2,4);
	hello_from_gpu<<<1,block_size>>>();
	cudaDeviceSynchronize();
	return 0;
}