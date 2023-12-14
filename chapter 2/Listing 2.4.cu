# include <stdio.h>
 

__global__ void hello_from_gpu()
{
	const int block_idx = blockIdx.x;
	const int thread_idx = threadIdx.x;
	const int block_size = blockDim.x;
	const int grid_size = gridDim.x;
	if (block_idx == 0 && thread_idx == 0)
	{
		printf("block_size : %d grid_size : %d\n",block_size,grid_size);
	}
	printf("Hello World from block %d and thread %d \n",block_idx,thread_idx);
}

int main()
{ 
	// <<<grid_size, block_size>>>
	// hello_from_gpu<<<1,1>>>();
	hello_from_gpu<<<2,4>>>();
	cudaDeviceSynchronize();
	return 0;
}