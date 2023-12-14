# include <stdio.h>
 

__global__ void hello_from_gpu()
{
	printf("Hello World from the GPU! \n");
}

int main()
{ 
	// <<<grid_size, block_size>>>
	// hello_from_gpu<<<1,1>>>();
	hello_from_gpu<<<2,4>>>();
	cudaDeviceSynchronize();
	return 0;
}