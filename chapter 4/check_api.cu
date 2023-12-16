// 检查核函数 cudaFree() cudaMalloc() cudaMemcpy() 

#include "error.cuh"
#include <stdio.h>
#include <math.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double *x, const double *y, double *z,const int N);
void check(const double *z, const int N);

int main()
{
    const int N = 100000000 + 1;
    
    double *host_x = (double*)malloc(sizeof(double)*N);
    double *host_y = (double*)malloc(sizeof(double)*N);
    double *host_z = (double*)malloc(sizeof(double)*N);

    for(int i = 0; i < N; i++)
    {
        host_x[i] = a;
        host_y[i] = b;
    }
    printf("host_x[0] = %lf, host_y[0] = %lf\n",host_x[0],host_y[0]);

    double *device_x, *device_y, *device_z;
    CHECK(cudaMalloc((void **)&device_x, (sizeof(double)*N)));
    CHECK(cudaMalloc((void **)&device_y, (sizeof(double)*N)));
    CHECK(cudaMalloc((void **)&device_z, (sizeof(double)*N)));

    // 下面这行  cudaMemcpyKind 参数错误
    CHECK(cudaMemcpy(device_x,host_x,sizeof(double)*N,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(device_y,host_y,sizeof(double)*N,cudaMemcpyHostToDevice));

    // 总线程数 N  线程块数 128  网格个数 N/128
    const int block_size = 128;
    int grid_size;
    if(N % block_size == 0) grid_size = N/block_size;
    else grid_size = N/block_size+1;

    printf("block_size = %d, grid_size = %d, thread_nums = %d\n",
                            block_size,grid_size,block_size*grid_size);

    add<<<grid_size,block_size>>>(device_x,device_y,device_z, N);
    CHECK(cudaMemcpy(host_z,device_z,sizeof(double)*N,cudaMemcpyDeviceToHost));
    printf("host_z[0] = %lf\n",host_z[0]);
    check(host_z,N);

    free(host_x);
    free(host_y);
    free(host_z);
    
    CHECK(cudaFree(device_x));
    CHECK(cudaFree(device_y));
    CHECK(cudaFree(device_z));

    return 0;
}

__global__ void add(const double *x, const double *y, double *z, const int N)
{
    // i = 128*blockIdx.x + threadIdx.x
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i <  N)    
    {
        z[i] = x[i] + y[i];
    }
    else return;
}
void check(const double *z, const int N)
{
    bool check_flag = false;
    for(int i = 0; i < N; i++)
    {
        if(fabs(z[i]-c) > EPSILON)
        {
            check_flag = true;
            printf("z[%d] = %lf\n",i,z[i]);
            return ;
        }
    }
    printf("%s\n", check_flag ? "Has errors" : "No errors");
}