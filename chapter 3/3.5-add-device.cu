#include <stdio.h>
#include <math.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add_1(const double *x, const double *y, double *z, const int N);
__global__ void add_2(const double *x, const double *y, double *z, const int N);
__global__ void add_3(const double *x, const double *y, double *z, const int N);

__device__ double add_device_1(const double x, const double y);
__device__ void add_device_2(const double x,const double y,double *z);
__device__ void add_device_3(const double x,const double y,double &z);
void check(const double *z, const int N);

int main()
{
    const int N = 100000001;
    
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
    cudaMalloc((void **)&device_x, (sizeof(double)*N));
    cudaMalloc((void **)&device_y, (sizeof(double)*N));
    cudaMalloc((void **)&device_z, (sizeof(double)*N));

    cudaMemcpy(device_x,host_x,sizeof(double)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(device_y,host_y,sizeof(double)*N,cudaMemcpyHostToDevice);

    // 总线程数 N  线程块数 128  网格个数 N/128
    const int block_size = 128;
    int grid_size;
    if(N % block_size == 0) grid_size = N/block_size;
    else grid_size = N/block_size+1;

    printf("block_size = %d, grid_size = %d, thread_nums = %d\n",
                            block_size,grid_size,block_size*grid_size);


    // add_1<<<grid_size,block_size>>>(device_x,device_y,device_z,N);
    // add_2<<<grid_size,block_size>>>(device_x,device_y,device_z,N);
    add_3<<<grid_size,block_size>>>(device_x,device_y,device_z,N);

    cudaMemcpy(host_z,device_z,sizeof(double)*N,cudaMemcpyDeviceToHost);
    printf("host_z[0] = %lf\n",host_z[0]);
    check(host_z,N);

    free(host_x);
    free(host_y);
    free(host_z);
    
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);

    return 0;
}

__global__ void add_1(const double *x, const double *y, double *z, const int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
    {
        z[i] = add_device_1(x[i],y[i]);
    }
}

__device__ double add_device_1(const double x, const double y)
{
    return x+y;
}


__global__ void add_2(const double *x, const double *y, double *z, const int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
    {
        add_device_2(x[i],y[i],&z[i]);
    }
}

__device__ void add_device_2(const double x,const double y,double *z)
{
    *z = x + y;
}

__global__ void add_3(const double *x, const double *y, double *z, const int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
    {
        add_device_3(x[i],y[i],z[i]);
    }
}

__device__ void add_device_3(const double x,const double y,double &z)
{
    z = x + y;
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