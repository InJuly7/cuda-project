#include <stdio.h>
#include <math.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main()
{
    const int N = 100000000;
    
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
    const int grid_size = N/block_size;
    printf("block_size = %d, grid_size = %d, thread_nums = %d\n",
                            block_size,grid_size,block_size*grid_size);

    add<<<grid_size,block_size>>>(device_x,device_y,device_z);
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

__global__ void add(const double *x, const double *y, double *z)
{
    // i = 128*blockIdx.x + threadIdx.x
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    z[i] = x[i] + y[i];
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