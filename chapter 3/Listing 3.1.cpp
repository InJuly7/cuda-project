#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void add(const double *x, const double *y, double *z, const int N);
void check(const double*z, const int N);

int main()
{
    const int N = 100000000;
    const int M = sizeof(double)*N;
    
    // 3*8*10000000 = 2400 * 1000 * 1000 --> 2.4GB
    double *x = (double*)malloc(sizeof(double)*N);
    double *y = (double*)malloc(sizeof(double)*N);
    double *z = (double*)malloc(sizeof(double)*N);

    for(int i = 0; i < N; i++)
    {
        x[i] = a;
        y[i] = b;
    }
    printf("x[0] = %lf, y[0] = %lf\n",x[0],y[0]);
    add(x,y,z,N);
    printf("z[0] = %lf\n",z[0]);
    check(z,N);

    free(x);
    free(y);
    free(z);

    return 0;

}

void add(const double *x, const double *y, double *z, const int N)
{
    for(int i = 0; i < N; i++)
    {
        z[i] = x[i] + y[i];
    }

}

void check(const double *z, const int N)
{
    bool check_flag = false;
    for(int i = 0; i < N; i++)
    {
        if(fabs(z[i]-c) > EPSILON)
        {
            check_flag = true;
        }
    }
    printf("%s\n", check_flag ? "Has errors" : "No errors");
}