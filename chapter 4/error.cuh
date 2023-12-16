// 预处理指令  确保当前文件在一个编译单元中不被重复包含
#pragma once
#include <stdio.h>

#define CHECK(call)                                                                                 \
do{                                                                                                 \
    const cudaError_t error_code = call;                                                            \
    if(error_code != cudaSuccess)                                                                   \
    {                                                                                               \
        printf("CUDA Error:\n");                                                                    \
        printf("    File:   %s\n",__FILE__);                                                        \
        printf("    Line:   %d\n",__LINE__);                                                        \
        printf("    Error code: %d\n", error_code);                                                 \
        printf("    Error text: %s\n", cudaGetErrorString(error_code));                             \
        exit(1);                                                                                    \
    }                                                                                               \
}while(0);                                                                                          
