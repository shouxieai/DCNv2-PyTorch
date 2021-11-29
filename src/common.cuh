#ifndef __COMMON_CUH
#define __COMMON_CUH


#include <stdio.h>
 

// 对于核函数执行失败，提示错误
// 这里peekatlast与getlast在于peek不会清除当前状态，结果是错误会由pytorch提示出来咱们仅仅在这里打印
// pytorch打印错误可以看到调用堆栈
// 如果用getlasterror，会导致当前状态清空，pytorch不会认为发生错误
#define CHECK_KERNEL_AND_PRINT_ERROR                            \
    do{                                                         \
        cudaError_t err = cudaPeekAtLastError();                \
        if (err != cudaSuccess){                                \
            printf("[%s:%d][%s]Kernel failed: %s, %d\n", __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(err), err);   \
        }                                                                                                           \
    }while(0);




#endif // __COMMON_CUH