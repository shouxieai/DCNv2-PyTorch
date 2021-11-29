

#ifndef CONV_CUH
#define CONV_CUH

#include <cuda_runtime.h>

// 对于assign系列函数，咱们指针是二级指针，需要assign上对应的连续空间内存地址。用来交给
// sgemmbatched使用，相当于批量矩阵乘法所使用
template<typename scalar_t>
void assign_forward_batched_pointer(
    scalar_t** input_batched, scalar_t** output_batched, 
    scalar_t** column_batched, scalar_t** weight_batched,  
    scalar_t* input, scalar_t* output,
    scalar_t* column, scalar_t* weight, scalar_t* bias,
    const int input_step, const int output_step, const int column_step,
    const int batch, cudaStream_t stream
);

template<typename scalar_t>
void assign_backward_batched_pointer(
    scalar_t** input_batched, scalar_t** grad_batched, scalar_t** column_batched,
    scalar_t** weight_batched, 
    scalar_t** dinput_batched, scalar_t** dcolumn_batched,
    scalar_t** dweight_batched,
    scalar_t* input, scalar_t* grad, scalar_t* column,
    scalar_t* weight,
    scalar_t* dinput, scalar_t* dcolumn,
    scalar_t* dweight,
    const int input_step, const int grad_step, const int column_step,
    const int weight_step,
    const int batch, cudaStream_t stream
);

template<typename scalar_t>
void conv_im2col(
    const scalar_t* input_data, 
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    scalar_t* column_data, cudaStream_t stream
);

template<typename scalar_t>
void conv_col2im(
    const scalar_t* dcolumn, 
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    scalar_t* dinput, cudaStream_t stream
);

template<typename scalar_t>
void set_bias(
    scalar_t* output, scalar_t* bias, int batch, int output_channels, int output_area, cudaStream_t stream
);


#endif //CONV_CUH