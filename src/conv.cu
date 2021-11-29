
#include "conv.cuh"
#include <cublas_v2.h>
#include "common.cuh"

template<typename scalar_t>
static __global__ void set_bias_kernel(
    scalar_t* output, scalar_t* bias, int output_channels, int output_area, int edge
){  
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if(position >= edge) return;

    int bias_channel = (position / output_area) % output_channels;
    output[position] = bias[bias_channel];
}

template<typename scalar_t>
void set_bias(
    scalar_t* output, scalar_t* bias, int batch, int output_channels, int output_area, cudaStream_t stream
){
    int jobs = batch * output_channels * output_area;
    int threads = 512;
    int blocks = ceil(jobs / (scalar_t)threads);
    set_bias_kernel<<<blocks, threads, 0, stream>>>(output, bias, output_channels, output_area, jobs);

    CHECK_KERNEL_AND_PRINT_ERROR;
}

template<typename scalar_t>
static __global__ void assign_forward_batched_pointer_kernel(
    scalar_t** input_batched, scalar_t** output_batched, 
    scalar_t** column_batched, scalar_t** weight_batched, 
    scalar_t* input, scalar_t* output,
    scalar_t* column, scalar_t* weight, scalar_t* bias,
    const int input_step, const int output_step, const int column_step,
    const int batch
){
    const int ibatch = blockIdx.x * blockDim.x + threadIdx.x;
    if(ibatch >= batch) return;

    input_batched[ibatch]  = input  + ibatch * input_step;
    output_batched[ibatch] = output + ibatch * output_step;
    column_batched[ibatch] = column + ibatch * column_step;
    weight_batched[ibatch] = weight;
}

template<typename scalar_t>
void assign_forward_batched_pointer(
    scalar_t** input_batched, scalar_t** output_batched, 
    scalar_t** column_batched, scalar_t** weight_batched,
    scalar_t* input, scalar_t* output,
    scalar_t* column, scalar_t* weight, scalar_t* bias,
    const int input_step, const int output_step, const int column_step,
    const int batch, cudaStream_t stream
){
    int threads = 64;
    int blocks = ceil(batch / (scalar_t)threads);
    assign_forward_batched_pointer_kernel<<<blocks, threads, 0, stream>>>(
        input_batched, output_batched,
        column_batched, weight_batched,
        input, output, 
        column, weight, bias, 
        input_step, output_step, column_step, batch
    );

    CHECK_KERNEL_AND_PRINT_ERROR;
}

template<typename scalar_t>
static __global__ void assign_backward_batched_pointer_kernel(
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
    const int batch
){
    const int ibatch = blockIdx.x * blockDim.x + threadIdx.x;
    if(ibatch >= batch) return;
    
    input_batched[ibatch]  = input  + ibatch * input_step;
    column_batched[ibatch] = column + ibatch * column_step;
    grad_batched[ibatch]   = grad   + ibatch * grad_step;
    weight_batched[ibatch] = weight;

    dinput_batched[ibatch]  = dinput  + ibatch * input_step;
    dcolumn_batched[ibatch] = dcolumn + ibatch * column_step;
    dweight_batched[ibatch] = dweight + ibatch * weight_step;
}

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
){
    int threads = 64;
    int blocks = ceil(batch / (scalar_t)threads);
    assign_backward_batched_pointer_kernel<<<blocks, threads, 0, stream>>>(
        input_batched, grad_batched, column_batched,
        weight_batched,
        dinput_batched,     dcolumn_batched,
        dweight_batched,
        input,      grad,   column,
        weight, 
        dinput,     dcolumn, 
        dweight, 
        input_step, grad_step, column_step, 
        weight_step,
        batch
    );

    CHECK_KERNEL_AND_PRINT_ERROR;
}

// 循环的次数，是batch,  hoxwo,  channels
template<typename scalar_t>
static __global__ void conv_im2col_kernel(
    const int edge,
    const scalar_t* input_data, 
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    scalar_t* column_data
){ 
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if(position >= edge) return;

    // 每次进行一个核的赋值
    const int col_channel   = position % channels;
    const int col_y         = (position / channels) % output_area;
    const int col_batch     = (position / channels / output_area) % batch;

    // column坐标映射到输入的起点坐标
    const int output_x      = col_y % output_width;
    const int output_y      = col_y / output_width;
    const int input_x       = output_x * stride_w - pad_w;
    const int input_y       = output_y * stride_h - pad_h;
    const int input_channel = col_channel;
    const int input_batch   = col_batch;

    const int input_position = ((input_batch * channels + input_channel) * height + input_y) * width + input_x;
    scalar_t* column_data_ptr  = column_data + position * kernel_h * kernel_w;
    const scalar_t* input_data_ptr   = input_data + input_position;
    for(int i = 0; i < kernel_h; ++i){
        int iy = input_y + i * dilation_h;
        if(iy < 0 || iy >= height)
            continue;
        
        for(int j = 0; j < kernel_w; ++j){
            int ix = input_x + j * dilation_w;
            if(ix < 0 || ix >= width)
                continue;

            column_data_ptr[i * kernel_w + j] = input_data_ptr[i * dilation_h * width + j * dilation_w];
        }
    }
}

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
){
    const int num_kernels = batch * output_area * channels;
    int threads = 512;
    int blocks = ceil(num_kernels / (scalar_t)threads);

    conv_im2col_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels, 
        input_data,
        batch, channels, height, width,
        output_height, output_width, output_area, 
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        column_data
    );

    CHECK_KERNEL_AND_PRINT_ERROR;
}


// 循环的次数，是batch,  hoxwo,  channels
template<typename scalar_t>
static __global__ void conv_col2im_kernel(
    const int edge,
    const scalar_t* dcolumn, 
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    scalar_t* dinput
){
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if(position >= edge) return;

    // 每次进行一个核的赋值
    const int col_channel   = position % channels;
    const int col_y         = (position / channels) % output_area;
    const int col_batch     = (position / channels / output_area) % batch;

    // column坐标映射到输入的起点坐标
    const int output_x      = col_y % output_width;
    const int output_y      = col_y / output_width;
    const int input_x       = output_x * stride_w - pad_w;
    const int input_y       = output_y * stride_h - pad_h;
    const int input_channel = col_channel;
    const int input_batch   = col_batch;

    // dcolumn is batch x channels*kernel_area x output_area
    const int input_position        = ((input_batch * channels + input_channel) * height + input_y) * width + input_x;
    scalar_t* input_data_ptr        = dinput + input_position;
    for(int i = 0; i < kernel_h; ++i){
        int iy = input_y + i * dilation_h;
        if(iy < 0 || iy >= height)
            continue;

        for(int j = 0; j < kernel_w; ++j){
            int ix = input_x + j * dilation_w;
            if(ix < 0 || ix >= width)
                continue;

            const scalar_t* dcolumn_ptr = dcolumn + ((((col_batch * channels + col_channel) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x;
            atomicAdd(input_data_ptr + i * dilation_h * width + j * dilation_w, *dcolumn_ptr);
        }
    }
}

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
){
    // channels*kernel_area x output_area
    const int num_kernels = batch * output_area * channels;
    int threads = 512;
    int blocks = ceil(num_kernels / (float)threads);
    conv_col2im_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels, 
        dcolumn,
        batch, channels, height, width,
        output_height, output_width, output_area, 
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        dinput
    );

    CHECK_KERNEL_AND_PRINT_ERROR;
}



#define DefineTemplate(name)                        \
static auto name##_float = name<float>;             \
static auto name##_double = name<double>;

DefineTemplate(assign_forward_batched_pointer);
DefineTemplate(assign_backward_batched_pointer);
DefineTemplate(conv_im2col);
DefineTemplate(conv_col2im);
DefineTemplate(set_bias);