
#include "dcnv2.cuh"
#include <cublas_v2.h>
#include "common.cuh"

template<typename scalar_t>
static __device__ scalar_t dcn_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
	const int width, const int height, scalar_t h, scalar_t w)
{
	int h_low = floor(h);
	int w_low = floor(w);
	int h_high = h_low + 1;
	int w_high = w_low + 1;

	scalar_t lh = h - h_low;
	scalar_t lw = w - w_low;
	scalar_t hh = 1 - lh, hw = 1 - lw;

	scalar_t v1 = 0;
	if (h_low >= 0 && w_low >= 0)
		v1 = bottom_data[h_low * data_width + w_low];
	scalar_t v2 = 0;
	if (h_low >= 0 && w_high <= width - 1)
		v2 = bottom_data[h_low * data_width + w_high];
	scalar_t v3 = 0;
	if (h_high <= height - 1 && w_low >= 0)
		v3 = bottom_data[h_high * data_width + w_low];
	scalar_t v4 = 0;
	if (h_high <= height - 1 && w_high <= width - 1)
		v4 = bottom_data[h_high * data_width + w_high];

	scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
	scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
	return val;
}
 
template<typename scalar_t>
static __device__ scalar_t dcn_im2col_bilinear2(
    const scalar_t* input, const int width_step, 
    const int width, const int height,
    scalar_t y, scalar_t x
){
    int y_low = floor(y);
    int x_low = floor(x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    scalar_t ly = y - y_low;
    scalar_t lx = x - x_low;
    scalar_t hy = 1 - ly;
    scalar_t hx = 1 - lx;

    /*
    p0(x_low, y_low, hx*hy)             p1(x_high, y_low, lx*hy)
            anchor(x, y)
 
    p2(x_low, y_high, hx*ly)            p3(x_high, y_high, lx*ly)
     */

    scalar_t v0 = 0;
    if(x_low >= 0 && y_low >= 0 && x_low < width && y_low < height)
        v0 = input[y_low * width_step + x_low];
    
    scalar_t v1 = 0;
    if(x_high >= 0 && x_high < width && y_low >= 0 && y_low < height)
        v1 = input[y_low * width_step + x_high];

    scalar_t v2 = 0;
    if(x_low >= 0 && x_low < width && y_high < height && y_high >= 0)
        v2 = input[y_high * width_step + x_low];

    scalar_t v3 = 0;
    if(x_high < width && x_high >= 0 && y_high < height && y_high >= 0)
        v3 = input[y_high * width_step + x_high];

    // printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d, %d, %d, %d\n",
    //     v0, v1, v2, v3,
    //     lx, ly, hx, hy, x, y, x_low, y_low, x_high, y_high
    // );
    return  v0 * (hx * hy) + 
            v1 * (lx * hy) + 
            v2 * (hx * ly) + 
            v3 * (lx * ly);
}

// 循环的次数，是batch,  hoxwo,  channels
template<typename scalar_t>
static __global__ void deformable_conv_im2col_kernel(
    const int edge,
    const scalar_t* input_data, const scalar_t* coord_offset,const scalar_t* coord_weight,
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_groups, const int channel_per_deformable_groups,
    scalar_t* column_data
){
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if(position >= edge) return;
    // input_data    b c h w
    // coord_offset  b (deformable_groups * 2 * kh * kw) h w
    // coord_weight  b (deformable_groups * 1 * kh * kw) h w
    // deformable_groups 为 把输入分为多少个group。每个group内使用一组offset, weight
    // channel_per_deformable_groups = channels / deformable_groups

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
    const int deformable_groups_index = input_channel / channel_per_deformable_groups;

    const int input_position = (input_batch * channels + input_channel) * height * width;
    scalar_t* column_data_ptr  = column_data + position * kernel_h * kernel_w;
    const scalar_t* input_data_ptr   = input_data + input_position;

    for(int i = 0; i < kernel_h; ++i){
        for(int j = 0; j < kernel_w; ++j){
            const scalar_t offset_y = coord_offset[(((((input_batch * deformable_groups + deformable_groups_index) * 2 + 0) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x];
            const scalar_t offset_x = coord_offset[(((((input_batch * deformable_groups + deformable_groups_index) * 2 + 1) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x];
            const scalar_t weight   = coord_weight[(((((input_batch * deformable_groups + deformable_groups_index) * 1 + 0) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x];
            const scalar_t input_y_new_position = input_y + i * dilation_h + offset_y;
            const scalar_t input_x_new_position = input_x + j * dilation_w + offset_x;
            scalar_t value = 0;
            if(input_y_new_position > -1 && input_x_new_position > -1 && input_y_new_position < height && input_x_new_position < width){
                // 如果坐标在范围内，才需要进行插值。否则不需要
                value = dcn_im2col_bilinear<scalar_t>(input_data_ptr, width, width, height, input_y_new_position, input_x_new_position) * weight;
            }
            *column_data_ptr++ = value;
        }
    }
}

template<typename scalar_t>
void deformable_conv_im2col(
    const scalar_t* input_data, const scalar_t* coord_offset, const scalar_t* coord_weight,
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_groups,
    scalar_t* column_data, cudaStream_t stream
){
    const int num_kernels = batch * output_area * channels;
    int threads = 512;
    int blocks = ceil(num_kernels / (scalar_t)threads);
    const int channel_per_deformable_groups = channels / deformable_groups;
    deformable_conv_im2col_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels, 
        input_data, coord_offset, coord_weight,
        batch, channels, height, width,
        output_height, output_width, output_area, 
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        deformable_groups, channel_per_deformable_groups,
        column_data
    );

    CHECK_KERNEL_AND_PRINT_ERROR;
}


// 循环的次数，是batch,  hoxwo,  channels
template<typename scalar_t>
static __global__ void deformable_conv_col2im_kernel(
    const int edge,
    const scalar_t* dcolumn,
    const scalar_t* input, const scalar_t* coord_offset,const scalar_t* coord_weight,
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_groups, const int channel_per_deformable_groups,
    scalar_t* dinput, scalar_t* dcoord_offset, scalar_t* dcoord_weight
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
    const int deformable_groups_index = input_channel / channel_per_deformable_groups;

    // dcolumn is batch x channels*kernel_area x output_area
    const int input_position        = (input_batch * channels + input_channel) * height * width;
    scalar_t* dinput_data_ptr       = dinput + input_position;
    const scalar_t* input_data_ptr  = input  + input_position;
    for(int i = 0; i < kernel_h; ++i){
        for(int j = 0; j < kernel_w; ++j){
            int offset_y_position = (((((input_batch * deformable_groups + deformable_groups_index) * 2 + 0) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x;
            int offset_x_position = (((((input_batch * deformable_groups + deformable_groups_index) * 2 + 1) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x;
            int weight_position   = (((((input_batch * deformable_groups + deformable_groups_index) * 1 + 0) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x;
            int column_grad_position = ((((col_batch * channels + col_channel) * kernel_h + i) * kernel_w + j) * output_height + output_y) * output_width + output_x;
            const scalar_t column_grad = dcolumn[column_grad_position];
            const scalar_t offset_y = coord_offset[offset_y_position];
            const scalar_t offset_x = coord_offset[offset_x_position];
            const scalar_t weight   = coord_weight[weight_position];
            const scalar_t input_y_new_position = input_y + i * dilation_h + offset_y;
            const scalar_t input_x_new_position = input_x + j * dilation_w + offset_x;
            scalar_t dvalue = column_grad * weight;

            if(input_y_new_position > -1 && input_x_new_position > -1 && input_y_new_position < height && input_x_new_position < width){
                // 如果坐标在范围内，才需要进行插值。否则不需要
                //value = dcn_im2col_bilinear<scalar_t>(input_data_ptr, width, width, height, input_y_new_position, input_x_new_position);

                int y_low = floor(input_y_new_position);
                int x_low = floor(input_x_new_position);
                int y_high = y_low + 1;
                int x_high = x_low + 1;
                int width_step = width;

                scalar_t ly = input_y_new_position - y_low;
                scalar_t lx = input_x_new_position - x_low;
                scalar_t hy = 1 - ly;
                scalar_t hx = 1 - lx;

                /*
                p0(x_low, y_low, hx*hy)             p1(x_high, y_low, lx*hy)
                        anchor(x, y)
            
                p2(x_low, y_high, hx*ly)            p3(x_high, y_high, lx*ly)
                */

                scalar_t v0 = 0;
                if(x_low >= 0 && y_low >= 0){
                    v0 = input_data_ptr[y_low * width_step + x_low];
                    atomicAdd(dinput_data_ptr + y_low * width_step + x_low, dvalue * hx * hy);
                }

                scalar_t v1 = 0;
                if(x_high < width && y_low >= 0){
                    v1 = input_data_ptr[y_low * width_step + x_high];
                    atomicAdd(dinput_data_ptr + y_low * width_step + x_high, dvalue * lx * hy);
                }

                scalar_t v2 = 0;
                if(x_low >= 0 && y_high < height){
                    v2 = input_data_ptr[y_high * width_step + x_low];
                    atomicAdd(dinput_data_ptr + y_high * width_step + x_low, dvalue * hx * ly);
                }

                scalar_t v3 = 0;
                if(x_high < width && y_high < height){
                    v3 = input_data_ptr[y_high * width_step + x_high];
                    atomicAdd(dinput_data_ptr + y_high * width_step + x_high, dvalue * lx * ly);
                }
                
                atomicAdd(dcoord_offset + offset_y_position, (v0 * hx * (-1) + v1 * lx * (-1) + v2 * hx * (+1) + v3 * lx * (+1)) * dvalue);
                atomicAdd(dcoord_offset + offset_x_position, (v0 * hy * (-1) + v1 * hy * (+1) + v2 * ly * (-1) + v3 * ly * (+1)) * dvalue);

                scalar_t ret_value = 
                        v0 * (hx * hy) + 
                        v1 * (lx * hy) + 
                        v2 * (hx * ly) + 
                        v3 * (lx * ly);
                atomicAdd(dcoord_weight + weight_position, column_grad * ret_value);
            }
        } 
    }
} 

template<typename scalar_t>
void deformable_conv_col2im(
    const scalar_t* dcolumn, const scalar_t* column,
    const scalar_t* input, const scalar_t* coord_offset, const scalar_t* coord_weight,
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_groups,
    scalar_t* dinput, scalar_t* dcoord_offset, scalar_t* dcoord_weight,
    cudaStream_t stream
){
    // channels*kernel_area x output_area 
    const int num_kernels = batch * output_area * channels;
    int threads = 256;
    int blocks = ceil(num_kernels / (float)threads);
    const int channel_per_deformable_groups = channels / deformable_groups;
  
    deformable_conv_col2im_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels, 
        dcolumn,
        input, coord_offset, coord_weight,
        batch, channels, height, width,
        output_height, output_width, output_area, 
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        deformable_groups, channel_per_deformable_groups,
        dinput, dcoord_offset, dcoord_weight
    );

    CHECK_KERNEL_AND_PRINT_ERROR;
}



#define DefineTemplate(name)                        \
static auto name##_float = name<float>;             \
static auto name##_double = name<double>;

DefineTemplate(deformable_conv_im2col);
DefineTemplate(deformable_conv_col2im);