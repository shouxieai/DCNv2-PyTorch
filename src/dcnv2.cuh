

#ifndef DCNV2_CUH
#define DCNV2_CUH

#include <cuda_runtime.h>

template<typename scalar_t>
void deformable_conv_im2col(
    const scalar_t* input_data, const scalar_t* coord_offset, const scalar_t* coord_weight,
    const int batch, const int channels, const int height, const int width, 
    const int output_height, const int output_width, const int output_area,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group,
    scalar_t* column_data, cudaStream_t stream
);

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
    const int deformable_group,
    scalar_t* dinput, scalar_t* dcoord_offset, scalar_t* dcoord_weight,
    cudaStream_t stream
);

#endif //DCNV2_CUH