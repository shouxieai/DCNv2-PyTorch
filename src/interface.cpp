#include "pre_compile.hpp"
#include "conv.cuh"
#include "dcnv2.cuh"

static THCState *state = at::globalContext().lazyInitCUDA();

void SgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
    float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
    float beta, float *c[], int64_t ldc, int64_t batchCount){
    THCudaBlas_SgemmBatched(state, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batchCount);
}

void SgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
    double alpha, const double *a[], int64_t lda, const double *b[], int64_t ldb,
    double beta, double *c[], int64_t ldc, int64_t batchCount){
    THCudaBlas_DgemmBatched(state, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batchCount);
}

static at::Tensor conv_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool has_bias
){
    AT_ASSERTM(input.type().is_cuda(),          "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(),         "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(),           "bias must be a CUDA tensor");

    //auto device_index = input.get_device();
    //cudaSetDevice(device_index);

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    // weight is out, in, h, w
    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int output_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int output_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_area = output_width * output_height;
    const int kernel_area = kernel_w * kernel_h;

    // columns格式是 hoxwo,  channelsxkernelxkernel
    auto column = at::zeros({batch, output_area, channels * kernel_area}, input.options());
    auto output = at::empty({batch, channels_out, output_height, output_width}, input.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_forward", ([&]{
        const int pointer_size = batch * sizeof(scalar_t*);
        auto input_batched  = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto output_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto column_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto weight_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));

        const int input_step = channels * height * width;
        const int output_step = channels_out * output_area;
        const int column_step = channels * kernel_area * output_area;

        if(has_bias){
            // 如果有bias，则设置到output中，最后output会以beta=1的方式相加
            set_bias(
                output.data<scalar_t>(), bias.data<scalar_t>(), batch, 
                channels_out, output_area, stream
            );
        }

        assign_forward_batched_pointer(
            input_batched, output_batched,
            column_batched, weight_batched,
            input.data<scalar_t>(), output.data<scalar_t>(), 
            column.data<scalar_t>(), weight.data<scalar_t>(), bias.data<scalar_t>(), 
            input_step, output_step, column_step, batch, stream
        );

        conv_im2col(
            input.data<scalar_t>(),
            batch, channels, height, width,
            output_height, output_width, output_area, 
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            column.data<scalar_t>(),
            stream
        );

        // op(A) is nxk   =   output_area,              channels * kernel_area
        // op(B) is kxm   =   channels * kernel_area,             channels_out
        // op(C) is nxm   =   output_area,                        channels_out
        // lda            =   channels * kernel_area
        // ldb            =   channels_out
        // ldc            =   output_area
        const int n = output_area;
        const int k = channels * kernel_area;
        const int m = channels_out;
        const scalar_t alpha = 1.0;
        const scalar_t beta = has_bias ? 1.0f : 0.0f;
        SgemmBatched(state, 't', 'n', n, m, k, alpha,
            (const scalar_t **)column_batched, k,
            (const scalar_t **)weight_batched, k,
            beta,
            output_batched, n,
            batch
        );

        THCudaFree(state, input_batched);
        THCudaFree(state, output_batched);
        THCudaFree(state, column_batched);
        THCudaFree(state, weight_batched);
    }));
    return output;
}

static std::vector<at::Tensor> conv_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool has_bias
){
    AT_ASSERTM(input.type().is_cuda(),          "input must be a CUDA tensor");
    AT_ASSERTM(grad.type().is_cuda(),           "grad must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(),         "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(),           "bias must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    // da = gc @ b.T
    // db = a.T @ gc
    // weight is out, in, h, w
    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int output_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int output_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_area = output_width * output_height;
    const int kernel_area = kernel_w * kernel_h;

    // columns格式是 hoxwo,  channelsxkernelxkernel
    auto column     = at::zeros({batch, output_area, channels * kernel_area}, input.options());
    auto dcolumn    = at::empty({batch, output_area, channels * kernel_area}, input.options());
    auto dweight    = at::empty({batch, channels_out, channels, kernel_h, kernel_w}, input.options());
    auto dinput     = at::zeros_like(input);
    auto stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_backward", ([&]{
        const int pointer_size = batch * sizeof(scalar_t*);
        auto input_batched  = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto column_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto grad_batched   = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto weight_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));

        auto dinput_batched  = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto dcolumn_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto dweight_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));

        const int input_step = channels * height * width;
        const int grad_step = channels_out * output_area;
        const int column_step = channels * kernel_area * output_area;
        const int weight_step = weight.numel();

        assign_backward_batched_pointer(
            input_batched, grad_batched, column_batched,
            weight_batched, 
            dinput_batched, dcolumn_batched,
            dweight_batched,
            input.data<scalar_t>(), grad.data<scalar_t>(), column.data<scalar_t>(),
            weight.data<scalar_t>(), 
            dinput.data<scalar_t>(), dcolumn.data<scalar_t>(), 
            dweight.data<scalar_t>(), 
            input_step, grad_step, column_step, 
            weight_step,
            batch, stream
        );

        // grad.shape = batch, channels_out, output_height, output_width
        // output = column @ weight
        // 先求output对column的导数
        // c 对 a 求导 = gc @ b.T
        // c 对 b 求导 = a.T @ gc

        // op(A) is nxk   =   output_area,                 channels_out
        // op(B) is kxm   =   channels_out,                channels * kernel_area
        // op(C) is nxm   =   output_area,                 channels * kernel_area
        // lda            =   output_area
        // ldb            =   channels * kernel_area
        // ldc            =   output_area
        // dcolumn = grad @ weight.T
        // grad.shape = batch, channels_out, output_height, output_width = [batch], [channels_out], [output_area]
        // weight.shape = channels_out, channels_in, kernel_h, kernel_w = [channels_out], [channels * kernel_area]
        // dcolumn.shape = [batch], [output_area], [channels * kernel_area]
        // output = column[output_area x channels*kernel_area] @ weight[channels*kernel_area x channels_out]
        //        = [output_area x channels_out]
        // 计算dcolumn = grad[output_area x channels_out] @ weight[channels_out x channels*kernel_area]
        // dcolumn = [output_area x channels*kernel_area]
        //         = channels*kernel_area x output_area
        const scalar_t alpha = 1.0;
        const scalar_t beta = 0.0;
        {
            const int n = output_area;
            const int k = channels_out;
            const int m = channels * kernel_area;
            SgemmBatched(state, 'n', 't', n, m, k, alpha,
                (const scalar_t **)grad_batched, n,
                (const scalar_t **)weight_batched, m,
                beta,
                dcolumn_batched, n,
                batch
            );
        };

        conv_col2im(
            dcolumn.data<scalar_t>(),
            batch, channels, height, width,
            output_height, output_width, output_area, 
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            dinput.data<scalar_t>(),
            stream
        );

        // 计算weight的导数，dweight
        // dweight = column.T @ grad
        conv_im2col(
            input.data<scalar_t>(),
            batch, channels, height, width,
            output_height, output_width, output_area, 
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            column.data<scalar_t>(),
            stream
        );

        // output = column[output_area x channels*kernel_area] @ weight[channels*kernel_area x channels_out]
        //        = [output_area x channels_out]
        // dweight = column.T @ grad
        //         = column[channels*kernel_area x output_area] @ [output_area x channels_out]
        //         = channels*kernel_area x channels_out
        {
            const int n = channels*kernel_area;
            const int k = output_area;
            const int m = channels_out;
            SgemmBatched(state, 'n', 'n', n, m, k, alpha,
                (const scalar_t **)column_batched, n,
                (const scalar_t **)grad_batched, k,
                beta,
                dweight_batched, n,
                batch
            );
        };

        THCudaFree(state, input_batched);
        THCudaFree(state, grad_batched);
        THCudaFree(state, weight_batched);
        THCudaFree(state, dinput_batched);
        THCudaFree(state, dcolumn_batched);
        THCudaFree(state, dweight_batched);
    }));
    
    if(has_bias){
        return {dinput, dweight.sum(0), grad.sum({0, 2, 3})};    
    }
    return {dinput, dweight.sum(0), at::empty(0, input.options())};
}








static at::Tensor deformable_conv_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& coord_offset, const at::Tensor& coord_weight,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int deformable_groups,
    bool has_bias
){
    AT_ASSERTM(input.type().is_cuda(),          "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(),         "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(),           "bias must be a CUDA tensor");
    AT_ASSERTM(coord_offset.type().is_cuda(),   "coord_offset must be a CUDA tensor");
    AT_ASSERTM(coord_weight.type().is_cuda(),   "coord_weight must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    // weight is out, in, h, w
    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int output_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int output_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_area = output_width * output_height;
    const int kernel_area = kernel_w * kernel_h;

    // columns格式是 hoxwo,  channelsxkernelxkernel
    auto column = at::zeros({batch, output_area, channels * kernel_area}, input.options());
    auto output = at::zeros({batch, channels_out, output_height, output_width}, input.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "deformable_conv_forward", ([&]{
        const int pointer_size = batch * sizeof(scalar_t*);
        auto input_batched  = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto output_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto column_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto weight_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));

        const int input_step = channels * height * width;
        const int output_step = channels_out * output_area;
        const int column_step = channels * kernel_area * output_area;

        if(has_bias){
            set_bias(
                output.data<scalar_t>(), bias.data<scalar_t>(), batch, 
                channels_out, output_area, stream
            );
        }

        assign_forward_batched_pointer(
            input_batched, output_batched,
            column_batched, weight_batched,
            input.data<scalar_t>(), output.data<scalar_t>(), 
            column.data<scalar_t>(), weight.data<scalar_t>(), bias.data<scalar_t>(), 
            input_step, output_step, column_step, batch, stream
        );

        deformable_conv_im2col(
            input.data<scalar_t>(), coord_offset.data<scalar_t>(), coord_weight.data<scalar_t>(),
            batch, channels, height, width,
            output_height, output_width, output_area, 
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            deformable_groups,
            column.data<scalar_t>(),
            stream
        );

        // op(A) is nxk   =   output_area,              channels * kernel_area
        // op(B) is kxm   =   channels * kernel_area,             channels_out
        // op(C) is nxm   =   output_area,                        channels_out
        // lda            =   channels * kernel_area
        // ldb            =   channels_out
        // ldc            =   output_area
        const int n = output_area;
        const int k = channels * kernel_area;
        const int m = channels_out;
        const scalar_t alpha = 1.0;
        const scalar_t beta = has_bias ? 1.0f : 0.0f;
        SgemmBatched(state, 't', 'n', n, m, k, alpha,
            (const scalar_t **)column_batched, k,
            (const scalar_t **)weight_batched, k,
            beta,
            output_batched, n,
            batch
        );

        THCudaFree(state, input_batched);
        THCudaFree(state, output_batched);
        THCudaFree(state, column_batched);
        THCudaFree(state, weight_batched);
    }));
    return output;
}

static std::vector<at::Tensor> deformable_conv_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& coord_offset, const at::Tensor& coord_weight,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int deformable_groups,
    bool has_bias
){
    AT_ASSERTM(input.type().is_cuda(),          "input must be a CUDA tensor");
    AT_ASSERTM(grad.type().is_cuda(),           "grad must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(),         "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(),           "bias must be a CUDA tensor");
    AT_ASSERTM(coord_offset.type().is_cuda(),   "coord_offset must be a CUDA tensor");
    AT_ASSERTM(coord_weight.type().is_cuda(),   "coord_weight must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    // da = gc @ b.T
    // db = a.T @ gc
    // weight is out, in, h, w
    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int output_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int output_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_area = output_width * output_height;
    const int kernel_area = kernel_w * kernel_h;

    // columns格式是 hoxwo,  channelsxkernelxkernel
    auto column     = at::zeros({batch, output_area, channels * kernel_area}, input.options());
    auto dcolumn    = at::zeros({batch, output_area, channels * kernel_area}, input.options());
    auto dweight    = at::zeros({batch, channels_out, channels, kernel_h, kernel_w}, input.options());
    auto dinput     = at::zeros_like(input);
    auto dcoord_offset = at::zeros_like(coord_offset);
    auto dcoord_weight = at::zeros_like(coord_weight);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "deformable_conv_backward", ([&]{
        const int pointer_size = batch * sizeof(scalar_t*);
        auto input_batched  = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto column_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto grad_batched   = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto weight_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));

        auto dinput_batched  = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto dcolumn_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));
        auto dweight_batched = static_cast<scalar_t**>(THCudaMalloc(state, pointer_size));

        const int input_step = channels * height * width;
        const int grad_step = channels_out * output_area;
        const int column_step = channels * kernel_area * output_area;
        const int weight_step = weight.numel();

        assign_backward_batched_pointer(
            input_batched, grad_batched, column_batched,
            weight_batched, 
            dinput_batched, dcolumn_batched,
            dweight_batched,
            input.data<scalar_t>(), grad.data<scalar_t>(), column.data<scalar_t>(),
            weight.data<scalar_t>(), 
            dinput.data<scalar_t>(), dcolumn.data<scalar_t>(), 
            dweight.data<scalar_t>(), 
            input_step, grad_step, column_step, 
            weight_step,
            batch, stream
        );

        // grad.shape = batch, channels_out, output_height, output_width
        // output = column @ weight
        // 先求output对column的导数
        // c 对 a 求导 = gc @ b.T
        // c 对 b 求导 = a.T @ gc

        // op(A) is nxk   =   output_area,                 channels_out
        // op(B) is kxm   =   channels_out,                channels * kernel_area
        // op(C) is nxm   =   output_area,                 channels * kernel_area
        // lda            =   output_area
        // ldb            =   channels * kernel_area
        // ldc            =   output_area
        // dcolumn = grad @ weight.T
        // grad.shape = batch, channels_out, output_height, output_width = [batch], [channels_out], [output_area]
        // weight.shape = channels_out, channels_in, kernel_h, kernel_w = [channels_out], [channels * kernel_area]
        // dcolumn.shape = [batch], [output_area], [channels * kernel_area]
        // output = column[output_area x channels*kernel_area] @ weight[channels*kernel_area x channels_out]
        //        = [output_area x channels_out]
        // 计算dcolumn = grad[output_area x channels_out] @ weight[channels_out x channels*kernel_area]
        // dcolumn = [output_area x channels*kernel_area]
        //         = channels*kernel_area x output_area
        const scalar_t alpha = 1.0;
        const scalar_t beta = 0.0;
        {
            const int n = output_area;
            const int k = channels_out;
            const int m = channels * kernel_area;
            SgemmBatched(state, 'n', 't', n, m, k, alpha,
                (const scalar_t **)grad_batched, n,
                (const scalar_t **)weight_batched, m,
                beta,
                dcolumn_batched, n,
                batch
            );
        };

        deformable_conv_col2im(
            dcolumn.data<scalar_t>(), column.data<scalar_t>(),
            input.data<scalar_t>(), coord_offset.data<scalar_t>(), coord_weight.data<scalar_t>(),
            batch, channels, height, width,
            output_height, output_width, output_area, 
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            deformable_groups,
            dinput.data<scalar_t>(), dcoord_offset.data<scalar_t>(), dcoord_weight.data<scalar_t>(),
            stream
        );

        // 计算weight的导数，dweight
        // dweight = column.T @ grad
        deformable_conv_im2col(
            input.data<scalar_t>(), coord_offset.data<scalar_t>(), coord_weight.data<scalar_t>(),
            batch, channels, height, width,
            output_height, output_width, output_area, 
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            deformable_groups,
            column.data<scalar_t>(),
            stream
        );

        // output = column[output_area x channels*kernel_area] @ weight[channels*kernel_area x channels_out]
        //        = [output_area x channels_out]
        // dweight = column.T @ grad
        //         = column[channels*kernel_area x output_area] @ [output_area x channels_out]
        //         = channels*kernel_area x channels_out
        {
            const int n = channels*kernel_area;
            const int k = output_area;
            const int m = channels_out;
            SgemmBatched(state, 'n', 'n', n, m, k, alpha,
                (const scalar_t **)column_batched, n,
                (const scalar_t **)grad_batched, k,
                beta,
                dweight_batched, n,
                batch
            );
        };
        THCudaFree(state, input_batched);
        THCudaFree(state, grad_batched);
        THCudaFree(state, weight_batched);
        THCudaFree(state, dinput_batched);
        THCudaFree(state, dcolumn_batched);
        THCudaFree(state, dweight_batched);
    }));
 
    if(has_bias){
        return {dinput, dweight.sum(0), grad.sum({0, 2, 3}), dcoord_offset, dcoord_weight};    
    }
    return {dinput, dweight.sum(0), at::empty(0, input.options()), dcoord_offset, dcoord_weight};
}




// 定义与Python的接口，使用pybind11，注意这里的_dcn名字，必须与so文件名一样
PYBIND11_MODULE(_dcn, m) {
    m.def("conv_forward", &conv_forward, "卷积", 
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("pad_h"), py::arg("pad_w"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("has_bias")
    );

    m.def("conv_backward", &conv_backward, "卷积求导", 
        py::arg("x"),
        py::arg("grad"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("pad_h"), py::arg("pad_w"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("has_bias")
    );

    m.def("deformable_conv_forward", &deformable_conv_forward, "动态卷积", 
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("coord_offset"), py::arg("coord_weight"),
        py::arg("pad_h"), py::arg("pad_w"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("deformable_groups"),
        py::arg("has_bias")
    );

    m.def("deformable_conv_backward", &deformable_conv_backward, "动态卷积求导", 
        py::arg("x"),
        py::arg("grad"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("coord_offset"), py::arg("coord_weight"),
        py::arg("pad_h"), py::arg("pad_w"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("deformable_groups"),
        py::arg("has_bias")
    );
}