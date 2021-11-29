
import math
import torch

from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import _dcn as dcn_native_impl


'''
定义DCN的自动微分模块
对接C++的实现
'''
class DeformableConV2Impl(Function):

    @staticmethod
    def forward(ctx, 
        input, weight, bias, coord_offset, coord_weight,
        padding, stride, dilation, deformable_groups, has_bias
    ):

        if not has_bias:
            # 如果没有bias，则需要提供一个空的tensor，使得类型匹配
            bias = input.new_tensor([])
        
        ctx.stride      = _pair(stride)
        ctx.padding     = _pair(padding)
        ctx.dilation    = _pair(dilation)
        ctx.has_bias    = has_bias
        ctx.kernel_size = weight.shape[2:4]
        ctx.deformable_groups = deformable_groups

        # 告诉torch，这几个tensor需要保存到反向时候使用
        ctx.save_for_backward(input, weight, bias, coord_offset, coord_weight)

        # 调用c++实现的推理接口
        return dcn_native_impl.deformable_conv_forward(
            input, weight, bias, coord_offset, coord_weight,
            pad_h=ctx.padding[0],       pad_w=ctx.padding[1],
            stride_h=ctx.stride[0],     stride_w=ctx.stride[1],
            dilation_h=ctx.dilation[0], dilation_w=ctx.dilation[1],
            deformable_groups=deformable_groups, has_bias=has_bias
        )


    # once_differentiable是告诉梯度检查器，这个只能执行一次反向
    # 因为他依赖输入的值
    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        
        # forward时保存的tensor中解包出来几个tensor变量
        input, weight, bias, coord_offset, coord_weight = ctx.saved_tensors

        dinput, dweight, dbias, dcoord_offset, dcoord_weight = dcn_native_impl.deformable_conv_backward(
            input, grad, weight, bias, coord_offset, coord_weight,
            pad_h=ctx.padding[0],       pad_w=ctx.padding[1], 
            stride_h=ctx.stride[0],     stride_w=ctx.stride[1], 
            dilation_h=ctx.dilation[0], dilation_w=ctx.dilation[1],
            deformable_groups=ctx.deformable_groups, has_bias=ctx.has_bias
        )

        # 如果没有偏置，则设置为空
        if not ctx.has_bias:
            dbias = None

        # 对于输入为：
        # input, weight, bias, coord_offset, coord_weight,
        # stride, padding, dilation, deformable_groups, has_bias
        # 返回他们每一项的导数，而标量不指定导数所以是None
        return dinput, dweight, dbias, dcoord_offset, dcoord_weight, None, None, None, None, None


'''
定义DCN的nn模块
'''
class DeformableConV2(nn.Module):

    def __init__(self, 
        in_feature, out_feature, kernel_size, 
        padding=0, stride=1, dilation=1, deformable_groups=1, bias=True
    ):
        super().__init__()

        self.in_feature     = in_feature
        self.out_feature    = out_feature
        self.padding        = _pair(padding)
        self.stride         = _pair(stride)
        self.dilation       = _pair(dilation)
        self.kernel_size    = _pair(kernel_size)
        self.has_bias       = bias
        self.deformable_groups = deformable_groups
        self.weight         = nn.Parameter(
            torch.empty(out_feature, in_feature, self.kernel_size[0], self.kernel_size[1])
        )
        
        # 如果有偏置，则创建，否则注册一个空的参数，使得forward时可以取到None
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_feature))
        else:
            self.register_parameter("bias", None)

        self.deformableConV2Invoke = DeformableConV2Impl.apply
        self.reset_parameters()


    def reset_parameters(self):

        n = self.in_feature
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.zero_()


    def forward(self, input, coord_offset, coord_weight):

        # 检查参数是否合理
        except_offset_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        except_weight_channels = self.deformable_groups * 1 * self.kernel_size[0] * self.kernel_size[1]

        assert except_offset_channels == coord_offset.shape[1], f"Wrong channels, Except offset channels = {except_offset_channels}, coord_offset shape[1] = {coord_offset.shape[1]}"
        assert except_weight_channels == coord_weight.shape[1], f"Wrong channels, Except weight channels = {except_weight_channels}, coord_weight shape[1] = {coord_weight.shape[1]}"

        assert coord_offset.shape[2] == input.shape[2], f"The input and coord_offset height do not match {coord_offset.shape[2]} != {input.shape[2]}"
        assert coord_offset.shape[3] == input.shape[3], f"The input and coord_offset width do not match {coord_offset.shape[3]} != {input.shape[3]}"

        assert coord_weight.shape[2] == input.shape[2], f"The input and coord_weight height do not match {coord_weight.shape[2]} != {input.shape[2]}"
        assert coord_weight.shape[3] == input.shape[3], f"The input and coord_weight width do not match {coord_weight.shape[3]} != {input.shape[3]}"

        return self.deformableConV2Invoke(
            input, self.weight, self.bias, coord_offset, coord_weight,
            self.padding, self.stride, self.dilation, self.deformable_groups, self.has_bias
        )



'''
定义DCN的nn模块
'''
class DeformableConv(nn.Module):

    def __init__(self, 
        in_feature, out_feature, kernel_size, 
        padding=0, stride=1, dilation=1, deformable_groups=1, bias=True
    ):
        super().__init__()
        
        kernel_size = _pair(kernel_size)

        # 定义DCN模块
        self.deformable_module = DeformableConV2(in_feature, out_feature, kernel_size, padding, stride, dilation, deformable_groups, bias)

        # 对于offset和weight，他需要padding为一样大小
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # 定义偏移和权重的卷积
        self.offset_and_weight = nn.Conv2d(
            in_feature, 
            deformable_groups * 3 * kernel_size[0] * kernel_size[1], kernel_size, 
            padding=padding, 
            stride=stride, 
            dilation=dilation, 
            bias=True
        )

        # 对偏移和权重的参数进行设置
        self.reset_parameters()


    def reset_parameters(self):
        # 默认偏移和权重的参数设置为0
        self.offset_and_weight.weight.data.zero_()
        self.offset_and_weight.bias.data.zero_()


    def forward(self, input):

        offset_and_weight = self.offset_and_weight(input)
        coord_offset_y, coord_offset_x, coord_weight = torch.chunk(offset_and_weight, 3, dim=1)
        coord_weight = torch.sigmoid(coord_weight)
        coord_offset = torch.cat((coord_offset_y, coord_offset_x), dim=1)
        return self.deformable_module(input, coord_offset, coord_weight)




'''
定义卷积的实现，自动微分模块
对接的是C++实现的底层模块
'''
class Conv2dImpl(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, padding, stride, dilation, has_bias):

        if not has_bias:
            # 如果没有bias，则需要提供一个空的tensor，使得类型匹配
            bias = input.new_tensor([])

        ctx.padding     = _pair(padding)
        ctx.stride      = _pair(stride)
        ctx.dilation    = _pair(dilation)
        ctx.has_bias    = has_bias
        ctx.save_for_backward(input, weight, bias)
        
        # 调用C++提供的功能
        return dcn_native_impl.conv_forward(
            input, weight, bias,
            pad_h=padding[0],       pad_w=padding[1], 
            stride_h=stride[0],     stride_w=stride[1], 
            dilation_h=dilation[0], dilation_w=dilation[1], 
            has_bias=has_bias
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):

        input, weight, bias = ctx.saved_tensors
        dinput, dweight, dbias = dcn_native_impl.conv_backward(input, grad, weight, bias,
            pad_h=ctx.padding[0],       pad_w=ctx.padding[1], 
            stride_h=ctx.stride[0],     stride_w=ctx.stride[1], 
            dilation_h=ctx.dilation[0], dilation_w=ctx.dilation[1],
            has_bias=ctx.has_bias
        )

        if not ctx.has_bias:
            # 如果没提供bias，也就不用有导数了
            dbias = None

        # 对于输入为：
        # input, weight, bias, pad, stride, dilation, has_bias
        # 返回对应的导数，而标量我们不给于导数，所以是None 
        return dinput, dweight, dbias, None, None, None, None


'''
定义卷积的nn模块
'''
class Conv2d(nn.Module):

    def __init__(self, 
        in_feature, out_feature, kernel_size, 
        padding=0, stride=1, dilation=1, bias=True
    ):
        super().__init__()

        self.in_feature     = in_feature
        self.out_feature    = out_feature
        self.padding        = _pair(padding)
        self.stride         = _pair(stride)
        self.dilation       = _pair(dilation)
        self.kernel_size    = _pair(kernel_size)
        self.has_bias       = bias
        self.weight         = nn.Parameter(
            torch.empty(out_feature, in_feature, self.kernel_size[0], self.kernel_size[1])
        )
        
        # 如果有偏置，则创建，否则注册一个空的参数，使得forward时可以取到None
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_feature))
        else:
            self.register_parameter("bias", None)

        self.conv2dInvoke = Conv2dImpl.apply
        self.reset_parameters()


    def reset_parameters(self):

        n = self.in_feature
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias:
            self.bias.data.zero_()


    def forward(self, input):
        return self.conv2dInvoke(
            input, self.weight, self.bias,
            self.padding, self.stride, self.dilation, self.has_bias
        )