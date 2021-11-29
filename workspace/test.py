
import torch
import dcn_v2 as dcn

from torch import nn

# 梯度的自动校验，用雅克比方法进行多次测试对比实现
from torch.autograd import gradcheck

# _pair，把标量转为tuple，如果是tuple则返回tuple
from torch.nn.modules.utils import _pair



# 定义一个美化格式的装饰器
def test(func):

    def call(*args):
        # 打印分隔符
        print("-" * 25 + f"Invoke {func.__name__}" + "-" * 25)
        ret = func(*args)

        # 打印结束的分隔符
        print("-" * 80)
        return ret
    return call


'''
实现卷积功能的测试和梯度检验
同时对比与官方的差距
'''
@test
def test_conv2d():

    m = dcn.Conv2d(in_feature=1, out_feature=1, kernel_size=3).to(device)
    input = torch.ones(1, 1, 3, 3, device=device, dtype=torch.float32)
    print("调用输出为：", m(input))


    ###############################################################
    # 梯度检查，为了检查所有梯度所以需要定义内部类
    print("进行梯度检验.....")
    class InnerClassConv2d(nn.Module):
    
        def __init__(self, 
            in_feature, out_feature, 
            padding=0, stride=1, dilation=1, bias=True
        ):
            super().__init__()

            self.in_feature     = in_feature
            self.out_feature    = out_feature
            self.padding        = _pair(padding)
            self.stride         = _pair(stride)
            self.dilation       = _pair(dilation)
            self.has_bias       = bias
            
            # 如果有偏置，则创建，否则注册一个空的参数，使得forward时可以取到None
            self.conv2dInvoke = dcn.Conv2dImpl.apply


        def forward(self, input, weight, bias):
            return self.conv2dInvoke(
                input, weight, bias,
                self.padding, self.stride, self.dilation, self.has_bias
            )


    # 定义检查的基本参数，梯度检查需要用double，否则会因为精度问题而校验失败
    dtype = torch.double
    in_feature  = 5
    out_feature = 1
    kernel_size = 2
    stride      = 2
    dilation    = 1
    padding     = 10
    batch       = 3
    in_w        = 3
    in_h        = 5
    has_bias    = True

    # 定义输入的tensor，所有tensor都需要梯度，目的是校验模块的正确性
    input           = torch.randn(batch, in_feature, in_h, in_w, device=device, dtype=dtype, requires_grad=True)
    weight          = torch.randn(out_feature, in_feature, kernel_size, kernel_size, requires_grad=True, dtype=dtype, device=device)
    bias            = torch.randn(out_feature, requires_grad=True, dtype=dtype, device=device)

    # 使用内部类建立模块
    m = InnerClassConv2d(in_feature, out_feature, padding, stride, dilation, has_bias)
    
    # 执行梯度校验
    finished = gradcheck(m, (input, weight, bias))
    #finished = gradcheck(nn.functional.conv2d, (input, weight, bias, padding, stride, dilation))

    # 如果执行到这里说明校验成功，否则会抛出异常
    if finished:
        print(
            "梯度检验**成功**，使用的参数是：\n"
            f"input.shape = {input.shape}\n"
            f"weight.shape = {weight.shape}\n"
            f"bias.shape = {bias.shape}"
        )


    print("对比torch实现的卷积结果...")
    # 计算自定义的卷积以及梯度，并储存梯度用来打印
    custom_output = m(input, weight, bias)
    grad = torch.ones_like(custom_output)
    custom_output.backward(grad)
    custom_dinput   = input.grad
    custom_dweight  = weight.grad
    custom_dbias    = bias.grad

    # 清空梯度，为之后的计算准备
    input.grad  = None
    weight.grad = None
    bias.grad   = None

    # 计算torch版本的卷积
    torch_output = nn.functional.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation)
    grad = torch.ones_like(torch_output)
    torch_output.backward(grad)
    torch_dinput    = input.grad
    torch_dweight   = weight.grad
    torch_dbias     = bias.grad

    loss = lambda a,b : torch.sum(torch.abs(a - b)).item()
    print("---前向推理output的绝对误差：",  loss(custom_output, torch_output))
    print("---反向结果dinput绝对误差: ",    loss(custom_dinput, torch_dinput))
    print("---反向结果dweight绝对误差: ",   loss(custom_dweight, torch_dweight))
    print("---反向结果dbias绝对误差: ",     loss(custom_dbias, torch_dbias))
    

'''
实现可变卷积功能的测试和梯度检验
'''
@test
def test_deformable_conv():
    m = dcn.DeformableConv(in_feature=1, out_feature=1, kernel_size=3).to(device)
    input = torch.ones(1, 1, 3, 3, device=device, dtype=torch.float32)
    
    print("调用输出为：", m(input))


    ###############################################################
    # 梯度检查，为了检查所有梯度所以需要定义内部类
    print("进行梯度检验.....")

    class InnerClassDeformableConV2(nn.Module):
        
        def __init__(self, 
            in_feature, out_feature, 
            padding=0, stride=1, dilation=1, deformable_groups=1, bias=True
        ):
            super().__init__()

            self.in_feature     = in_feature
            self.out_feature    = out_feature
            self.padding        = _pair(padding)
            self.stride         = _pair(stride)
            self.dilation       = _pair(dilation)
            self.has_bias       = bias
            self.deformable_groups = deformable_groups
            self.deformableConV2Invoke = dcn.DeformableConV2Impl.apply

        def forward(self, input, weight, bias, coord_offset, coord_weight):

            return self.deformableConV2Invoke(
                input, weight, bias, coord_offset, coord_weight,
                self.padding, self.stride, self.dilation, self.deformable_groups, self.has_bias
            )

    
    # 定义检查的基本参数，梯度检查需要用double，否则会因为精度问题而校验失败
    dtype = torch.double
    in_feature  = 5
    out_feature = 1
    kernel_size = 2
    stride      = 2
    dilation    = 1
    padding     = 1
    batch       = 3
    in_w        = 3
    in_h        = 5
    has_bias    = True
    deformable_groups = 2

    # 定义输入的tensor，所有tensor都需要梯度，目的是校验模块的正确性
    input           = torch.randn(batch, in_feature, in_h, in_w, device=device, dtype=dtype, requires_grad=True)
    weight          = torch.randn(out_feature, in_feature, kernel_size, kernel_size, requires_grad=True, dtype=dtype, device=device)
    bias            = torch.randn(out_feature, requires_grad=True, dtype=dtype, device=device)
    coord_offset    = torch.randn(batch, deformable_groups * 2 * kernel_size * kernel_size, in_h, in_w, requires_grad=True, dtype=dtype, device=device)
    coord_weight    = torch.randn(batch, deformable_groups * 1 * kernel_size * kernel_size, in_h, in_w, requires_grad=True, dtype=dtype, device=device).sigmoid()

    # 使用内部类建立模块
    m = InnerClassDeformableConV2(in_feature, out_feature, padding, stride, dilation, deformable_groups, has_bias)

    # 执行梯度校验
    finished = gradcheck(m, (
        input, weight, bias, coord_offset, coord_weight
    ))

    # 如果执行到这里说明校验成功，否则会抛出异常
    if finished:
        print(
            "梯度检验**成功**，使用的参数是：\n"
            f"input.shape = {input.shape}\n"
            f"weight.shape = {weight.shape}\n"
            f"bias.shape = {bias.shape}\n"
            f"coord_offset.shape = {coord_offset.shape}\n"
            f"coord_weight.shape = {coord_weight.shape}"
        )


if __name__ == "__main__":

    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)

    test_conv2d()
    test_deformable_conv()
    # d = dcn.DeformableConV2(3, 3, 3, 1).cuda()
    # d.weight.data.fill_(1)
    # d.bias.data.fill_(0.5)

    # print(d.weight.data.size())
    # print(d.bias.data.size())

    # input = torch.full((1, 3, 3, 3), 1).cuda()
    # offset_and_mask = torch.full((1, 27, 3, 3), 1).cuda()

    # coord_offset_y, coord_offset_x, coord_weight = torch.chunk(offset_and_mask, 3, dim=1)
    # coord_offset = torch.cat((coord_offset_y, coord_offset_x), dim=1)
    # coord_weight = torch.sigmoid(coord_weight)
    # print(coord_weight)

    # output = d(input, coord_offset, coord_weight)
    # print(output)