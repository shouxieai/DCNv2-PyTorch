# DCN Pytorch扩展的实现
DCNv2重写，学习为目的

## 使用方法
1. 配置Makefile中的pytorch路径、anaconda路径等
2. 执行`make dcn -j6`
3. 切换到workspace `cd workspace`，然后执行测试`python test.py`
```shell
(base) root@c7d0e554c5f3:/datav/shared/100_du/03.23# make dcn -j6
Compile precompile handle src/pre_compile.hpp
Compile src/dcnv2.cu
Compile src/conv.cu
Compile src/interface.cpp
Link workspace/_dcn.so
(base) root@c7d0e554c5f3:/datav/shared/100_du/03.23# cd workspace/
(base) root@c7d0e554c5f3:/datav/shared/100_du/03.23/workspace# python test.py
-------------------------Invoke test_conv2d-------------------------
调用输出为： tensor([[[[-0.9571]]]], device='cuda:0', grad_fn=<Conv2dImplBackward>)
进行梯度检验.....
梯度检验**成功**，使用的参数是：
input.shape = torch.Size([3, 5, 5, 3])
weight.shape = torch.Size([1, 5, 2, 2])
bias.shape = torch.Size([1])
对比torch实现的卷积结果...
---前向推理output的绝对误差： 5.218048215738236e-15
---反向结果dinput绝对误差:  0.0
---反向结果dweight绝对误差:  2.220446049250313e-16
---反向结果dbias绝对误差:  0.0
--------------------------------------------------------------------------------
-------------------------Invoke test_deformable_conv-------------------------
调用输出为： tensor([[[[-0.0573]]]], device='cuda:0', grad_fn=<DeformableConV2ImplBackward>)
进行梯度检验.....
梯度检验**成功**，使用的参数是：
input.shape = torch.Size([3, 5, 5, 3])
weight.shape = torch.Size([1, 5, 2, 2])
bias.shape = torch.Size([1])
coord_offset.shape = torch.Size([3, 16, 5, 3])
coord_weight.shape = torch.Size([3, 8, 5, 3])
--------------------------------------------------------------------------------
(base) root@c7d0e554c5f3:/datav/shared/100_du/03.23/workspace# 
```

# 知识点
1. 关于卷积和可变卷积的代码实现
2. 关于batched sgemm的实现
3. 关于预编译的使用
4. 关于梯度检查的使用
5. 关于卷积/可变卷积实现的方式和细节

# Reference
- [1]. [DCNv2实现，支持PyTorch1.7及以上](https://github.com/lbin/DCNv2)
- [2]. [Dai Deformable Convolutional Networks ICCV2017 paper.pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)
---
* 剩下的代码说明、解释等，慢慢写吧