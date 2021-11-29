

cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

# 由于cpp可能与cu同名，但是不同文件
# 比如开心做的
# 因此，对于cuda的程序，把cu改成cuo
cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(patsubst %.cu,%.cuo,$(cu_srcs))
cu_objs := $(subst src/,objs/,$(cu_objs))


# 定义名称参数
workspace := workspace
binary := pro
ext_name := _dcn
sbinary := $(ext_name).so
pch_file := src/pre_compile.hpp
gch_file := src/pre_compile.hpp.gch

anaconda_root 	:= /data/datav/newbb/lean/anaconda3/envs/torch1.5
cuda_root 		:= /usr/local/cuda-10.2
pytorch_sdk 	:= $(anaconda_root)/lib/python3.7/site-packages/torch

# 这里定义头文件库文件和链接目标没有加-I -L -l，后面用foreach一次性增加 
include_paths := $(cuda_root)/include \
				 $(pytorch_sdk)/include/torch/csrc/api/include \
				 $(pytorch_sdk)/include \
				 $(pytorch_sdk)/include/TH \
				 $(pytorch_sdk)/include/THC \
				 $(anaconda_root)/include/python3.7m \
				 src \
				 /data/sxai/lean/opencv4.2.0/include/opencv4/

# 这里需要清楚的认识链接的库到底链接是谁，这个非常重要
# 要求链接对象一定是预期的
library_paths := $(cuda_root)/lib64 \
				 $(anaconda_root)/lib \
				 $(pytorch_sdk)/lib \
				 /data/sxai/lean/opencv4.2.0/lib/

link_librarys := cudart cublas opencv_core opencv_imgcodecs opencv_imgproc \
				c10_cuda c10 caffe2_nvrtc torch_cpu torch_cuda torch torch_python python3.7m

# 定义编译选项,  -w屏蔽警告
# compute_75,code=sm_75是针对RTX2080Ti显卡，如果其他显卡请修改
# pytorch依赖的，需要添加-std=c++14
# 如果使用opencv，则需要用-D_GLIBCXX_USE_CXX11_ABI=0的方式编译opencv，才可以加入项目中使用，否则会报错cxx11::std之类的错误，这是标准库的问题
# 关于-D的几个选项，来自setup.py，pytorch的cuda扩展编译中命令行提取出来的，可以自行提取
# 	-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ 
# 	-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ 
# 	-DTORCH_API_INCLUDE_EXTENSION_H
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++14 -w -fopenmp -Wsign-compare -fwrapv -Wall \
				 -D_GLIBCXX_USE_CXX11_ABI=0 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(ext_name) \
				 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__  \
				 -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
				 -DTORCH_API_INCLUDE_EXTENSION_H
				 
cu_compile_flags  := -m64 -Xcompiler -fPIC -g -G -O0 -std=c++14 -w -Xcompiler -fopenmp -gencode=arch=compute_75,code=sm_75 \
				 -D_GLIBCXX_USE_CXX11_ABI=0 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(ext_name) \
				 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
				 --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ \
				 -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H

#-as-needed 就是忽略链接时没有用到的动态库
#--no-as-needed 就是不忽略链接时没有用到的动态库
link_flags        := -Wl,--no-as-needed

# 2种类型
# 1. 字符串
# 2. 字符串数组
# 空格隔开就是数组
#
# 对头文件、库文件、目标统一增加-I -L -l
# foreach var,list,cmd
#     var  = item
#     list = link_librarys
#     cmd  = -Wl,-rpath=$(item)
#
# output = []
# for item in link_librarys:
#     output.append(f"-Wl,-rpath={item}")
# rpaths = output
#
# -L  指定链接目标时查找的目录
# -l  指定链接的目标名称，符合libname.so， -lname 规则
# -I  指定编译时头文件查找目录
rpaths        := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))


# 合并选项
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(link_flags) $(rpaths) $(library_paths) $(link_librarys)



# 定义cpp的编译方式
# $@   生成项
# $<   依赖项第一个
# $^   依赖项所有
# $?  $+
# -include是使用预编译头文件，避免头文件重复编译造成的耗时
objs/%.o : src/%.cpp $(gch_file)
	@mkdir -p $(dir $@)
	@echo Compile $<
	@g++ -c $< -o $@ $(cpp_compile_flags) -include $(pch_file)


# 定义cuda文件的编译方式
objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $<
	@nvcc -c $< -o $@ $(cu_compile_flags)


$(gch_file) : $(pch_file)
	@mkdir -p $(dir $@)
	@echo Compile precompile handle $<
	@g++ $< -o $@ $(cpp_compile_flags)


# 定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ $^ -o $@ $(link_flags)

# 定义链接workspace/sb.so
$(workspace)/$(sbinary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ -shared $^ -o $@ $(link_flags)

# 定义pro快捷编译指令，这里只发生编译，不执行
pro : $(workspace)/$(binary)

#@strip $<

dcn  : $(workspace)/$(sbinary)

# 定义编译并执行的指令，并且执行目录切换到workspace下
run : dcn
	@cd $(workspace) && python test.py

debug :
	@echo $(cpp_objs)
	@echo $(cu_objs)

clean :
	@rm -rf objs $(workspace)/$(binary) $(workspace)/$(sbinary) src/pre_compile.hpp.gch

# 指定伪标签，作为指令
.PHONY : clean debug run pro