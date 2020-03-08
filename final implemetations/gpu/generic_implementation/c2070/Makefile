################################################################################
# Build script for kmeans
################################################################################
# Add path where NVIDIA SDK is located
NVIDIA_SDK_PATH	:= /usr/local/cuda/NVIDIA_GPU_Computing_SDK
# Add source files here
EXECUTABLE	:= example
# CUDA source files (compiled with cudacc)
CUFILES_sm_20	:= example.cu \
	kmeans_clustering.cu \

# Additional flags to nvcc
# For debugging add these flags in CUDACCFLAGS: -g -G

CUDACCFLAGS		:= --ptxas-options=-v
# CUDA dependency files
CU_DEPS_sm_20		:= \
	kmeans_clustering.cu \

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	file_io.cpp \

# common.mk by default will always add sm_10 and sm_20 unless overridden by CUFILES
GENCODE_ARCH    := -gencode=arch=compute_20,code=\"sm_20,compute_20\"

#To show the various commands used for compilation set verbose to 1
verbose    := 1
################################################################################
# Rules and targets

ROOTDIR := $(NVIDIA_SDK_PATH)/C/common

BINDIR := ./bin/linux

include $(NVIDIA_SDK_PATH)/C/common/common.mk

#Remember to export following before running make
#export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/../lib:/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/lib/linux

