CUDA_ARCH = 70
ARCH = -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}
OPTIONS = -O3 -use_fast_math -w -std=c++11

MGPU_DIR = ext/moderngpu/include/
CUB_DIR = ext/cub/cub/
GRB_DIR = $(dir $(lastword $(MAKEFILE_LIST)))

GRB_DEPS = ext/moderngpu/src/mgpucontext.cu \
					 ext/moderngpu/src/mgpuutil.cpp

LIBS = -lboost_program_options \
			 -lcublas \
			 -lcusparse \
			 -lcurand
