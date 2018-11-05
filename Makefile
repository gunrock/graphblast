# Makefile

ARCH=-gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60
OPTIONS=-O3 -use_fast_math

all: gbfs gsssp glgc gsssp_simple

gbfs: example/*
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o bin/gbfs \
		example/gbfs.cu \
		ext/moderngpu/src/mgpucontext.cu \
		ext/moderngpu/src/mgpuutil.cpp \
		-Iext/moderngpu/include \
		-Iext/cub/cub \
		-I. \
		-Itest \
		-lboost_program_options \
		-lcublas \
		-lcusparse \
		-lcurand

gsssp: example/*
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o bin/gsssp \
		example/gsssp.cu \
		ext/moderngpu/src/mgpucontext.cu \
		ext/moderngpu/src/mgpuutil.cpp \
		-Iext/moderngpu/include \
		-Iext/cub/cub \
		-I. \
		-Itest \
		-lboost_program_options \
		-lcublas \
		-lcusparse \
		-lcurand

glgc: example/*
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o bin/glgc \
		example/glgc.cu \
		ext/moderngpu/src/mgpucontext.cu \
		ext/moderngpu/src/mgpuutil.cpp \
		-Iext/moderngpu/include \
		-Iext/cub/cub \
		-I. \
		-Itest \
		-lboost_program_options \
		-lcublas \
		-lcusparse \
		-lcurand

gsssp_simple: example/*
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o bin/gsssp_simple \
		example/gsssp_simple.cu \
		ext/moderngpu/src/mgpucontext.cu \
		ext/moderngpu/src/mgpuutil.cpp \
		-Iext/moderngpu/include \
		-Iext/cub/cub \
		-I. \
		-Itest \
		-lboost_program_options \
		-lcublas \
		-lcusparse \
		-lcurand

clean:
	rm -f bin/gbfs
