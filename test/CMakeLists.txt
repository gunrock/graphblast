# CMakeLists.txt for unittest
#PROJECT( graphblas C CXX )
 
# required cmakeversion                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA)
find_package(Boost COMPONENTS program_options REQUIRED)
#find_package(OpenMP REQUIRED)

#set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "." )
#find_package(MKL REQUIRED)

set( PROJ_NAME      "/test" )
set( PROJ_PATH      ${CMAKE_SOURCE_DIR})
set( PROJ_OUT_PATH  ${CMAKE_BINARY_DIR})
#set( PROJ_SOURCES   ".cpp" )
set( PROJ_HEADERS   "" )
set( PROJ_LIBRARIES "" )
set( PROJ_INCLUDES  "../" )
#set( mgpu_SRC_DIR "../../ext/moderngpu/src")
#set( mgpu_SRC_FILES ${mgpu_SRC_DIR}/mgpucontext.cu ${mgpu_SRC_DIR}/mgpuutil.cpp)

#set( CUDA_CURAND_LIBRARY "$ENV{CUDA_HOME}/lib64/libcurand.so" )
#set( CUDA_CUBLAS_LIBRARY "$ENV{CUDA_HOME}/lib64/libcublas.so" )
set( CUDA_CUSPARSE_LIBRARY "$ENV{CUDA_HOME}/lib64/libcusparse.so" )

#FILE( GLOB_RECURSE PROJ_SOURCES ../graphblas/*.cu ../graphblas/*.cpp )
#FILE( GLOB_RECURSE PROJ_LIBRARIES ../../ext/cublas1.1/*.cu )
FILE( GLOB_RECURSE PROJ_HEADERS ../graphblas/*.hpp ../graphblas/*.cuh)

# nvcc flags
set(CUDA_NVCC_FLAGS -arch=sm_35;-lineinfo;-O3;-use_fast_math;-lineinfo;-Xptxas=-v)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-fpermissive;-arch=sm_35;-lineinfo;-Xptxas=-v;-dlcm=ca;-maxrregcount=64)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_21)

# needed for cudamalloc
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fpermissive -std=c++11 -m64" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/lib/libiomp5 -liomp5" )
#set(CMAKE_EXE_LINKER_FLAGS "-lboost_unit_test_framework -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_gnu_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
#set(CMAKE_EXE_LINKER_FLAGS "-Xlinker --verbose")

# ptx generation
#include_directories( ${PROJ_INCLUDES} )
#cuda_compile_ptx( cuda_ptx_files "matrix/main.cu" )
#add_custom_target( ptx ALL DEPENDS ${cuda_ptx_files} ${PROJ_HEADERS} "matrix/main.cu") #SOURCES main.cu ${PROJ_HEADERS} )

# standard compilation
#include_directories( ${PROJ_INCLUDES} ${Boost_INCLUDE_DIRS} ${MKL_INCLUDE_DIR} )
#add_library( grb ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
#set_target_properties( grb PROPERTIES LINKER_LANGUAGE CXX )
#add_executable( benchmark_spgemm${PROJ_NAME} "benchmark_spgemm/main.cu" )
#target_link_libraries( spgemm${PROJ_NAME} grb "/usr/local/lib/libboost_unit_test_framework.a" "/usr/local/lib/libboost_program_options.a" ${MKL_LIBRARIES} )
#target_link_libraries( benchmark_spgemm${PROJ_NAME} grb ${Boost_LIBRARIES} ${MKL_LIBRARIES} pthread )
#set_target_properties( benchmark_spgemm${PROJ_NAME} PROPERTIES LINKER_LANGUAGE CXX )

# standard compilation
include_directories( ${PROJ_INCLUDES} ${Boost_INCLUDE_DIRS} $ENV{MKLROOT} ${OpenMP_INCLUDE_PATH} )
cuda_add_library( grb ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
#cuda_add_executable( matrix${PROJ_NAME} "matrix/main.cu" )#${mgpu_SRC_FILES}
#cuda_add_executable( mmio${PROJ_NAME} "mmio/main.cu" )#${mgpu_SRC_FILES}
#cuda_add_executable( spmm${PROJ_NAME} "spmm/main.cu" )#${mgpu_SRC_FILES}
#cuda_add_executable( spgemm${PROJ_NAME} "spgemm/main.cu" )#${mgpu_SRC_FILES}
#cuda_add_executable( benchmark_spmm${PROJ_NAME} "benchmark_spmm/main.cu" )#${mgpu_SRC_FILES}
cuda_add_executable( benchmark_spgemm${PROJ_NAME} "benchmark_spgemm/main.cu" )#${mgpu_SRC_FILES}
#target_link_libraries( matrix${PROJ_NAME} grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( mmio${PROJ_NAME} grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( spmm${PROJ_NAME} grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( spgemm${PROJ_NAME} grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} ${MKL_LIBRARIES} )
#target_link_libraries( benchmark_spmm${PROJ_NAME} grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
target_link_libraries( benchmark_spgemm${PROJ_NAME} grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
