# CMakeLists.txt for unittest
 
# required cmakeversion                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA)
find_package(Boost REQUIRED)

set( PROJ_NAME      "test_matrix" )
set( PROJ_PATH      ${CMAKE_SOURCE_DIR})
set( PROJ_OUT_PATH  ${CMAKE_BINARY_DIR})
set( PROJ_SOURCES   "test_matrix2.cpp" )
set( PROJ_HEADERS   "" )
set( PROJ_LIBRARIES "" )
set( PROJ_INCLUDES  "../" )
#set( mgpu_SRC_DIR "../../ext/moderngpu/src")
#set( mgpu_SRC_FILES ${mgpu_SRC_DIR}/mgpucontext.cu ${mgpu_SRC_DIR}/mgpuutil.cpp)
set( MMIO_SRC_FILES "../graphblas/mmio.cpp" )

#set( CUDA_CURAND_LIBRARY "$ENV{CUDA_HOME}/lib64/libcurand.so" )
#set( CUDA_CUBLAS_LIBRARY "$ENV{CUDA_HOME}/lib64/libcublas.so" )
set( CUDA_CUSPARSE_LIBRARY "$ENV{CUDA_HOME}/lib64/libcusparse.so" )

#FILE( GLOB_RECURSE PROJ_SOURCES ../graphblas/*.cpp )
#FILE( GLOB_RECURSE PROJ_LIBRARIES ../../ext/cublas1.1/*.cu )
FILE( GLOB_RECURSE PROJ_HEADERS ../graphblas/*.hpp )

# nvcc flags
set(CUDA_NVCC_FLAGS  ${CUDA_NVCC_FLAGS};-arch=sm_35;-lineinfo;-dlcm=ca)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_21)

# needed for cudamalloc
set(CMAKE_CXX_FLAGS "-fpermissive -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
#set(CMAKE_EXE_LINKER_FLAGS "-Xlinker --verbose")

# ptx generation
#include_directories( ${PROJ_INCLUDES} )
#cuda_compile_ptx( cuda_ptx_files main.cu )
#add_custom_target( ptx ALL DEPENDS ${cuda_ptx_files} ${PROJ_HEADERS} main.cu SOURCES main.cu ${PROJ_HEADERS} )

# standard compilation
#PROJECT( test_matrix C CXX) 
#include_directories( ${PROJ_INCLUDES} )
#add_library( grb ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
#set_target_properties( grb PROPERTIES LINKER_LANGUAGE CXX )
#add_executable( ${PROJ_NAME} ${PROJ_SOURCES} ${MMIO_SRC_FILES} )#${mgpu_SRC_FILES})
#target_link_libraries( ${PROJ_NAME} grb )#${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} )

# standard compilation
include_directories( ${PROJ_INCLUDES} )
cuda_add_library( grb ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
cuda_add_executable( ${PROJ_NAME} ${PROJ_SOURCES} ${MMIO_SRC_FILES} )#${mgpu_SRC_FILES})
target_link_libraries( ${PROJ_NAME} grb )#${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} )
