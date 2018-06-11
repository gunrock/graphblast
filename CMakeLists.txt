# CMakeLists.txt for unittest
 
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
set( PROJ_HEADERS   "" )
set( PROJ_LIBRARIES "" )
set( PROJ_INCLUDES  "./" "ext/moderngpu/include" "$ENV{CUDA_HOME}/samples/common/inc/" )
set( mgpu_SRC_FILES "ext/moderngpu/src/mgpucontext.cu" "ext/moderngpu/src/mgpuutil.cpp")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CUDA_PROPAGATE_HOST_FLAGS OFF )

#set( CUDA_CURAND_LIBRARY "$ENV{CUDA_HOME}/lib64/libcurand.so" )
set( CUDA_CUBLAS_LIBRARY "$ENV{CUDA_HOME}/lib64/libcublas.so" )
set( CUDA_CUSPARSE_LIBRARY "$ENV{CUDA_HOME}/lib64/libcusparse.so" )

#FILE( GLOB_RECURSE PROJ_SOURCES graphblas/*.cu ../graphblas/*.cpp )
#FILE( GLOB_RECURSE PROJ_LIBRARIES ext/cublas1.1/*.cu )
FILE( GLOB_RECURSE PROJ_HEADERS graphblas/*.hpp)

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_35 -lineinfo -O3 -use_fast_math -Xptxas=-v -std=c++11")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-fpermissive;-arch=sm_35;-lineinfo;-Xptxas=-v;-dlcm=ca;-maxrregcount=64)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_21)

# needed for cudamalloc
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fpermissive -pg -m64 -std=c++11" )
#set(CMAKE_CXX_FLAGS "-fpermissive -g -m64 -std=c++11" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
#set(CMAKE_EXE_LINKER_FLAGS "-Xlinker --verbose")

# ptx generation
#include_directories( ${PROJ_INCLUDES} )
#cuda_compile_ptx( cuda_ptx_files "matrix/main.cu" )
#add_custom_target( ptx ALL DEPENDS ${cuda_ptx_files} ${PROJ_HEADERS} "matrix/main.cu") #SOURCES main.cu ${PROJ_HEADERS} )

# standard compilation
#add_library( grb ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
#set_target_properties( grb PROPERTIES LINKER_LANGUAGE CXX )
#add_executable( benchmark_spgemm${PROJ_NAME} "benchmark_spgemm/main.cu" )
#target_link_libraries( spgemm${PROJ_NAME} grb "/usr/local/lib/libboost_unit_test_framework.a" "/usr/local/lib/libboost_program_options.a" ${MKL_LIBRARIES} )
#target_link_libraries( benchmark_spgemm${PROJ_NAME} grb ${Boost_LIBRARIES} ${MKL_LIBRARIES} pthread )
#set_target_properties( benchmark_spgemm${PROJ_NAME} PROPERTIES LINKER_LANGUAGE CXX )

# standard compilation
include_directories( ${PROJ_INCLUDES} ${Boost_INCLUDE_DIRS} )
message( ${Boost_INCLUDE_DIRS} )
message( ${Boost_LIBRARY_DIRS} )
#add_executable(        bmatrix "test/bmatrix.cpp" )
#target_link_libraries( bmatrix ${Boost_LIBRARIES} )

# CUDA compilation
cuda_add_library( grb ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
#cuda_add_executable( gmatrix "test/gmatrix.cu" )
#cuda_add_executable( gmmio "test/gmmio.cu" )
cuda_add_executable( gspmm "test/gspmm.cu" ${mgpu_SRC_FILES} )
#cuda_add_executable( gspgemm "test/gspgemm.cu" )
#cuda_add_executable( gbmatrix "test/gbmatrix.cu" )
#cuda_add_executable( gbspmv "test/gbspmv.cu" ${mgpu_SRC_FILES} )
cuda_add_executable( gbspmm "test/gbspmm.cu" ${mgpu_SRC_FILES} )
cuda_add_executable( gbdense "test/gbdense.cu" ${mgpu_SRC_FILES} )
cuda_add_executable( gbdensecusparse2 "test/gbdensecusparse2.cu" ${mgpu_SRC_FILES} )
cuda_add_executable( gbgemm "test/gbgemm.cu" ${mgpu_SRC_FILES} )
#cuda_add_executable( gbfeatures "test/gbfeatures.cu" ${mgpu_SRC_FILES} )
#cuda_add_executable( gbspgemm "test/gbspgemm.cu" )
#target_link_libraries( gmatrix grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gmmio grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
target_link_libraries( gspmm grb ${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gspgemm grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gbmatrix grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gbspmv grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
target_link_libraries( gbspmm grb ${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${Boost_LIBRARIES} )
target_link_libraries( gbdense grb ${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${Boost_LIBRARIES} )
target_link_libraries( gbdensecusparse2 grb ${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${Boost_LIBRARIES} )
target_link_libraries( gbgemm grb ${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gbfeatures grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gbspgemm grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES})
