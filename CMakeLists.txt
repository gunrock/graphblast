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
set( PROJ_INCLUDES  "./" "ext/moderngpu/include" "ext/cub/cub")
set( mgpu_SRC_FILES "ext/moderngpu/src/mgpucontext.cu" "ext/moderngpu/src/mgpuutil.cpp")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

#set( CUDA_CURAND_LIBRARY "$ENV{CUDA_HOME}/lib64/libcurand.so" )
#set( CUDA_CUBLAS_LIBRARY "$ENV{CUDA_HOME}/lib64/libcublas.so" )
set( CUDA_CUSPARSE_LIBRARY "$ENV{CUDA_HOME}/lib64/libcusparse.so" )

#FILE( GLOB_RECURSE PROJ_SOURCES graphblas/*.cu ../graphblas/*.cpp )
#FILE( GLOB_RECURSE PROJ_LIBRARIES ext/cublas1.1/*.cu )
FILE( GLOB_RECURSE PROJ_HEADERS graphblas/*.hpp)

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_35 -lineinfo -O3 -use_fast_math -Xptxas=-v")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-fpermissive;-arch=sm_35;-lineinfo;-Xptxas=-v;-dlcm=ca;-maxrregcount=64)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_21)

# needed for cudamalloc
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fpermissive -g -m64 -std=c++11" )
#set(CMAKE_CXX_FLAGS "-fpermissive -pg -m64 -std=c++11" )
#set(CMAKE_CXX_FLAGS "-fpermissive -g -m64 -std=c++11 -H" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
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
message( ${Boost_INCLUDE_DIRS} )
message( ${Boost_LIBRARY_DIRS} )
include_directories( ${PROJ_INCLUDES} ${Boost_INCLUDE_DIRS} )
cuda_add_library( grb ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
#cuda_add_executable( gmmio         "test/gmmio.cu" )
#cuda_add_executable( gutil        "test/gutil.cu" ) x
cuda_add_executable( gspmspv       "test/gspmspv.cu"       ${mgpu_SRC_FILES} )
cuda_add_executable( gpull         "test/gpull.cu"         ${mgpu_SRC_FILES} )
cuda_add_executable( gpush         "test/gpush.cu"         ${mgpu_SRC_FILES} )
cuda_add_executable( gassign       "test/gassign.cu"       ${mgpu_SRC_FILES} )
cuda_add_executable( gbfs          "test/gbfs.cu"          ${mgpu_SRC_FILES} )
cuda_add_executable( gbfs2         "test/gbfs2.cu"         ${mgpu_SRC_FILES} )
cuda_add_executable( gvector       "test/gvector.cu"       ${mgpu_SRC_FILES} )
cuda_add_executable( gdensevector  "test/gdensevector.cu"  ${mgpu_SRC_FILES} )
cuda_add_executable( gsparsevector "test/gsparsevector.cu" ${mgpu_SRC_FILES} )
cuda_add_executable( gdescriptor   "test/gdescriptor.cu"   ${mgpu_SRC_FILES} )
#cuda_add_executable( gbinaryop     "test/gbinaryop.cu" ) x
cuda_add_executable( gsparsematrix "test/gsparsematrix.cu" )
#cuda_add_executable( gmatrix "test/gmatrix.cu" ) x
#cuda_add_executable( gspmm   "test/gspmm.cu" ${mgpu_SRC_FILES} ) x
#cuda_add_executable( gspgemm "test/gspgemm.cu" ) x
#cuda_add_executable( gbspmm "test/gbspmm.cu" ${mgpu_SRC_FILES} ) x
#cuda_add_executable( gbspgemm "test/gbspgemm.cu" ) x
#target_link_libraries( gmmio         grb ${Boost_LIBRARIES} )
#target_link_libraries( gutil        grb ${Boost_LIBRARIES} )
target_link_libraries( gspmspv       grb ${Boost_LIBRARIES} )
target_link_libraries( gpull         grb ${Boost_LIBRARIES} )
target_link_libraries( gpush         grb ${Boost_LIBRARIES} )
target_link_libraries( gassign       grb ${Boost_LIBRARIES} )
target_link_libraries( gbfs          grb ${Boost_LIBRARIES} )
target_link_libraries( gbfs2         grb ${Boost_LIBRARIES} )
target_link_libraries( gvector       grb ${Boost_LIBRARIES} )
target_link_libraries( gdensevector  grb ${Boost_LIBRARIES} )
target_link_libraries( gsparsevector grb ${Boost_LIBRARIES} )
target_link_libraries( gdescriptor   grb ${Boost_LIBRARIES} )
#target_link_libraries( gbinaryop     grb ${Boost_LIBRARIES} )
target_link_libraries( gsparsematrix grb ${Boost_LIBRARIES} )
#target_link_libraries( gspmm   grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gspgemm grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gbspmm grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES} )
#target_link_libraries( gbspgemm grb ${CUDA_CUSPARSE_LIBRARY} ${Boost_LIBRARIES})
