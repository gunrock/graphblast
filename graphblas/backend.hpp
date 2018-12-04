#ifndef GRAPHBLAS_BACKEND_HPP_
#define GRAPHBLAS_BACKEND_HPP_

#ifdef GRB_USE_SEQUENTIAL
#define __GRB_BACKEND_ROOT sequential
// These defines will allow the same operators to workfor both CPU and GPU
#define GRB_HOST_DEVICE
#else
  #ifdef GRB_USE_CUDA
  #define __GRB_BACKEND_ROOT cuda
  #define GRB_HOST_DEVICE __host__ __device__
  #else
  #pragma message "Error: No GraphBLAS library specified!"
  #endif
#endif

#endif  // GRAPHBLAS_BACKEND_HPP_
