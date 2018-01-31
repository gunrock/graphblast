#ifndef GRB_BACKEND_HPP
#define GRB_BACKEND_HPP

#ifdef GRB_USE_SEQUENTIAL
#define __GRB_BACKEND_ROOT sequential
// These defines will allow the same operators to workfor both CPU and GPU
#define GRB_HOST_DEVICE   
#else
  #ifdef GRB_USE_APSPIE
  #define __GRB_BACKEND_ROOT apspie
  #define GRB_HOST_DEVICE    __host__ __device__
  #else
  #pragma message "Error: No GraphBLAS library specified!"
  #endif
#endif

#endif  // GRB_BACKEND_HPP
