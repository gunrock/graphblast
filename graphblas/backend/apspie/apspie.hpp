#ifndef SBP_BACKEND_APSPIE_APSPIE_HPP
#define SBP_BACKEND_APSPIE_APSPIE_HPP

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                               \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA(call) do {                                       \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
  } } while (0)

#include "graphblas/backend/apspie/Vector.hpp"
#include "graphblas/backend/apspie/Matrix.hpp"
#include "graphblas/backend/apspie/mxm.hpp"
#include "graphblas/backend/apspie/transpose.hpp"
#include "graphblas/backend/apspie/reduce.hpp"
#include "graphblas/backend/apspie/util.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"

#endif  // SBP_BACKEND_APSPIE_APSPIE_HPP
