#ifndef GRAPHBLAS_BACKEND_CUDA_GEMM_HPP_
#define GRAPHBLAS_BACKEND_CUDA_GEMM_HPP_

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include <iostream>

namespace graphblas {
namespace backend {

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,      typename SemiringT>
Info gemm(DenseMatrix<c>*        C,
          const Matrix<m>*       mask,
          const BinaryOpT*       accum,
          const SemiringT*       op,
          const DenseMatrix<a>*  A,
          const DenseMatrix<b>*  B,
          Descriptor*            desc) {
  std::cout << "GEMM\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_GEMM_HPP_
