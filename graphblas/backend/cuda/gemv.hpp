#ifndef GRAPHBLAS_BACKEND_CUDA_GEMV_HPP_
#define GRAPHBLAS_BACKEND_CUDA_GEMV_HPP_

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include <iostream>

namespace graphblas {
namespace backend {

template <typename W, typename a, typename U, typename M,
          typename BinaryOpT,      typename SemiringT>
Info gemv(DenseVector<W>*        w,
          const Vector<M>*       mask,
          BinaryOpT              accum,
          SemiringT              op,
          const DenseMatrix<a>*  A,
          const DenseVector<U>*  u,
          Descriptor*            desc) {
  // Set transpose flag to true for A
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename W, typename a, typename U, typename M,
          typename BinaryOpT,      typename SemiringT>
Info gemv(DenseVector<W>*        w,
          const Vector<M>*       mask,
          BinaryOpT              accum,
          SemiringT              op,
          const DenseMatrix<a>*  A,
          const SparseVector<U>* u,
          Descriptor*            desc) {
  // Set transpose flag to true for A
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_GEMV_HPP_
