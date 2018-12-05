#ifndef GRAPHBLAS_BACKEND_CUDA_SPMM_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SPMM_HPP_

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include <iostream>

namespace graphblas {
namespace backend {

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,      typename SemiringT>
Info spmm(DenseMatrix<c>*        C,
          const Matrix<m>*       mask,
          const BinaryOpT*       accum,
          const SemiringT*       op,
          const SparseMatrix<a>* A,
          const DenseMatrix<b>*  B,
          Descriptor*            desc) {
  std::cout << "SpMat x DeMat SpMM\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,      typename SemiringT>
Info spmm(DenseMatrix<c>*        C,
          const Matrix<m>*       mask,
          const BinaryOpT*       accum,
          const SemiringT*       op,
          const DenseMatrix<a>*  A,
          const SparseMatrix<b>* B,
          Descriptor*            desc) {
  std::cout << "DeMat x SpMat SpMM\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SPMM_HPP_
