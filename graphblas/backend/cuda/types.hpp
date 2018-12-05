#ifndef GRAPHBLAS_BACKEND_CUDA_TYPES_HPP_
#define GRAPHBLAS_BACKEND_CUDA_TYPES_HPP_

namespace graphblas {
namespace backend {

enum SparseMatrixFormat {
  GrB_SPARSE_MATRIX_CSRCSC,
  GrB_SPARSE_MATRIX_CSRONLY,
  GrB_SPARSE_MATRIX_CSCONLY
};

enum LoadBalanceMode {
  GrB_LOAD_BALANCE_SIMPLE,
  GrB_LOAD_BALANCE_TWC,
  GrB_LOAD_BALANCE_MERGE
};
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_TYPES_HPP_
