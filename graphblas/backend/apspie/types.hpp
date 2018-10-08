#ifndef GRB_BACKEND_APSPIE_TYPES_HPP
#define GRB_BACKEND_APSPIE_TYPES_HPP

namespace graphblas
{
namespace backend
{
  enum SparseMatrixFormat
  {
    GrB_SPARSE_MATRIX_CSRCSC,
		GrB_SPARSE_MATRIX_CSRONLY,
		GrB_SPARSE_MATRIX_CSCONLY
  };

  enum LoadBalanceMode
  {
    GrB_LOAD_BALANCE_SIMPLE,
    GrB_LOAD_BALANCE_LB
  };
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_TYPES_HPP
