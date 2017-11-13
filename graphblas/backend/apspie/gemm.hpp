#ifndef GRB_BACKEND_APSPIE_GEMM_HPP
#define GRB_BACKEND_APSPIE_GEMM_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"

namespace graphblas
{
namespace backend
{

  template <int variant, typename c, typename a, typename b, typename m,
            typename BinaryOpT,      typename SemiringT>
  Info gemm( DenseMatrix<c>*        C,
             const SparseMatrix<m>* mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
             const DenseMatrix<a>*  A,
             const DenseMatrix<b>*  B,
             const Descriptor*      desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GrB_BACKEND_APSPIE_GEMM_HPP
