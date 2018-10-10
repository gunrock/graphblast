#ifndef GRB_BACKEND_APSPIE_GEMM_HPP
#define GRB_BACKEND_APSPIE_GEMM_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

namespace graphblas
{
namespace backend
{

  template <typename c, typename a, typename b, typename m,
            typename BinaryOpT,      typename SemiringT>
  Info gemm( DenseMatrix<c>*        C,
             const Matrix<m>*       mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
             const DenseMatrix<a>*  A,
             const DenseMatrix<b>*  B,
             Descriptor*            desc )
  {
    std::cout << "GEMM\n";
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GrB_BACKEND_APSPIE_GEMM_HPP
