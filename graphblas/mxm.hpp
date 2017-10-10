#ifndef GRB_MXM_HPP
#define GRB_MXM_HPP

//#include "graphblas/types.hpp"

#define __GRB_BACKEND_MXM_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/mxm.hpp>
#include __GRB_BACKEND_MXM_HEADER
#undef __GRB_BACKEND_MXM_HEADER

namespace graphblas
{
  template <typename c, typename m, typename a, typename b>
  Info mxm( Matrix<c>&       C,
            const Matrix<m>& mask,
            const BinaryOp&  accum,
            const Semiring&  op,
            const Matrix<a>& A,
            const Matrix<b>& B,
            Descriptor&      desc ) 
  {
    return backend::mxm( C.matrix_, mask, accum, op, A.matrix_, B.matrix_, 
        desc.descriptor_ );
  }

  template <typename c, typename a, typename b>
  Info mxm( Matrix<c>&       C,
            const int        mask,
            const int        accum,
            const Semiring&  op,
            const Matrix<a>& A,
            const Matrix<b>& B,
            Descriptor&      desc )
  {
    Matrix<c> GrB_NULL_MATRIX;
    BinaryOp  GrB_NULL_BINARYOP = 0;
    return backend::mxm<c,c,a,b>( C.matrix_, GrB_NULL_MATRIX.matrix_, 
        GrB_NULL_BINARYOP, op, A.matrix_, B.matrix_, desc.descriptor_ );
  }

  /*// For testing
  template <typename c, typename a, typename b>
  Info mxmCompute( Matrix<c>&       C,
            const Semiring&  op,
            const Matrix<a>& A,
            const Matrix<b>& B,
            const int TA,
            const int TB,
            const int NT,
            const bool ROW_MAJOR )
  {
    return backend::mxmCompute( C.matrix_, op, A.matrix_, B.matrix_, TA, TB, NT, 
            ROW_MAJOR );
  }*/

}  // graphblas

#endif  // GRB_MXM_HPP
