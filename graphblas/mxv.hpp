#ifndef GRB_MXV_HPP
#define GRB_MXV_HPP

//#include "graphblas/types.hpp"

#define __GRB_BACKEND_MXV_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/mxv.hpp>
#include __GRB_BACKEND_MXV_HEADER
#undef __GRB_BACKEND_MXV_HEADER

namespace graphblas
{
  template <typename c, typename m, typename a, typename b>
  Info mxv( Matrix<c>&       C,
            const Matrix<m>& mask,
            const BinaryOp&  accum,
            const Semiring&  op,
            const Matrix<a>& A,
            const Matrix<b>& B,
            Descriptor&      desc ) 
  {
    return backend::mxv( C.matrix_, mask, accum, op, A.matrix_, 
        B.matrix_, desc.descriptor_ );
  }

  template <typename c, typename a, typename b>
  Info mxv( Matrix<c>&       C,
            const int        mask,
            const int        accum,
            const Semiring&  op,
            const Matrix<a>& A,
            const Matrix<b>& B,
            Descriptor&      desc )
  {
    Matrix<c> GrB_NULL_MATRIX;
    BinaryOp  GrB_NULL_BINARYOP = 0;
    return backend::mxv<c,c,a,b>( C.matrix_, GrB_NULL_MATRIX.matrix_, 
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

#endif  // GRB_MXV_HPP
