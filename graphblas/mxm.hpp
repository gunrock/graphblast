#ifndef GRB_MXM_HPP
#define GRB_MXM_HPP

//#include "graphblas/types.hpp"
#include <boost/optional.hpp>

#define __GRB_BACKEND_MXM_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/mxm.hpp>
#include __GRB_BACKEND_MXM_HEADER
#undef __GRB_BACKEND_MXM_HEADER

namespace graphblas
{
  template <typename c, typename m, typename a, typename b>
  Info mxm( Matrix<c>&                         C,
            const boost::optional<Matrix<m>&>  mask = ,
            const boost::optional<BinaryOp&>   accum,
            const boost::optional<Semiring&>   op,
            const Matrix<a>&                   A,
            const Matrix<b>&                   B,
            const boost::optional<Descriptor&> desc = ) 
  {
    return backend::mxm( C.matrix, mask.matrix, accum, op, A.matrix, B.matrix, 
        desc.descriptor );
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
    return backend::mxmCompute( C.matrix, op, A.matrix, B.matrix, TA, TB, NT, 
            ROW_MAJOR );
  }*/

}  // graphblas

#endif  // GRB_MXM_HPP
