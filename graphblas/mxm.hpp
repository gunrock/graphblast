#ifndef GRB_MXM_HPP
#define GRB_MXM_HPP

#include <graphblas/types.hpp>

#define __GRB_BACKEND_MXM_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/mxm.hpp>
#include __GRB_BACKEND_MXM_HEADER
#undef __GRB_BACKEND_MXM_HEADER

namespace graphblas
{
	template <typename c, typename m, typename a, typename b,
					 MatrixType c_spar, MatrixType m_spar, 
					 MatrixType a_spar, MatrixType b_spar>
  Info mxm( Matrix<c,c_spar>&        C,
					  const Matrix<m,m_spar>&  mask,
						const BinaryOp&   accum,
						const Semiring&   op,
						const Matrix<a,a_spar>&  A,
						const Matrix<b,b_spar>&  B,
						const Descriptor& desc ) 
	{
		return backend::mxm( C.matrix, mask, accum, op, A.matrix, B.matrix, desc );
	}

	template <typename c, typename a, typename b, 
					 MatrixType c_spar, MatrixType a_spar, MatrixType b_spar>
	Info mxm( Matrix<c,c_spar>&       C,
					  const Semiring&  op,
						const Matrix<a,a_spar>& A,
						const Matrix<b,b_spar>& B )
	{
    return backend::mxm( C.matrix, op, A.matrix, B.matrix );
	}

}  // graphblas

#endif  // GRB_MXM_HPP
