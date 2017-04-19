#ifndef GRB_BACKEND_APSPIE_MXM_HPP
#define GRB_BACKEND_APSPIE_MXM_HPP

#include <iostream>

#include <graphblas/backend/apspie/Matrix.hpp>
#include <graphblas/backend/apspie/SparseMatrix.hpp>
#include <graphblas/backend/apspie/DenseMatrix.hpp>
#include <graphblas/backend/apspie/spmm.hpp>
#include <graphblas/types.hpp>

namespace graphblas
{
namespace backend
{
	template <typename c, typename m, typename a, typename b>
  Info mxm( Matrix<c>&        C,
					  const Matrix<m>&  mask,
						const BinaryOp&   accum,
						const Semiring&   op,
						const Matrix<a>&  A,
						const Matrix<b>&  B,
						const Descriptor& desc ); 

  template <typename c, typename a, typename b>
  Info mxm( Matrix<c>&       C,
	  			  const Semiring&  op,
		  			const Matrix<a>& A,
			  		const Matrix<b>& B )
  {
		Storage A_storage, B_storage;
		A.get_storage( A_storage );
		B.get_storage( B_storage );
		// Decision tree:
		// a) Sp x Sp: SpGEMM (TODO)
		// b) Sp x De:   SpMM (DONE) 
		// c) De x Sp:   SpMM (TODO)
		// c) De x De:   GEMM (TODO)
		if( A_storage == Sparse && B_storage == Dense ) {
			C.set_storage( Dense );
			return spmm( C.dense, op, A.sparse, B.dense );
  }}
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MXM_HPP
